"""Service `backend` — orchestrateur (SQUELETTE À COMPLÉTER).

Rôle attendu : exposé au navigateur (via le frontend nginx), il valide
l'entrée avec le **même schéma Pydantic** que le modèle, appelle le service
`model` en interne (`http://model:8000/predict`), et expose `/health`,
`/score`, `/metrics`.

👉 Inspirez-vous du service `model` (déjà fourni) pour le pattern `/metrics`
   et le middleware de logging. Mini-cours : `02_FastAPI_metrics_Prometheus`.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from loguru import logger
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from app.middleware import LoggingMiddleware
from app.registry import UserRegistryClient, UserRegistryError
from app.schemas import (
    EmploymentApplication,
    Feedback,
    HealthResponse,
    Prediction,
    TrainRequest,
    TrainResponse,
)

# URL du service model — configurable par variable d'env (dev/staging/prod)
MODEL_URL = os.environ.get("MODEL_URL", "http://model:8000")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8088").split(",")
ABSTENTION_THRESHOLD = float(os.environ.get("ABSTENTION_THRESHOLD", "0.55"))
CLASS_2_ESCALATION_THRESHOLD = float(
    os.environ.get("CLASS_2_ESCALATION_THRESHOLD", "0.04")
)
DRIFT_METRICS_FILE = Path(
    os.environ.get("DRIFT_METRICS_FILE", "/app/reports/drift.prom")
)
EVALUATION_METRICS_FILE = Path(
    os.environ.get("EVALUATION_METRICS_FILE", "/app/reports/evaluation.prom")
)
RETRAIN_METRICS_FILE = Path(
    os.environ.get("RETRAIN_METRICS_FILE", "/app/reports/retrain.prom")
)
HISTORY_EXCLUDED_INPUT_FIELDS = {
    "usager_id",
    "session_id",
    "age",
    "nationalite_hors_ue",
}


def _feedback_db_path() -> Path:
    return Path(
        os.environ.get("FEEDBACK_DB", str(Path.cwd() / "data" / "feedbacks.db"))
    )


def _init_feedback_db() -> None:
    """Create the durable prediction ledger and feedback store."""
    db_path = _feedback_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                request_id TEXT PRIMARY KEY,
                usager_id TEXT,
                session_id TEXT,
                input_json TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                probabilities_json TEXT NOT NULL,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                needs_human_review INTEGER NOT NULL DEFAULT 0,
                review_reasons_json TEXT NOT NULL DEFAULT '[]'
            )"""
        )
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(predictions)")
        }
        for column in ("usager_id", "session_id"):
            if column not in columns:
                connection.execute(f"ALTER TABLE predictions ADD COLUMN {column} TEXT")
        if "needs_human_review" not in columns:
            connection.execute(
                "ALTER TABLE predictions ADD COLUMN needs_human_review INTEGER NOT NULL DEFAULT 0"
            )
        if "review_reasons_json" not in columns:
            connection.execute(
                "ALTER TABLE predictions ADD COLUMN review_reasons_json TEXT NOT NULL DEFAULT '[]'"
            )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS feedbacks (
                request_id TEXT PRIMARY KEY,
                prediction INTEGER NOT NULL,
                true_label INTEGER NOT NULL,
                comments TEXT,
                created_at TEXT NOT NULL,
                used_for_training INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(request_id) REFERENCES predictions(request_id)
            )"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS request_id_sequence (
                id INTEGER PRIMARY KEY AUTOINCREMENT
            )"""
        )


def _next_request_id() -> str:
    """Allocate a persistent identifier for one scoring request."""
    _init_feedback_db()
    with sqlite3.connect(_feedback_db_path()) as connection:
        cursor = connection.execute("INSERT INTO request_id_sequence DEFAULT VALUES")
        request_number = int(cursor.lastrowid)
        request_id = f"REQ{request_number:05d}"
        while connection.execute(
            "SELECT 1 FROM predictions WHERE request_id = ?", (request_id,)
        ).fetchone():
            cursor = connection.execute("INSERT INTO request_id_sequence DEFAULT VALUES")
            request_number = int(cursor.lastrowid)
            request_id = f"REQ{request_number:05d}"
        return request_id


_init_feedback_db()

app = FastAPI(title="Retour-Emploi Backend Orchestrator", version="1.0.0")
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-ID"],
)


# TODO 1 — exposer /metrics avec prometheus-fastapi-instrumentator
#   (cf. service model). Pensez à une métrique métier : compteur d'erreurs
#   upstream lors de l'appel au model.
# Expose /metrics (latence, RPS, codes retour) + métrique métier sur les
# erreurs remontées par le service model lors de l'appel upstream.
# Métriques métier custom
MODEL_UPSTREAM_ERRORS_TOTAL = Counter(
    "backend_model_upstream_errors_total",
    "Nombre d'erreurs remontées par le service model lors d'un appel /score ou /train.",
    labelnames=("kind",),
)
# Buckets fins sur la plage attendue (appel interne réseau, quelques ms à ~1s)
# pour obtenir des p50/p95/p99 précis via histogram_quantile en Grafana.
MODEL_CALL_DURATION_SECONDS = Histogram(
    "backend_model_call_duration_seconds",
    "Durée de l'appel HTTP interne vers le service model (hors traitement backend).",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0),
)
BACKEND_PREDICTIONS_TOTAL = Counter(
    "backend_predictions_total",
    "Sorties du modèle comptées par classe, abstentions comprises.",
    labelnames=("predicted_class", "model_version"),
)
BACKEND_ABSTENTIONS_TOTAL = Counter(
    "backend_abstentions_total",
    "Dossiers orientés vers une revue humaine après le scoring.",
)
BACKEND_PREDICTION_PROBA = Histogram(
    "backend_prediction_proba",
    "Distribution des probabilités de défaut renvoyées au client (dérive du comportement du modèle).",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
# Métriques HTTP automatiques + endpoint /metrics
Instrumentator(should_group_status_codes=False).instrument(app).expose(
    app, endpoint="/metrics", include_in_schema=False
)


@app.get("/drift-metrics", include_in_schema=False)
async def drift_metrics() -> PlainTextResponse:
    """Serve the latest batch drift report in Prometheus text format."""
    if not DRIFT_METRICS_FILE.is_file():
        return PlainTextResponse("", media_type="text/plain; version=0.0.4")
    return PlainTextResponse(
        DRIFT_METRICS_FILE.read_text(encoding="utf-8"),
        media_type="text/plain; version=0.0.4",
    )


@app.get("/evaluation-metrics", include_in_schema=False)
async def evaluation_metrics() -> PlainTextResponse:
    """Serve the latest release-gate evaluation for Prometheus."""
    if not EVALUATION_METRICS_FILE.is_file():
        return PlainTextResponse("", media_type="text/plain; version=0.0.4")
    return PlainTextResponse(
        EVALUATION_METRICS_FILE.read_text(encoding="utf-8"),
        media_type="text/plain; version=0.0.4",
    )


@app.get("/retrain-metrics", include_in_schema=False)
async def retrain_metrics() -> PlainTextResponse:
    """Serve the latest local retrainer result for Prometheus."""
    if not RETRAIN_METRICS_FILE.is_file():
        return PlainTextResponse("", media_type="text/plain; version=0.0.4")
    return PlainTextResponse(
        RETRAIN_METRICS_FILE.read_text(encoding="utf-8"),
        media_type="text/plain; version=0.0.4",
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness du backend (ne dépend PAS du model)."""
    return HealthResponse(status="ok")


def _human_review_reasons(
    predicted_class: int,
    probabilities: dict[str, float],
) -> list[str]:
    reasons = []
    max_probability = max(float(value) for value in probabilities.values())
    class_2_probability = float(probabilities.get("2", 0.0))
    if max_probability < ABSTENTION_THRESHOLD:
        reasons.append("low_confidence")
    if predicted_class == 0 and class_2_probability >= CLASS_2_ESCALATION_THRESHOLD:
        reasons.append("class_2_risk")
    return reasons



# TODO 2 — route POST /score :
#   - reçoit une EmploymentApplication (validée par Pydantic),
#   - appelle MODEL_URL/predict en interne (httpx async),
#   - propage le header X-Request-ID,
#   - gère les erreurs : model injoignable → 503, model en erreur → 502,
#   - retourne un objet Prediction.
#
# @app.post("/score", response_model=Prediction)
# async def score(application: EmploymentApplication, request: Request) -> Prediction:
#     ...
@app.post("/score", response_model=Prediction, status_code=status.HTTP_200_OK)
async def score(application: EmploymentApplication, request: Request) -> Prediction:
    """Valide la demande, l'envoie au modèle et renvoie le résultat."""
    trace_id = getattr(request.state, "request_id", request.headers.get("X-Request-ID", "n/a"))
    session_id = application.session_id or request.headers.get("X-Session-ID") or str(uuid4())

    if application.usager_id and os.environ.get("USER_REGISTRY_URL"):
        try:
            await UserRegistryClient().ensure_user_exists(application.usager_id)
        except UserRegistryError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"User registry unavailable: {exc}",
            ) from exc

    request_id = _next_request_id()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            with MODEL_CALL_DURATION_SECONDS.time():
                response = await client.post(
                    f"{MODEL_URL.rstrip('/')}/predict",
                    json=application.model_dump(exclude={"usager_id", "session_id"}),
                    headers={"X-Request-ID": trace_id, "X-Session-ID": session_id},
                )
    except httpx.RequestError as exc:
        MODEL_UPSTREAM_ERRORS_TOTAL.labels(kind="unreachable").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service unavailable",
        ) from exc

    if response.status_code >= 400:
        MODEL_UPSTREAM_ERRORS_TOTAL.labels(kind="bad_status").inc()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Model error {response.status_code}: {response.text[:200]}",
        )

    try:
        payload = response.json()
    except ValueError as exc:  # pragma: no cover - protection défensive
        MODEL_UPSTREAM_ERRORS_TOTAL.labels(kind="invalid_json").inc()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Invalid JSON returned by model",
        ) from exc

    payload["request_id"] = request_id
    payload["usager_id"] = application.usager_id
    payload["session_id"] = session_id

    try:
        prediction = Prediction(**payload)
    except Exception as exc:
        MODEL_UPSTREAM_ERRORS_TOTAL.labels(kind="schema_mismatch").inc()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Model response schema mismatch",
        ) from exc

    review_reasons = _human_review_reasons(
        prediction.prediction,
        prediction.probabilities,
    )
    prediction.needs_human_review = bool(review_reasons)
    prediction.review_reasons = review_reasons
    BACKEND_PREDICTIONS_TOTAL.labels(
        predicted_class=str(prediction.prediction),
        model_version=prediction.model_version,
    ).inc()
    if prediction.needs_human_review:
        BACKEND_ABSTENTIONS_TOTAL.inc()
        logger.bind(
            request_id=prediction.request_id,
            review_reasons=review_reasons,
            max_probability=max(prediction.probabilities.values()),
            class_2_probability=prediction.probabilities.get("2", 0.0),
        ).warning("Prediction routed for human review")
    BACKEND_PREDICTION_PROBA.observe(
        prediction.probabilities[str(prediction.prediction)]
    )
    _init_feedback_db()
    with sqlite3.connect(_feedback_db_path()) as connection:
        connection.execute(
            """INSERT OR IGNORE INTO predictions
            (request_id, usager_id, session_id, input_json, prediction,
             probabilities_json, model_version, created_at, needs_human_review,
             review_reasons_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction.request_id,
                application.usager_id,
                session_id,
                json.dumps(
                    application.model_dump(exclude=HISTORY_EXCLUDED_INPUT_FIELDS),
                    ensure_ascii=False,
                ),
                prediction.prediction,
                json.dumps(prediction.probabilities),
                prediction.model_version,
                datetime.now(timezone.utc).isoformat(),
                int(prediction.needs_human_review),
                json.dumps(review_reasons),
            ),
        )
    return prediction


@app.get("/history")
async def history(
    session_id: str | None = None,
    usager_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict[str, object]]:
    """Return recent predictions for a counselor session or user."""
    _init_feedback_db()
    clauses: list[str] = []
    parameters: list[object] = []
    if session_id:
        clauses.append("p.session_id = ?")
        parameters.append(session_id)
    if usager_id:
        clauses.append("p.usager_id = ?")
        parameters.append(usager_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with sqlite3.connect(_feedback_db_path()) as connection:
        rows = connection.execute(
            f"""SELECT p.request_id, p.usager_id, p.session_id, p.prediction,
                p.probabilities_json, p.model_version, p.created_at, f.true_label,
                p.input_json, f.comments, p.needs_human_review, p.review_reasons_json
                FROM predictions AS p
                LEFT JOIN feedbacks AS f ON f.request_id = p.request_id
                {where} ORDER BY p.created_at DESC LIMIT ?""",
            [*parameters, limit],
        ).fetchall()
    return [
        {
            "request_id": row[0],
            "usager_id": row[1],
            "session_id": row[2],
            "prediction": row[3],
            "probabilities": json.loads(row[4]),
            "model_version": row[5],
            "created_at": row[6],
            "true_label": row[7],
            "inputs": {
                key: value
                for key, value in json.loads(row[8]).items()
                if key not in HISTORY_EXCLUDED_INPUT_FIELDS
            },
            "comments": row[9],
            "needs_human_review": bool(row[10]),
            "review_reasons": json.loads(row[11]),
        }
        for row in rows
    ]


@app.post("/feedback", status_code=status.HTTP_201_CREATED)
async def feedback(item: Feedback) -> dict[str, str]:
    """Store a labelled outcome for a prediction already served by /score."""
    _init_feedback_db()
    with sqlite3.connect(_feedback_db_path()) as connection:
        existing_prediction = connection.execute(
            "SELECT prediction FROM predictions WHERE request_id = ?",
            (item.request_id,),
        ).fetchone()
        if existing_prediction is None:
            raise HTTPException(status_code=404, detail="Unknown request_id")
        if existing_prediction[0] != item.prediction:
            raise HTTPException(status_code=409, detail="Prediction does not match request_id")

        existing_feedback = connection.execute(
            "SELECT prediction, true_label FROM feedbacks WHERE request_id = ?",
            (item.request_id,),
        ).fetchone()
        if existing_feedback is not None:
            if existing_feedback == (item.prediction, item.true_label):
                return {"status": "already_registered", "request_id": item.request_id}
            raise HTTPException(status_code=409, detail="Contradictory feedback")

        connection.execute(
            """INSERT INTO feedbacks
            (request_id, prediction, true_label, comments, created_at)
            VALUES (?, ?, ?, ?, ?)""",
            (
                item.request_id,
                item.prediction,
                item.true_label,
                item.comments,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    return {"status": "stored", "request_id": item.request_id}


@app.get("/feedback/count")
async def feedback_count() -> dict[str, int]:
    """Return total and not-yet-consumed annotations for the retrain trigger."""
    _init_feedback_db()
    with sqlite3.connect(_feedback_db_path()) as connection:
        total = connection.execute("SELECT COUNT(*) FROM feedbacks").fetchone()[0]
        new = connection.execute(
            "SELECT COUNT(*) FROM feedbacks WHERE used_for_training = 0"
        ).fetchone()[0]
    return {"count": int(total), "new": int(new)}


@app.post("/train", response_model=TrainResponse, status_code=status.HTTP_200_OK)
async def train(request_data: TrainRequest, request: Request) -> TrainResponse:
    """Proxy contrôlé vers le service de réentraînement du modèle."""
    request_id = getattr(request.state, "request_id", request.headers.get("X-Request-ID", "n/a"))
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{MODEL_URL.rstrip('/')}/train",
                json=request_data.model_dump(),
                headers={
                    "X-Request-ID": request_id,
                    "X-Train-Token": request.headers.get("X-Train-Token", ""),
                },
            )
    except httpx.RequestError as exc:
        MODEL_UPSTREAM_ERRORS_TOTAL.labels(kind="unreachable").inc()
        raise HTTPException(status_code=503, detail="Model service unavailable") from exc

    if response.status_code >= 400:
        MODEL_UPSTREAM_ERRORS_TOTAL.labels(kind="training_error").inc()
        raise HTTPException(status_code=502, detail=response.text[:200])

    try:
        return TrainResponse(**response.json())
    except Exception as exc:
        MODEL_UPSTREAM_ERRORS_TOTAL.labels(kind="invalid_training_response").inc()
        raise HTTPException(
            status_code=502,
            detail="Invalid training response from model",
        ) from exc
