"""Service `model` — API de scoring Retour-Emploi (fourni — votre exemple de référence).

Reprise de l'API M1-B2 (routes `/health`, `/info`, `/predict`) + ajout de
l'endpoint `/metrics` Prometheus (latence/RPS/erreurs via instrumentator +
métriques métier custom). Service **interne** : il est appelé par le
`backend`, jamais directement par le navigateur — donc pas de CORS ici.
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from app.metrics import MODEL_INFO, observe_prediction
from app.middleware import LoggingMiddleware
from app.schemas import (
    EmploymentApplication,
    HealthResponse,
    InfoResponse,
    Prediction,
    TrainRequest,
    TrainResponse,
)
from preprocess import create_features

# --- Loguru -----------------------------------------------------------------

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True)
logger.add(
    LOGS_DIR / "api.log",
    rotation="10 MB",
    retention="7 days",
    serialize=True,
    enqueue=True,
    level="INFO",
)

# --- Lifespan ---------------------------------------------------------------

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_ARTIFACT",
        str(
            MODELS_DIR
            / "cisia_emploi_xgboost_multimodal_ethique_best_class_2_ethique_calibrated.joblib"
        ),
    )
)
META_PATH = MODEL_PATH.with_suffix(".json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle + métadonnées au démarrage, libère à l'arrêt."""
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise RuntimeError(f"Model artifacts missing in {MODELS_DIR}")
    app.state.model = joblib.load(MODEL_PATH)
    app.state.metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    MODEL_INFO.labels(
        model_name=app.state.metadata["model_name"],
        model_version=app.state.metadata["model_version"],
        scenario=app.state.metadata.get("scenario_name", "multimodal_ethique"),
        calibration_method=app.state.metadata.get("calibration", {}).get(
            "method", "none"
        ),
    ).set(1)
    logger.info(
        "Model loaded: {name} {version}",
        name=app.state.metadata["model_name"],
        version=app.state.metadata["model_version"],
    )
    yield
    app.state.model = None
    logger.info("Model released")


app = FastAPI(
    title="Retour-Emploi Model Service",
    version="2.1.0",
    description="Service interne de scoring Retour-Emploi.",
    lifespan=lifespan,
)
app.add_middleware(LoggingMiddleware)

# Expose /metrics (latence, RPS, codes retour). should_group_status_codes=False
# pour distinguer 422 (validation) de 500 (erreur modèle) dans Grafana.
Instrumentator(should_group_status_codes=False).instrument(app).expose(
    app, endpoint="/metrics", include_in_schema=False
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness : 503 si le modèle n'est pas chargé."""
    if not hasattr(app.state, "model") or app.state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )
    return HealthResponse(status="ok")


@app.get("/info", response_model=InfoResponse)
async def info() -> InfoResponse:
    """Métadonnées du modèle chargé."""
    meta = app.state.metadata
    return InfoResponse(
        api_version=app.version,
        model_name=meta["model_name"],
        model_version=meta["model_version"],
        model_created_at=meta.get("created_at_utc", ""),
        metrics_holdout=meta.get("evaluation", {}).get("metrics"),
        sklearn_version=meta.get("versions", {}).get("scikit_learn"),
        dataset_sha256=meta.get("dataset", {}).get("sha256"),
        model_artifact=meta.get("artifacts", {}).get("model_path"),
        calibration_method=meta.get("calibration", {}).get("method"),
        decision_policy=meta.get("decision_policy"),
    )


@app.post("/predict", response_model=Prediction, status_code=status.HTTP_200_OK)
async def predict(application: EmploymentApplication, request: Request) -> Prediction:
    """Prédit le délai de retour à l'emploi avec le pipeline CISIA."""
    request_id = getattr(request.state, "request_id", "n/a")
    try:
        X = pd.DataFrame([application.model_dump()])
        X = create_features(X)
        probabilities = app.state.model.predict_proba(X)[0]
        pred = int(app.state.model.classes_[int(probabilities.argmax())])
    except Exception as exc:
        logger.bind(request_id=request_id).exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc.__class__.__name__}",
        ) from exc

    class_labels = {
        0: "Retour rapide",
        1: "Retour moyen",
        2: "Risque de chômage longue durée",
    }
    observe_prediction(
        predicted_class=pred,
        proba_default=float(probabilities[pred]),
    )
    return Prediction(
        prediction=pred,
        prediction_label=class_labels[pred],
        probabilities={
            str(index): round(float(value), 4)
            for index, value in enumerate(probabilities)
        },
        model_version=app.state.metadata["model_version"],
        request_id=request_id,
    )


@app.post("/train", response_model=TrainResponse, status_code=status.HTTP_200_OK)
async def train(request_data: TrainRequest, request: Request) -> TrainResponse:
    """Réentraîne une copie du modèle et la rend active après validation."""
    request_id = getattr(request.state, "request_id", "n/a")
    expected_token = os.environ.get("TRAIN_API_TOKEN")
    provided_token = request.headers.get("X-Train-Token")
    if expected_token and provided_token != expected_token:
        logger.bind(request_id=request_id).warning("Training rejected")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Training forbidden")
    if os.environ.get("ALLOW_DIRECT_TRAIN", "false").lower() not in {"1", "true", "yes"}:
        logger.bind(request_id=request_id).warning("Direct training disabled")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Direct training is disabled; use the batch retraining and promotion workflow",
        )

    try:
        records = [record.model_dump() for record in request_data.records]
        frame = create_features(pd.DataFrame(records))
        target = frame.pop("classe_retour_emploi")
        active_model = app.state.model
        base_estimator = getattr(active_model, "estimator", active_model)
        candidate = clone(base_estimator)
        if base_estimator is not active_model:
            candidate = CalibratedClassifierCV(
                candidate,
                method=active_model.method,
                cv=3,
            )

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment(request_data.experiment_name)
        with mlflow.start_run() as run:
            candidate.fit(frame, target)
            predictions = candidate.predict(frame)
            mlflow.log_param("rows", len(frame))
            mlflow.log_metric("train_accuracy", float(accuracy_score(target, predictions)))
            mlflow.log_metric("train_f1_macro", float(f1_score(target, predictions, average="macro")))
            mlflow.log_metric(
                "train_precision_macro",
                float(precision_score(target, predictions, average="macro", zero_division=0)),
            )
            mlflow.log_metric(
                "train_recall_macro",
                float(recall_score(target, predictions, average="macro", zero_division=0)),
            )
            mlflow.log_metric(
                "train_f1_classe_2",
                float(
                    f1_score(
                        target,
                        predictions,
                        labels=[2],
                        average=None,
                        zero_division=0,
                    )[0]
                ),
            )
            mlflow.set_tag("request_id", request_id)
            mlflow.set_tag("model_type", type(candidate).__name__)
            run_id = run.info.run_id

        app.state.model = candidate
        app.state.metadata["model_version"] = f"trained-{run_id[:8]}"
        logger.bind(request_id=request_id, rows=len(frame), mlflow_run_id=run_id).info(
            "Model trained successfully"
        )
        return TrainResponse(
            status="trained",
            rows=len(frame),
            model_version=app.state.metadata["model_version"],
            mlflow_run_id=run_id,
        )
    except Exception as exc:
        logger.bind(request_id=request_id).exception("Training failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {exc.__class__.__name__}",
        ) from exc
