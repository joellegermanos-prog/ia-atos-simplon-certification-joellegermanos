import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import joblib
import pandas as pd
import pytest

for module_name in ("app.main", "app.middleware", "app.schemas", "app"):
    sys.modules.pop(module_name, None)
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import main as backend_main
from app.main import app
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_SERVICE_ROOT = REPO_ROOT / "services" / "model"
if str(MODEL_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_SERVICE_ROOT))

create_features = importlib.import_module("preprocess").create_features

VALID_APPLICATION = {
    "niveau_diplome": "Bac+2",
    "anciennete_poste_ans": 3.0,
    "code_rome_vise": "M1805",
    "code_insee_commune": "75056",
    "synthese_entretien": "Recherche un emploi stable.",
}


def test_score_calls_model_and_returns_prediction(monkeypatch, tmp_path):
    monkeypatch.setenv("FEEDBACK_DB", str(tmp_path / "feedbacks.db"))

    async def fake_post(self, url, json, headers=None):
        assert url.endswith("/predict")
        assert headers["X-Request-ID"] in {"req-123", "req-124"}
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "prediction": 1,
                "prediction_label": "Retour moyen",
                "probabilities": {"0": 0.20, "1": 0.42, "2": 0.38},
                "model_version": "v1.2.3",
                "request_id": headers["X-Request-ID"],
            },
            text="ok",
        )

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)

    client = TestClient(app)
    application = {**VALID_APPLICATION, "session_id": "session-123"}
    response = client.post("/score", json=application, headers={"X-Request-ID": "req-123"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] == 1
    assert payload["probabilities"]["1"] == 0.42
    assert payload["needs_human_review"] is True
    assert payload["review_reasons"] == ["low_confidence"]
    assert payload["request_id"] == "REQ00001"
    assert payload["session_id"] == "session-123"

    second_response = client.post(
        "/score", json=application, headers={"X-Request-ID": "req-124"}
    )
    assert second_response.status_code == 200
    assert second_response.json()["request_id"] == "REQ00002"
    assert second_response.json()["session_id"] == "session-123"

    history = client.get("/history?session_id=session-123")
    assert {entry["request_id"] for entry in history.json()} == {"REQ00001", "REQ00002"}
    assert all(entry["needs_human_review"] for entry in history.json())
    assert "backend_abstentions_total" in client.get("/metrics").text


def test_score_matches_notebook_decisions_on_calibrated_holdout(
    monkeypatch, tmp_path
):
    model_path = (
        MODEL_SERVICE_ROOT
        / "models"
        / "cisia_emploi_xgboost_multimodal_ethique_best_class_2_ethique_calibrated.joblib"
    )
    metadata = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    model = joblib.load(model_path)
    dataset = pd.read_csv(
        REPO_ROOT / "data" / "dataset_trajectoire_emploi.csv",
        dtype={"code_rome_vise": "string", "code_insee_commune": "string"},
    )
    holdout = dataset.loc[metadata["dataset"]["test_indices"]]
    notebook_decisions = {
        293: 1,
        1756: 0,
        839: "à revoir",
        1537: 2,
    }

    assert backend_main.ABSTENTION_THRESHOLD == 0.55
    assert backend_main.CLASS_2_ESCALATION_THRESHOLD == 0.04
    assert backend_main.ENABLE_CLASS_2_ESCALATION is True
    assert set(notebook_decisions).issubset(set(holdout.index))

    class_two_index = list(model.classes_).index(2)
    examples = []
    for index, expected_decision in notebook_decisions.items():
        row = holdout.loc[index]
        payload = {
            "niveau_diplome": str(row["niveau_diplome"]),
            "anciennete_poste_ans": float(row["anciennete_poste_ans"]),
            "code_rome_vise": str(row["code_rome_vise"]),
            "code_insee_commune": str(row["code_insee_commune"]),
            "est_allocataire": int(row["est_allocataire"]),
            "synthese_entretien": (
                "" if pd.isna(row["synthese_entretien"]) else str(row["synthese_entretien"])
            ),
        }
        probabilities = model.predict_proba(create_features(pd.DataFrame([payload])))[0]
        prediction = int(model.classes_[int(probabilities.argmax())])
        review = (
            float(probabilities.max()) < 0.55
            or (
                prediction == 0
                and float(probabilities[class_two_index]) >= 0.04
            )
        )
        decision = "à revoir" if review else prediction
        assert decision == expected_decision
        examples.append((payload, expected_decision))

    async def calibrated_model_post(self, url, json, headers=None):
        assert url.endswith("/predict")
        probabilities = model.predict_proba(
            create_features(pd.DataFrame([json]))
        )[0]
        prediction = int(model.classes_[int(probabilities.argmax())])
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "prediction": prediction,
                "prediction_label": {
                    0: "Retour rapide",
                    1: "Retour moyen",
                    2: "Risque de chômage longue durée",
                }[prediction],
                "probabilities": {
                    str(class_index): round(float(value), 4)
                    for class_index, value in enumerate(probabilities)
                },
                "model_version": metadata["model_version"],
                "request_id": headers["X-Request-ID"],
            },
            text="ok",
        )

    monkeypatch.setenv("FEEDBACK_DB", str(tmp_path / "feedbacks.db"))
    monkeypatch.setattr("httpx.AsyncClient.post", calibrated_model_post)
    client = TestClient(app)

    for payload, notebook_decision in examples:
        response = client.post("/score", json=payload)

        assert response.status_code == 200
        result = response.json()
        api_decision = (
            "à revoir" if result["needs_human_review"] else result["prediction"]
        )
        assert api_decision == notebook_decision


def test_score_persists_session_and_history(monkeypatch, tmp_path):
    monkeypatch.setenv("FEEDBACK_DB", str(tmp_path / "feedbacks.db"))

    async def fake_post(self, url, json, headers=None):
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "prediction": 1,
                "prediction_label": "Retour moyen",
                "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1},
                "model_version": "v-history",
                "request_id": headers["X-Request-ID"],
            },
            text="ok",
        )

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    client = TestClient(app)
    response = client.post(
        "/score",
        json={
            **VALID_APPLICATION,
            "usager_id": "user-history",
            "session_id": "session-history",
        },
    )

    assert response.status_code == 200
    assert response.json()["session_id"] == "session-history"
    history = client.get("/history?session_id=session-history")
    assert history.status_code == 200
    assert history.json()[0]["usager_id"] == "user-history"
    assert history.json()[0]["needs_human_review"] is False


def test_score_escalates_class_2_risk(monkeypatch, tmp_path):
    monkeypatch.setenv("FEEDBACK_DB", str(tmp_path / "feedbacks.db"))

    async def fake_post(self, url, json, headers=None):
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "prediction": 0,
                "prediction_label": "Retour rapide",
                "probabilities": {"0": 0.80, "1": 0.15, "2": 0.05},
                "model_version": "v1.2.3",
                "request_id": headers["X-Request-ID"],
            },
            text="ok",
        )

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    response = TestClient(app).post("/score", json=VALID_APPLICATION)

    assert response.status_code == 200
    assert response.json()["needs_human_review"] is True
    assert response.json()["review_reasons"] == ["class_2_risk"]


def test_score_can_disable_class_2_escalation(monkeypatch, tmp_path):
    monkeypatch.setenv("FEEDBACK_DB", str(tmp_path / "feedbacks.db"))
    monkeypatch.setattr(backend_main, "ENABLE_CLASS_2_ESCALATION", True)

    async def fake_post(self, url, json, headers=None):
        assert "class_2_escalation_enabled" not in json
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "prediction": 0,
                "prediction_label": "Retour rapide",
                "probabilities": {"0": 0.80, "1": 0.15, "2": 0.05},
                "model_version": "v1.2.3",
                "request_id": headers["X-Request-ID"],
            },
            text="ok",
        )

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    response = TestClient(app).post(
        "/score",
        json={**VALID_APPLICATION, "class_2_escalation_enabled": False},
    )

    assert response.status_code == 200
    assert response.json()["needs_human_review"] is False
    assert response.json()["review_reasons"] == []
    assert response.json()["class_2_escalation_enabled"] is False


def test_server_policy_can_disable_class_2_escalation(monkeypatch):
    monkeypatch.setattr(backend_main, "ENABLE_CLASS_2_ESCALATION", False)

    response = TestClient(app).get("/policy")

    assert response.status_code == 200
    assert response.json()["class_2_escalation_available"] is False
    assert response.json()["class_2_escalation_threshold"] == 0.04


def test_server_policy_overrides_client_escalation_choice(monkeypatch, tmp_path):
    monkeypatch.setenv("FEEDBACK_DB", str(tmp_path / "feedbacks.db"))
    monkeypatch.setattr(backend_main, "ENABLE_CLASS_2_ESCALATION", False)

    async def fake_post(self, url, json, headers=None):
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "prediction": 0,
                "prediction_label": "Retour rapide",
                "probabilities": {"0": 0.80, "1": 0.15, "2": 0.05},
                "model_version": "v1.2.3",
                "request_id": headers["X-Request-ID"],
            },
            text="ok",
        )

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    response = TestClient(app).post(
        "/score",
        json={**VALID_APPLICATION, "class_2_escalation_enabled": True},
    )

    assert response.status_code == 200
    assert response.json()["class_2_escalation_enabled"] is False
    assert response.json()["needs_human_review"] is False
    assert response.json()["review_reasons"] == []


def test_score_at_confidence_threshold_is_not_abstained(monkeypatch, tmp_path):
    monkeypatch.setenv("FEEDBACK_DB", str(tmp_path / "feedbacks.db"))

    async def fake_post(self, url, json, headers=None):
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "prediction": 1,
                "prediction_label": "Retour moyen",
                "probabilities": {"0": 0.20, "1": 0.55, "2": 0.25},
                "model_version": "v1.2.3",
                "request_id": headers["X-Request-ID"],
            },
            text="ok",
        )

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    response = TestClient(app).post("/score", json=VALID_APPLICATION)

    assert response.status_code == 200
    assert response.json()["needs_human_review"] is False
    assert response.json()["review_reasons"] == []


def test_train_validates_minimum_records():
    response = TestClient(app).post(
        "/train",
        json={"records": [VALID_APPLICATION]},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("limit", [0, 201])
def test_history_rejects_limit_out_of_range(limit):
    response = TestClient(app).get(f"/history?limit={limit}")

    assert response.status_code == 422
    assert any(error["loc"][-1] == "limit" for error in response.json()["detail"])


def test_score_rejects_empty_usager_id():
    response = TestClient(app).post(
        "/score",
        json={**VALID_APPLICATION, "usager_id": ""},
    )

    assert response.status_code == 422
    assert any(error["loc"][-1] == "usager_id" for error in response.json()["detail"])


def test_feedback_rejects_true_label_outside_range():
    client = TestClient(app)
    for true_label in (-1, 3):
        response = client.post(
            "/feedback",
            json={"request_id": "unused", "prediction": 1, "true_label": true_label},
        )

        assert response.status_code == 422
        assert any(error["loc"][-1] == "true_label" for error in response.json()["detail"])


@pytest.mark.parametrize("failure", ["upstream_status", "invalid_json", "schema_mismatch"])
def test_score_model_response_failures_return_502(monkeypatch, failure):
    def response_json():
        if failure == "invalid_json":
            raise ValueError("invalid JSON")
        if failure == "schema_mismatch":
            return {"prediction": 1}
        return {}

    async def fake_post(self, url, json, headers=None):
        return SimpleNamespace(
            status_code=500 if failure == "upstream_status" else 200,
            json=response_json,
            text="model upstream failure",
        )

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    response = TestClient(app).post("/score", json=VALID_APPLICATION)

    assert response.status_code == 502


@pytest.mark.parametrize("route", ["/score", "/train"])
def test_model_unavailable_returns_503(monkeypatch, route):
    async def fake_post(self, url, json, headers=None):
        raise httpx.ConnectError(
            "model unavailable",
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    if route == "/score":
        payload = VALID_APPLICATION
    else:
        record = {**VALID_APPLICATION, "classe_retour_emploi": 1}
        payload = {"records": [record] * 10}

    response = TestClient(app).post(route, json=payload)

    assert response.status_code == 503


def test_user_registry_unavailable_returns_503(monkeypatch):
    from app.registry import UserRegistryError

    monkeypatch.setenv("USER_REGISTRY_URL", "http://registry.invalid")

    async def reject_user(self, user_id):
        raise UserRegistryError("User registry unavailable")

        monkeypatch.setattr("app.main.UserRegistryClient.ensure_user_exists", reject_user)
    response = TestClient(app).post(
        "/score",
        json={**VALID_APPLICATION, "usager_id": "unknown-user"},
    )

    assert response.status_code == 503


def test_usager_check_returns_previous_inference_count(monkeypatch, tmp_path):
    import sqlite3

    db_path = tmp_path / "feedbacks.db"
    monkeypatch.setenv("FEEDBACK_DB", str(db_path))
    backend_main._init_feedback_db()
    with sqlite3.connect(db_path) as connection:
        connection.executemany(
            """INSERT INTO predictions
            (request_id, usager_id, input_json, prediction, probabilities_json,
             model_version, created_at)
            VALUES (?, ?, '{}', 0, '{}', 'test', 'now')""",
            [("previous-1", "user-123"), ("previous-2", "user-123")],
        )

    response = TestClient(app).get("/usagers/check?usager_id=%20user-123%20")

    assert response.status_code == 200
    assert response.json() == {
        "usager_id": "user-123",
        "already_scored": True,
        "previous_inferences": 2,
    }


def test_usager_check_allows_new_usager(monkeypatch, tmp_path):
    monkeypatch.setenv("FEEDBACK_DB", str(tmp_path / "feedbacks.db"))

    response = TestClient(app).get("/usagers/check?usager_id=new-user")

    assert response.status_code == 200
    assert response.json()["already_scored"] is False
    assert response.json()["previous_inferences"] == 0


@pytest.mark.parametrize("failure", ["upstream_status", "invalid_json", "schema_mismatch"])
def test_train_model_response_failures_return_502(monkeypatch, failure):
    def response_json():
        if failure == "invalid_json":
            raise ValueError("invalid JSON")
        if failure == "schema_mismatch":
            return {}
        return {}

    async def fake_post(self, url, json, headers=None):
        return SimpleNamespace(
            status_code=500 if failure == "upstream_status" else 200,
            json=response_json,
            text="model upstream failure",
        )

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    record = {**VALID_APPLICATION, "classe_retour_emploi": 1}
    response = TestClient(app).post("/train", json={"records": [record] * 10})

    assert response.status_code == 502


def _register_prediction(request_id: str = "req-feedback") -> None:
    import os
    import sqlite3
    from pathlib import Path

    db_path = Path(
        os.environ.get("FEEDBACK_DB", str(Path.cwd() / "data" / "feedbacks.db"))
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                request_id TEXT PRIMARY KEY,
                input_json TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                probabilities_json TEXT NOT NULL,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS feedbacks (
                request_id TEXT PRIMARY KEY,
                prediction INTEGER NOT NULL,
                true_label INTEGER NOT NULL,
                comments TEXT,
                created_at TEXT NOT NULL,
                used_for_training INTEGER NOT NULL DEFAULT 0
            )"""
        )
        connection.execute(
            """INSERT OR REPLACE INTO predictions
            (request_id, input_json, prediction, probabilities_json,
             model_version, created_at)
            VALUES (?, '{}', 1, '{"0": 0.2, "1": 0.8, "2": 0.0}', 'test', 'now')""",
            (request_id,),
        )


def test_pending_prediction_update_rescores_and_preserves_request_id(monkeypatch, tmp_path):
    monkeypatch.setenv("FEEDBACK_DB", str(tmp_path / "feedbacks.db"))
    _register_prediction("req-edit")
    sent_payload = {}

    async def fake_post(self, url, json, headers=None):
        sent_payload.update(json)
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "prediction": 2,
                "prediction_label": "Risque de chômage longue durée",
                "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
                "model_version": "v-edited",
                "request_id": headers["X-Request-ID"],
            },
            text="ok",
        )

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    response = TestClient(app).put(
        "/predictions/req-edit",
        json={
            **VALID_APPLICATION,
            "usager_id": "user-edited",
            "session_id": "session-edit",
            "synthese_entretien": "Informations corrigées avant annotation.",
        },
    )

    assert response.status_code == 200
    assert response.json()["request_id"] == "req-edit"
    assert response.json()["prediction"] == 2
    assert "usager_id" not in sent_payload

    history = TestClient(app).get("/history?usager_id=user-edited")
    assert history.status_code == 200
    assert history.json()[0]["request_id"] == "req-edit"
    assert history.json()[0]["prediction"] == 2
    assert history.json()[0]["inputs"]["synthese_entretien"] == (
        "Informations corrigées avant annotation."
    )


def test_pending_prediction_can_be_deleted(monkeypatch, tmp_path):
    monkeypatch.setenv("FEEDBACK_DB", str(tmp_path / "feedbacks.db"))
    _register_prediction("req-delete")

    response = TestClient(app).delete("/predictions/req-delete")

    assert response.status_code == 200
    assert response.json() == {"status": "deleted", "request_id": "req-delete"}
    assert TestClient(app).get("/history?limit=200").json() == []


@pytest.mark.parametrize("method", ["put", "delete"])
def test_prediction_mutations_reject_feedback(monkeypatch, tmp_path, method):
    import sqlite3

    db_path = tmp_path / "feedbacks.db"
    monkeypatch.setenv("FEEDBACK_DB", str(db_path))
    _register_prediction("req-locked")
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """INSERT INTO feedbacks
            (request_id, prediction, true_label, created_at)
            VALUES (?, 1, 1, 'now')""",
            ("req-locked",),
        )

    client = TestClient(app)
    if method == "put":
        response = client.put("/predictions/req-locked", json=VALID_APPLICATION)
    else:
        response = client.delete("/predictions/req-locked")

    assert response.status_code == 409


@pytest.mark.parametrize("method", ["put", "delete"])
def test_prediction_mutations_reject_unknown_request_id(method):
    client = TestClient(app)
    if method == "put":
        response = client.put("/predictions/not-found", json=VALID_APPLICATION)
    else:
        response = client.delete("/predictions/not-found")

    assert response.status_code == 404


def test_feedback_requires_known_prediction():
    response = TestClient(app).post(
        "/feedback",
        json={"request_id": "missing", "prediction": 1, "true_label": 1},
    )
    assert response.status_code == 404


def test_feedback_is_idempotent_and_rejects_conflict():
    _register_prediction()
    client = TestClient(app)
    payload = {"request_id": "req-feedback", "prediction": 1, "true_label": 2}

    first = client.post("/feedback", json=payload)
    second = client.post("/feedback", json=payload)
    conflict = client.post(
        "/feedback",
        json={**payload, "true_label": 0},
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json()["status"] == "already_registered"
    assert conflict.status_code == 409
    assert client.get("/feedback/count").json()["new"] == 1


def test_feedback_triggers_retrainer_at_threshold(monkeypatch, tmp_path):
    monkeypatch.setenv("FEEDBACK_DB", str(tmp_path / "feedbacks.db"))
    monkeypatch.setenv("RETRAIN_MIN_FEEDBACK", "1")
    monkeypatch.setenv("RETRAIN_API_TOKEN", "test-token")
    monkeypatch.setattr(backend_main, "RETRAINER_URL", "http://retrainer:8002")
    _register_prediction("req-trigger")
    called = {}

    async def fake_post(self, url, headers=None):
        called["url"] = url
        called["token"] = headers["X-Train-Token"]
        return SimpleNamespace(status_code=202, json=lambda: {"status": "started"})

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    response = TestClient(app).post(
        "/feedback",
        json={"request_id": "req-trigger", "prediction": 1, "true_label": 2},
    )

    assert response.status_code == 201
    assert response.json()["retraining_status"] == "started"
    assert called == {
        "url": "http://retrainer:8002/retrain",
        "token": "test-token",
    }


def test_feedback_is_saved_when_retrainer_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("FEEDBACK_DB", str(tmp_path / "feedbacks.db"))
    monkeypatch.setenv("RETRAIN_MIN_FEEDBACK", "1")
    monkeypatch.setenv("RETRAIN_API_TOKEN", "test-token")
    _register_prediction("req-retrainer-unavailable")

    async def fail_post(self, url, headers=None):
        raise httpx.ConnectError(
            "retrainer unavailable",
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("httpx.AsyncClient.post", fail_post)
    client = TestClient(app)
    response = client.post(
        "/feedback",
        json={
            "request_id": "req-retrainer-unavailable",
            "prediction": 1,
            "true_label": 2,
        },
    )

    assert response.status_code == 201
    assert response.json()["retraining_status"] == "unavailable"
    assert client.get("/feedback/count").json()["new"] == 1


def test_evaluation_metrics_endpoint_serves_latest_gate(monkeypatch, tmp_path):
    metrics_path = tmp_path / "evaluation.prom"
    metrics_path.write_text("cisia_evaluation_gate_status 1\n", encoding="utf-8")
    monkeypatch.setattr(backend_main, "EVALUATION_METRICS_FILE", metrics_path)

    response = TestClient(app).get("/evaluation-metrics")

    assert response.status_code == 200
    assert response.text == "cisia_evaluation_gate_status 1\n"


def test_retrain_metrics_endpoint_serves_latest_run(monkeypatch, tmp_path):
    metrics_path = tmp_path / "retrain.prom"
    metrics_path.write_text(
        'cisia_retrain_last_run_status_info{status="skipped_low_volume"} 1\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(backend_main, "RETRAIN_METRICS_FILE", metrics_path)

    response = TestClient(app).get("/retrain-metrics")

    assert response.status_code == 200
    assert 'status="skipped_low_volume"' in response.text
