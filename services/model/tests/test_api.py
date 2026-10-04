"""Tests API + contract test du modèle — service model (fourni)."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

MODELS_DIR = Path(__file__).parent.parent / "models"
sys.path.insert(0, str(MODELS_DIR.parent))

from preprocess import create_features

# --- Tests API --------------------------------------------------------------

def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_returns_503_when_model_not_loaded(client):
    loaded_model = client.app.state.model
    client.app.state.model = None
    try:
        resp = client.get("/health")
    finally:
        client.app.state.model = loaded_model

    assert resp.status_code == 503


def test_info_reports_calibrated_artifact_and_decision_policy(client):
    response = client.get("/info")

    assert response.status_code == 200
    info = response.json()
    assert info["model_version"] == "v1.1.0-calibrated"
    assert info["model_artifact"].endswith("_calibrated.joblib")
    assert info["calibration_method"] == "isotonic"
    assert info["metrics_holdout"]["confidence_abstention_rate"] == 0.304
    assert info["metrics_holdout"]["human_review_rate"] == 0.416
    assert info["decision_policy"] == {
        "abstention_threshold": 0.55,
        "class_2_escalation_threshold": 0.04,
    }


def test_predict_valid_returns_class_and_proba(client, valid_payload):
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1, 2)
    assert set(body["probabilities"]) == {"0", "1", "2"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.001
    assert body["model_version"] == client.app.state.metadata["model_version"]


def test_predict_negative_seniority_returns_422(client, valid_payload):
    bad = dict(valid_payload)
    bad["anciennete_poste_ans"] = -1  # ancienneté négative interdite
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422
    assert any(error["loc"][-1] == "anciennete_poste_ans" for error in resp.json()["detail"])


def test_predict_internal_error_returns_500(client, valid_payload, monkeypatch):
    def fail_prediction(_features):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(client.app.state.model, "predict_proba", fail_prediction)

    resp = client.post("/predict", json=valid_payload)

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Prediction failed: RuntimeError"


def test_predict_does_not_require_age(client, valid_payload):
    assert "age" not in valid_payload
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 200


def test_train_requires_at_least_ten_records(client, valid_payload):
    record = dict(valid_payload)
    record["classe_retour_emploi"] = 0
    resp = client.post("/train", json={"records": [record] * 9})
    assert resp.status_code == 422


def test_train_rejects_invalid_token(client, valid_payload, monkeypatch):
    monkeypatch.setenv("TRAIN_API_TOKEN", "secret")
    record = dict(valid_payload)
    record["classe_retour_emploi"] = 0
    resp = client.post("/train", json={"records": [record] * 10})
    assert resp.status_code == 403


def test_train_is_disabled_by_default(client, valid_payload, monkeypatch):
    monkeypatch.delenv("TRAIN_API_TOKEN", raising=False)
    monkeypatch.delenv("ALLOW_DIRECT_TRAIN", raising=False)
    record = dict(valid_payload)
    record["classe_retour_emploi"] = 0
    resp = client.post("/train", json={"records": [record] * 10})
    assert resp.status_code == 409


def test_train_returns_mlflow_run(client, valid_payload, monkeypatch, tmp_path):
    monkeypatch.delenv("TRAIN_API_TOKEN", raising=False)
    monkeypatch.setenv("ALLOW_DIRECT_TRAIN", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    records = []
    for index in range(12):
        record = dict(valid_payload)
        record["anciennete_poste_ans"] = float(index)
        record["classe_retour_emploi"] = index % 3
        records.append(record)

    resp = client.post("/train", json={"records": records})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "trained"
    assert body["rows"] == 12
    assert body["mlflow_run_id"]


def test_train_internal_error_returns_500(client, valid_payload, monkeypatch, tmp_path):
    monkeypatch.delenv("TRAIN_API_TOKEN", raising=False)
    monkeypatch.setenv("ALLOW_DIRECT_TRAIN", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    monkeypatch.setattr(client.app.state, "model", object())

    class FailingCandidate(ClassifierMixin, BaseEstimator):
        def fit(self, features, target):
            raise RuntimeError("training failed")

    monkeypatch.setattr("app.main.clone", lambda _model: FailingCandidate())
    records = [
        {**valid_payload, "classe_retour_emploi": index % 3}
        for index in range(10)
    ]

    response = client.post("/train", json={"records": records})

    assert response.status_code == 500
    assert response.json()["detail"] == "Training failed: RuntimeError"


def test_metrics_endpoint_exposes_prometheus(client, valid_payload):
    client.post("/predict", json=valid_payload)  # génère au moins 1 observation
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "cisia_emploi_predictions_total" in resp.text
    assert (
        'cisia_model_info{calibration_method="isotonic",model_name="cisia_emploi"'
        in resp.text
    )


# --- Contract test du modèle (bloque la release en CI) ----------------------

def test_model_contract_features_and_output():
    """Le pipeline calibré CISIA accepte le schéma emploi et sort trois probabilités."""
    model = joblib.load(
        MODELS_DIR
        / "cisia_emploi_xgboost_multimodal_ethique_best_class_2_ethique_calibrated.joblib"
    )
    row = {
        "niveau_diplome": "Bac+2",
        "anciennete_poste_ans": 3.0,
        "code_rome_vise": "M1805",
        "code_insee_commune": "75056",
        "est_allocataire": 0,
        "synthese_entretien": (
            "Recherche un emploi stable dans le domaine informatique."
        ),
    }
    X = create_features(pd.DataFrame([row]))
    probabilities = model.predict_proba(X)[0]
    prediction = int(model.classes_[int(probabilities.argmax())])
    assert prediction in (0, 1, 2)
    assert len(probabilities) == 3
    assert abs(float(probabilities.sum()) - 1.0) < 0.001
    assert hasattr(model, "calibrated_classifiers_")
