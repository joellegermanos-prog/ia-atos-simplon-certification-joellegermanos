"""Validate a model artifact through the production FastAPI service."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = ROOT / "services" / "model"


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: validate_model_service.py MODEL_ARTIFACT")

    artifact = Path(sys.argv[1]).resolve()
    os.environ["MODEL_ARTIFACT"] = str(artifact)
    sys.path.insert(0, str(SERVICE_ROOT))

    from app.main import app

    logger.remove()
    with TestClient(app) as client:
        info_response = client.get("/info")
        if info_response.status_code != 200:
            raise RuntimeError(f"/info returned HTTP {info_response.status_code}")
        info = info_response.json()
        expected_info_fields = {
            "api_version",
            "model_name",
            "model_version",
            "model_created_at",
            "metrics_holdout",
            "sklearn_version",
            "dataset_sha256",
            "model_artifact",
            "calibration_method",
            "decision_policy",
        }
        if set(info) != expected_info_fields:
            raise RuntimeError("/info response does not match the production schema")
        if info["calibration_method"] != "isotonic":
            raise RuntimeError("The model service did not load an isotonic calibrator")

        predict_response = client.post(
            "/predict",
            json={
                "niveau_diplome": "Bac+2",
                "anciennete_poste_ans": 3.0,
                "code_rome_vise": "M1805",
                "est_allocataire": 1,
                "code_insee_commune": "75056",
                "synthese_entretien": "Recherche un emploi stable.",
            },
        )
        if predict_response.status_code != 200:
            raise RuntimeError(
                f"/predict returned HTTP {predict_response.status_code}: "
                f"{predict_response.text[:500]}"
            )
        prediction = predict_response.json()
        expected_prediction_fields = {
            "prediction",
            "prediction_label",
            "probabilities",
            "model_version",
            "request_id",
        }
        if set(prediction) != expected_prediction_fields:
            raise RuntimeError("/predict response does not match the production schema")
        if set(prediction["probabilities"]) != {"0", "1", "2"}:
            raise RuntimeError("/predict did not return probabilities for all classes")

        model = app.state.model
        if not isinstance(model, CalibratedClassifierCV):
            raise RuntimeError("Loaded candidate is not a CalibratedClassifierCV")
        base_pipeline = model.calibrated_classifiers_[0].estimator
        if not isinstance(base_pipeline, Pipeline) or not isinstance(
            base_pipeline.named_steps.get("classifier"), XGBClassifier
        ):
            raise RuntimeError("Calibrated candidate lost its SHAP-compatible XGBoost pipeline")

    print(json.dumps({
        "artifact": str(artifact),
        "info_status": info_response.status_code,
        "predict_status": predict_response.status_code,
        "calibration_method": info["calibration_method"],
        "shap_compatible_xgboost_pipeline": True,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
