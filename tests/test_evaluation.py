import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_model import (
    check_thresholds,
    compute_metrics,
    extract_model_hyperparameters,
    load_baseline,
    load_reference_set,
    render_prometheus_metrics,
)

MODEL_PATH = ROOT / "services" / "model" / "models" / "cisia_emploi_xgboost_multimodal_ethique_best_class_2_ethique.joblib"
META_PATH = ROOT / "services" / "model" / "models" / "cisia_emploi_xgboost_multimodal_ethique_best_class_2_ethique.json"

def test_reference_baseline_matches_current_model():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    df = load_reference_set()

    metrics = compute_metrics(model, df, meta)
    baseline = load_baseline()

    for name in ["f1_macro", "f1_classe_2", "roc_auc_ovr_macro", "recall_classe_2"]:
        assert abs(metrics[name] - baseline["metrics"][name]) < 1e-12

def test_thresholds_detect_degradation():
    baseline = {
        "metrics": {
            "f1_macro": 0.6996,
            "f1_classe_2": 0.5957,
            "roc_auc_ovr_macro": 0.8634,
            "recall_classe_2": 0.6222,
        }
    }
    metrics = {
        "f1_macro": 0.45,
        "f1_classe_2": 0.21,
        "roc_auc_ovr_macro": 0.47,
        "recall_classe_2": 0.33,
    }

    violations = check_thresholds(metrics, baseline)
    assert len(violations) > 0


def test_prometheus_metrics_export_gate_status_and_metric_limits():
    baseline = {
        "metrics": {
            "f1_macro": 0.70,
            "f1_classe_2": 0.60,
            "recall_classe_2": 0.60,
            "roc_auc_ovr_macro": 0.80,
        }
    }
    metrics = {
        "f1_macro": 0.68,
        "f1_classe_2": 0.58,
        "recall_classe_2": 0.58,
        "roc_auc_ovr_macro": 0.78,
    }

    passed = render_prometheus_metrics(metrics, baseline, [], generated_at=123.0)
    failed = render_prometheus_metrics(
        metrics, baseline, ["f1_macro below floor"], generated_at=124.0
    )

    assert "cisia_evaluation_gate_status 1" in passed
    assert 'cisia_evaluation_metric_value{metric="f1_macro"} 0.68' in passed
    assert 'cisia_evaluation_metric_absolute_min{metric="f1_macro"} 0.6' in passed
    assert 'cisia_evaluation_metric_max_drop_vs_baseline{metric="f1_macro"} 0.04' in passed
    assert "cisia_evaluation_timestamp_seconds 123.0" in passed
    assert "cisia_evaluation_gate_status 0" in failed
    assert "cisia_evaluation_violations_total 1" in failed

def test_mlflow_params_are_loaded_from_model_metadata():
    from scripts.evaluate_model import build_mlflow_params

    meta = {
        "model_version": "v2.0.0",
        "dataset_sha256": "abc123",
        "hyperparameters": {"n_estimators": 200, "max_depth": 10},
        "target": {"column": "classe_retour_emploi"},
    }

    params = build_mlflow_params(meta, "v2.0.0", 512)

    assert params["model_version"] == "v2.0.0"
    assert params["dataset_sha256"] == "abc123"
    assert params["hyperparameters.n_estimators"] == 200
    assert params["hyperparameters.max_depth"] == 10
    assert params["n_reference"] == 512


def test_cisia_hyperparameters_are_loaded_from_configuration():
    meta = {
        "configuration": {
            "model_parameters": {"n_estimators": 400, "max_depth": 6}
        }
    }

    assert extract_model_hyperparameters(meta) == {
        "n_estimators": 400,
        "max_depth": 6,
    }

def test_release_gate_blocks_on_violation():
    baseline = {
        "metrics": {
            "f1_macro": 0.6996,
            "f1_classe_2": 0.5957,
            "roc_auc_ovr_macro": 0.8634,
            "recall_classe_2": 0.6222,
        }
    }
    metrics = {
        "f1_macro": 0.45,
        "f1_classe_2": 0.21,
        "roc_auc_ovr_macro": 0.47,
        "recall_classe_2": 0.33,
    }

    violations = check_thresholds(metrics, baseline)
    assert violations