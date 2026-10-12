import json
import sqlite3
import sys

import joblib
import pandas as pd
import pytest

from scripts.retrain import (
    TARGET_COLUMN,
    build_training_data,
    calibration_indices,
    next_promoted_version,
    reference_holdout_indices,
    should_retrain,
)

SCHEMA = """
CREATE TABLE predictions (
    request_id TEXT PRIMARY KEY,
    input_json TEXT NOT NULL,
    prediction INTEGER NOT NULL,
    probabilities_json TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE feedbacks (
    request_id TEXT PRIMARY KEY,
    prediction INTEGER NOT NULL,
    true_label INTEGER NOT NULL,
    comments TEXT,
    created_at TEXT NOT NULL,
    used_for_training INTEGER NOT NULL DEFAULT 0
);
"""


def _training_row(source: pd.DataFrame) -> dict:
    """First dataset row outside the fixed calibration and holdout splits."""
    protected = set(reference_holdout_indices()) | set(calibration_indices())
    return source.loc[next(i for i in source.index if i not in protected)].to_dict()


def _insert_feedback(db_path, request_id: str, row: dict, label: int) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.executescript(SCHEMA.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS"))
        connection.execute(
            "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
            (request_id, json.dumps(row, default=str), label, "{}", "test", "now"),
        )
        connection.execute(
            "INSERT INTO feedbacks VALUES (?, ?, ?, ?, ?, 0)",
            (request_id, label, label, None, "now"),
        )


def test_retrain_trigger_stays_off_below_threshold():
    assert should_retrain(199, 200) is False


def test_retrain_trigger_starts_at_threshold():
    assert should_retrain(200, 200) is True


def test_promoted_model_version_increments_semver_and_migrates_legacy_tag():
    assert next_promoted_version("v1.0.1", "v1.0.0") == "v1.0.2"
    assert next_promoted_version("promoted-20260929004805", "v1.0.0") == "v1.0.1"
    assert next_promoted_version(None, "unknown") == "v1.0.0"


def test_low_feedback_run_writes_structured_result(tmp_path, monkeypatch):
    from scripts import retrain

    result_path = tmp_path / "reports" / "retrain-result.json"
    metrics_path = tmp_path / "reports" / "retrain.prom"
    monkeypatch.setattr(retrain, "load_unconsumed_feedbacks", pd.DataFrame)
    monkeypatch.setenv("RETRAIN_RESULT_PATH", str(result_path))
    monkeypatch.setenv("RETRAIN_METRICS_PATH", str(metrics_path))
    monkeypatch.setattr(sys, "argv", ["retrain.py", "--min-feedback", "2"])

    assert retrain.main() == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result == {
        "status": "skipped_low_volume",
        "new_feedbacks": 0,
        "excluded_feedbacks": 0,
        "min_feedback": 2,
    }
    metrics = metrics_path.read_text(encoding="utf-8")
    assert 'cisia_retrain_last_run_status_info{status="skipped_low_volume"} 1' in metrics
    assert "cisia_retrain_last_run_new_feedbacks 0" in metrics
    assert "cisia_retrain_last_run_min_feedback 2" in metrics
    assert "cisia_retrain_candidate_trained 0" in metrics


def test_feedback_training_data_keeps_cisia_three_class_labels(tmp_path, monkeypatch):
    db_path = tmp_path / "feedbacks.db"
    monkeypatch.setenv("FEEDBACK_DB", str(db_path))
    source = pd.read_csv("data/dataset_trajectoire_emploi.csv")
    row = _training_row(source)
    request_id = row.pop("usager_id")
    true_label = int(row.pop(TARGET_COLUMN))

    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE predictions (
                request_id TEXT PRIMARY KEY,
                input_json TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                probabilities_json TEXT NOT NULL,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE feedbacks (
                request_id TEXT PRIMARY KEY,
                prediction INTEGER NOT NULL,
                true_label INTEGER NOT NULL,
                comments TEXT,
                created_at TEXT NOT NULL,
                used_for_training INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        connection.execute(
            "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
            (request_id, json.dumps(row), true_label, "{}", "test", "now"),
        )
        connection.execute(
            "INSERT INTO feedbacks VALUES (?, ?, ?, ?, ?, 0)",
            (request_id, true_label, true_label, None, "now"),
        )

    from scripts.retrain import load_unconsumed_feedbacks

    feedbacks = load_unconsumed_feedbacks()
    _, target = build_training_data(feedbacks)

    assert set(target.unique()) == {0, 1, 2}
    assert len(target) == (
        len(source)
        - len(reference_holdout_indices())
        - len(calibration_indices())
        + 1
    )


def test_feedback_to_promotion_end_to_end(tmp_path, monkeypatch):
    from scripts import retrain

    db_path = tmp_path / "feedbacks.db"
    monkeypatch.setenv("FEEDBACK_DB", str(db_path))
    source = pd.read_csv("data/dataset_trajectoire_emploi.csv")
    row = _training_row(source)
    request_id = row.pop("usager_id")
    true_label = int(row.pop(TARGET_COLUMN))

    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE predictions (
                request_id TEXT PRIMARY KEY,
                input_json TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                probabilities_json TEXT NOT NULL,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE feedbacks (
                request_id TEXT PRIMARY KEY,
                prediction INTEGER NOT NULL,
                true_label INTEGER NOT NULL,
                comments TEXT,
                created_at TEXT NOT NULL,
                used_for_training INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        connection.execute(
            "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
            (request_id, json.dumps(row), true_label, "{}", "test", "now"),
        )
        connection.execute(
            "INSERT INTO feedbacks VALUES (?, ?, ?, ?, ?, 0)",
            (request_id, true_label, true_label, None, "now"),
        )

    monkeypatch.setattr(retrain, "CANDIDATE_PATH", tmp_path / "candidate.joblib")
    monkeypatch.setattr(retrain, "CANDIDATE_METADATA_PATH", tmp_path / "candidate.json")
    monkeypatch.setattr(retrain, "PROMOTED_PATH", tmp_path / "promoted.joblib")
    monkeypatch.setattr(retrain, "PROMOTED_METADATA_PATH", tmp_path / "promoted.json")
    monkeypatch.setattr(retrain, "DECISION_LOG", tmp_path / "decisions.jsonl")
    monkeypatch.setenv("RETRAIN_METRICS_PATH", str(tmp_path / "retrain.prom"))
    result_path = tmp_path / "retrain-result.json"
    monkeypatch.setenv("RETRAIN_RESULT_PATH", str(result_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    monkeypatch.setattr(sys, "argv", ["retrain.py", "--min-feedback", "1"])

    assert retrain.main() == 0
    assert (tmp_path / "candidate.joblib").exists()
    from sklearn.calibration import CalibratedClassifierCV

    candidate = joblib.load(tmp_path / "candidate.joblib")
    assert isinstance(candidate, CalibratedClassifierCV)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    record = json.loads((tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert result["status"] == record["decision"]["status"]
    assert result["mlflow_run_id"] == record["candidate"]["mlflow_run_id"]
    assert record["schema_version"] == retrain.DECISION_SCHEMA_VERSION
    assert record["decision"]["status"] in {"promoted", "rejected"}
    assert record["candidate"]["mlflow_run_id"]
    assert record["candidate"]["registry"]["name"] == retrain.REGISTERED_MODEL_NAME
    assert record["production"]["model_version"]
    assert len(record["data"]["reference_set_sha256"]) == 64
    assert record["data"]["excluded_calibration_rows"] == 400
    assert len(record["data"]["calibration_indices_sha256"]) == 64
    assert set(record["metrics"]["delta_candidate_minus_production"]) == set(record["metrics"]["candidate"])
    assert set(result["metrics_reference"]) == {"candidate", "production"}
    for metrics in result["metrics_reference"].values():
        assert "abstention_rate_pmax_below_0_55" in metrics
        assert "critical_error_2_to_0_rate" in metrics
        assert "critical_error_2_to_0_count" in metrics
    assert {rule["status"] for rule in record["quality_gate"]["rules"]} <= {"PASS", "FAIL"}
    assert (record["quality_gate"]["status"] == "PASS") == record["decision"]["promote"]

    with sqlite3.connect(db_path) as connection:
        consumed = connection.execute(
            "SELECT used_for_training FROM feedbacks WHERE request_id = ?",
            (request_id,),
        ).fetchone()[0]
    assert consumed == 1


def test_reference_holdout_is_never_used_for_training():
    features, _ = build_training_data(pd.DataFrame())
    assert features.index.intersection(reference_holdout_indices()).empty


def test_historical_fit_calibration_and_holdout_splits_are_disjoint():
    from scripts.retrain import load_calibration_data

    training_features, _ = build_training_data(pd.DataFrame())
    calibration_features, _ = load_calibration_data()
    training_indices = set(map(int, training_features.index))
    calibration_index_set = set(calibration_indices())
    holdout_index_set = set(reference_holdout_indices())

    assert len(training_indices) == 1600
    assert len(calibration_index_set) == 400
    assert len(holdout_index_set) == 500
    assert len(calibration_features) == 400
    assert not training_indices & calibration_index_set
    assert not training_indices & holdout_index_set
    assert not calibration_index_set & holdout_index_set


def test_calibration_index_hash_is_verified(tmp_path, monkeypatch):
    from scripts import retrain

    manifest = json.loads(retrain.CALIBRATION_INDEX_PATH.read_text(encoding="utf-8"))
    manifest["indices"][0] += 5000
    tampered_path = tmp_path / "calibration_indices.json"
    tampered_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(retrain, "CALIBRATION_INDEX_PATH", tampered_path)

    with pytest.raises(ValueError, match="hash does not match"):
        calibration_indices()


def test_promoted_metadata_keeps_calibrated_artifact_contract(tmp_path, monkeypatch):
    from scripts import retrain

    production_metadata_path = tmp_path / "production.json"
    production_metadata_path.write_text(
        retrain.PRODUCTION_METADATA_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    candidate_calibration = {
        "method": "isotonic",
        "estimator": "sklearn.calibration.CalibratedClassifierCV",
        "cv": "prefit",
        "calibration_rows": 400,
    }
    candidate_metadata_path = tmp_path / "candidate.json"
    candidate_metadata_path.write_text(
        json.dumps({
            "configuration": {"model_parameters": retrain.XGB_PARAMS},
            "dataset": {},
            "training": {},
            "calibration": candidate_calibration,
        }),
        encoding="utf-8",
    )
    promoted_metadata_path = tmp_path / "promoted.json"
    monkeypatch.setattr(retrain, "PRODUCTION_METADATA_PATH", production_metadata_path)
    monkeypatch.setattr(retrain, "CANDIDATE_METADATA_PATH", candidate_metadata_path)
    monkeypatch.setattr(retrain, "PROMOTED_METADATA_PATH", promoted_metadata_path)
    monkeypatch.setattr(retrain, "PROMOTED_PATH", tmp_path / "promoted.joblib")
    metrics = {
        "f1_macro": 0.7,
        "abstention_rate_pmax_below_0_55": 0.2,
        "critical_error_2_to_0_rate": 0.1,
    }

    retrain.write_promoted_metadata(metrics, feedback_count=3)

    promoted_metadata = json.loads(promoted_metadata_path.read_text(encoding="utf-8"))
    assert promoted_metadata["calibration"] == candidate_calibration
    assert promoted_metadata["evaluation"]["metrics"] == metrics
    assert promoted_metadata["artifacts"]["model_path"] == "promoted.joblib"
    assert promoted_metadata["artifacts"]["metadata_path"] == "promoted.json"


def test_feedback_replaying_reference_row_is_excluded(tmp_path, monkeypatch):
    from scripts.retrain import drop_reference_overlap, load_unconsumed_feedbacks

    db_path = tmp_path / "feedbacks.db"
    monkeypatch.setenv("FEEDBACK_DB", str(db_path))
    reference = pd.read_csv("data/reference_set.csv").iloc[0].to_dict()
    reference.pop("usager_id")
    reference_label = int(reference.pop(TARGET_COLUMN))
    _insert_feedback(db_path, "REF-REPLAY", reference, reference_label)

    fresh = _training_row(pd.read_csv("data/dataset_trajectoire_emploi.csv"))
    fresh.pop("usager_id")
    fresh_label = int(fresh.pop(TARGET_COLUMN))
    _insert_feedback(db_path, "FRESH", fresh, fresh_label)

    usable, excluded = drop_reference_overlap(load_unconsumed_feedbacks())

    assert excluded == 1
    assert usable["request_id"].tolist() == ["FRESH"]


def test_feedback_replaying_calibration_row_is_excluded(tmp_path, monkeypatch):
    from scripts.retrain import (
        drop_calibration_overlap,
        load_unconsumed_feedbacks,
    )

    db_path = tmp_path / "feedbacks.db"
    monkeypatch.setenv("FEEDBACK_DB", str(db_path))
    calibration = pd.read_csv("data/dataset_trajectoire_emploi.csv").iloc[
        calibration_indices()[0]
    ].to_dict()
    calibration.pop("usager_id")
    calibration_label = int(calibration.pop(TARGET_COLUMN))
    _insert_feedback(db_path, "CALIBRATION-REPLAY", calibration, calibration_label)

    fresh = _training_row(pd.read_csv("data/dataset_trajectoire_emploi.csv"))
    fresh.pop("usager_id")
    fresh_label = int(fresh.pop(TARGET_COLUMN))
    _insert_feedback(db_path, "FRESH", fresh, fresh_label)

    usable, excluded = drop_calibration_overlap(load_unconsumed_feedbacks())

    assert excluded == 1
    assert usable["request_id"].tolist() == ["FRESH"]


def test_only_read_feedbacks_are_marked_consumed(tmp_path, monkeypatch):
    from scripts.retrain import mark_feedbacks_consumed

    db_path = tmp_path / "feedbacks.db"
    monkeypatch.setenv("FEEDBACK_DB", str(db_path))
    row = _training_row(pd.read_csv("data/dataset_trajectoire_emploi.csv"))
    row.pop("usager_id")
    label = int(row.pop(TARGET_COLUMN))
    _insert_feedback(db_path, "READ", row, label)
    _insert_feedback(db_path, "ARRIVED-DURING-FIT", row, label)

    mark_feedbacks_consumed(["READ"])

    with sqlite3.connect(db_path) as connection:
        flags = dict(connection.execute("SELECT request_id, used_for_training FROM feedbacks"))
    assert flags == {"READ": 1, "ARRIVED-DURING-FIT": 0}