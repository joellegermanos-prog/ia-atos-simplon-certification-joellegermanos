"""Train a CISIA multiclass candidate from validated production feedbacks.

The script deliberately creates a candidate artifact only. Promotion and
replacement of the serving model remain separate release-gate decisions.

Usage:
    python scripts/retrain.py --min-feedback 200
    python scripts/retrain.py --min-feedback 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = ROOT / "services" / "model"
DATA = ROOT / "data"
MODELS = SERVICE_ROOT / "models"
SOURCE_DATASET = DATA / "dataset_trajectoire_emploi.csv"
CANDIDATE_PATH = MODELS / "cisia_emploi_candidate.joblib"
CANDIDATE_METADATA_PATH = MODELS / "cisia_emploi_candidate.json"
PROMOTED_PATH = MODELS / "cisia_emploi_promoted.joblib"
PROMOTED_METADATA_PATH = MODELS / "cisia_emploi_promoted.json"
DECISION_LOG = Path(os.environ.get("DECISION_LOG", str(ROOT / "decisions_log.jsonl")))
PRODUCTION_PATH = MODELS / "cisia_emploi_xgboost_multimodal_ethique_best_class_2_ethique.joblib"
PRODUCTION_METADATA_PATH = MODELS / "cisia_emploi_xgboost_multimodal_ethique_best_class_2_ethique.json"
REFERENCE_SET = DATA / "reference_set.csv"
MLFLOW_EXPERIMENT = "cisia-emploi-retraining"
REGISTERED_MODEL_NAME = "cisia-emploi"
DECISION_SCHEMA_VERSION = "2.0"

sys.path.insert(0, str(SERVICE_ROOT))
from preprocess import (
    TARGET_COLUMN,
    build_preprocessor,
    create_features,
    load_dataset,
)

try:  # Supports both `python scripts/retrain.py` and pytest imports.
    from promotion import decide_promotion
except ModuleNotFoundError:  # pragma: no cover - import-mode compatibility
    from scripts.promotion import decide_promotion

# The previous M6 reference implementation used RandomForestClassifier here.
# It is intentionally left as a comment for traceability; M5 production uses
# the CISIA XGBoost multiclass pipeline below.
# from sklearn.ensemble import RandomForestClassifier
# candidate = Pipeline([("preprocessor", build_preprocessor()),
#                       ("classifier", RandomForestClassifier(...))])

# Hyperparamètres de la config `best_class_2_ethique` (examen/src/train.py).
XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "n_estimators": 400,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 3,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}
CLASS_WEIGHTS = {0: 1.0, 1: 1.0, 2: 2.5}


def feedback_db_path() -> Path:
    return Path(os.environ.get("FEEDBACK_DB", str(DATA / "feedbacks.db")))


def should_retrain(feedback_count: int, min_feedback: int) -> bool:
    """Return whether the number of new feedbacks reaches the trigger."""
    return feedback_count >= min_feedback


def load_unconsumed_feedbacks() -> pd.DataFrame:
    """Return raw production inputs joined with their validated labels."""
    db_path = feedback_db_path()
    if not db_path.exists():
        return pd.DataFrame()

    with sqlite3.connect(db_path) as connection:
        feedbacks = pd.read_sql_query(
            """SELECT p.request_id, p.input_json, f.true_label
               FROM predictions AS p
               JOIN feedbacks AS f ON f.request_id = p.request_id
               WHERE f.used_for_training = 0""",
            connection,
        )

    if feedbacks.empty:
        return feedbacks

    inputs = pd.json_normalize(feedbacks["input_json"].map(json.loads))
    inputs = inputs.drop(columns=["usager_id", "session_id"], errors="ignore")
    rows = pd.concat(
        [feedbacks[["request_id", "true_label"]].reset_index(drop=True), inputs],
        axis=1,
    )
    rows = rows.rename(columns={"true_label": TARGET_COLUMN})
    return rows


def reference_holdout_indices() -> list[int]:
    """Return the dataset rows frozen as reference set for the production model."""
    metadata = json.loads(PRODUCTION_METADATA_PATH.read_text(encoding="utf-8"))
    return [int(index) for index in metadata["dataset"]["test_indices"]]


def _normalize_value(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    try:
        return str(round(float(value), 3))
    except (TypeError, ValueError):
        return str(value).strip()


def _feature_keys(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Build a normalized row signature used to detect reference-set duplicates."""
    return frame[columns].apply(lambda row: "|".join(_normalize_value(v) for v in row), axis=1)


def drop_reference_overlap(feedbacks: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove feedbacks whose inputs replay a reference-set observation."""
    if feedbacks.empty or not REFERENCE_SET.exists():
        return feedbacks, 0
    reference = create_features(pd.read_csv(REFERENCE_SET))
    candidates = create_features(feedbacks)
    columns = [
        column
        for column in ("niveau_diplome", "anciennete_poste_ans", "code_rome_vise",
                       "departement", "est_allocataire", "synthese_entretien")
        if column in reference.columns and column in candidates.columns
    ]
    overlap = _feature_keys(candidates, columns).isin(set(_feature_keys(reference, columns)))
    return feedbacks.loc[~overlap.to_numpy()].reset_index(drop=True), int(overlap.sum())


def build_training_data(feedbacks: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Append labelled production rows to the historical CISIA training set.

    The reference holdout is removed so that the candidate is evaluated on
    observations it has never seen, like the production model.
    """
    historical_X, historical_y = load_dataset(SOURCE_DATASET, include_sensitive=False)
    holdout = historical_X.index.intersection(reference_holdout_indices())
    historical_X, historical_y = historical_X.drop(index=holdout), historical_y.drop(index=holdout)
    if feedbacks.empty:
        return historical_X, historical_y

    required_feedback_columns = {
        "niveau_diplome",
        "anciennete_poste_ans",
        "code_rome_vise",
        "code_insee_commune",
        "est_allocataire",
        "synthese_entretien",
        TARGET_COLUMN,
    }
    missing_columns = sorted(required_feedback_columns - set(feedbacks.columns))
    if missing_columns:
        raise ValueError(
            "Feedback registry contains incomplete prediction inputs; "
            f"missing columns: {missing_columns}"
        )

    feedback_frame = create_features(feedbacks)
    feedback_features = feedback_frame[historical_X.columns].copy()
    feedback_target = pd.to_numeric(feedback_frame[TARGET_COLUMN], errors="raise").astype(int)
    if not set(feedback_target.unique()).issubset({0, 1, 2}):
        raise ValueError("Feedback labels must be limited to CISIA classes 0, 1 and 2")

    return (
        pd.concat([historical_X, feedback_features], ignore_index=True),
        pd.concat([historical_y, feedback_target], ignore_index=True),
    )


def build_candidate() -> Pipeline:
    """Build the same ethical multimodal pipeline as production (no sensitive features)."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(include_sensitive=False, include_text=True)),
            ("classifier", XGBClassifier(**XGB_PARAMS)),
        ]
    )


def write_candidate_metadata(
    candidate: Pipeline,
    rows: int,
    feedback_count: int,
    metrics: dict[str, float],
    feedback_request_ids: list[str] | None = None,
    excluded_feedbacks: int = 0,
) -> None:
    holdout = reference_holdout_indices()
    payload = {
        "model_name": "cisia_emploi",
        "model_type": "xgboost",
        "model_version": f"candidate-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(SOURCE_DATASET.relative_to(ROOT)).replace("\\", "/"),
            "sha256": hashlib.sha256(SOURCE_DATASET.read_bytes()).hexdigest(),
            "excluded_reference_indices": holdout,
            "excluded_reference_rows": len(holdout),
            "reference_set": str(REFERENCE_SET.relative_to(ROOT)).replace("\\", "/"),
            "reference_set_sha256": (
                hashlib.sha256(REFERENCE_SET.read_bytes()).hexdigest()
                if REFERENCE_SET.exists() else None
            ),
            "baseline_model": PRODUCTION_PATH.name,
        },
        "training": {
            "rows": rows,
            "new_feedbacks": feedback_count,
            "feedback_request_ids": feedback_request_ids or [],
            "excluded_feedbacks_replaying_reference": excluded_feedbacks,
            "metrics_train": metrics,
        },
        "target": {
            "column": TARGET_COLUMN,
            "expected_values": [0, 1, 2],
        },
        "configuration": {"model_parameters": XGB_PARAMS},
    }
    CANDIDATE_METADATA_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def mark_feedbacks_consumed(request_ids: list[str]) -> None:
    """Mark only the feedbacks read by this retrain run."""
    if not request_ids:
        return
    db_path = feedback_db_path()
    with sqlite3.connect(db_path) as connection:
        connection.executemany(
            "UPDATE feedbacks SET used_for_training = 1 WHERE request_id = ?",
            [(request_id,) for request_id in request_ids],
        )


def evaluate_on_reference(model: Pipeline) -> dict[str, float]:
    """Evaluate a model on the frozen CISIA reference set."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if not REFERENCE_SET.exists():
        raise FileNotFoundError(f"Reference set not found: {REFERENCE_SET}")
    reference = pd.read_csv(REFERENCE_SET)
    target = reference[TARGET_COLUMN].astype(int)
    features = create_features(reference.drop(columns=[TARGET_COLUMN]))
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "f1_macro": float(f1_score(target, predictions, average="macro", zero_division=0)),
        "f1_classe_2": float(
            f1_score(target, predictions, labels=[2], average=None, zero_division=0)[0]
        ),
        "recall_classe_2": float(
            recall_score(target, predictions, labels=[2], average=None, zero_division=0)[0]
        ),
        "precision_classe_2": float(
            precision_score(target, predictions, labels=[2], average=None, zero_division=0)[0]
        ),
        "roc_auc_ovr_macro": float(
            roc_auc_score(target, probabilities, multi_class="ovr", average="macro")
        ),
    }


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def production_identity() -> dict[str, object]:
    """Describe the serving model and, when available, its MLflow registry entry."""
    metadata = json.loads(PRODUCTION_METADATA_PATH.read_text(encoding="utf-8"))
    registry: dict[str, object] = {"name": REGISTERED_MODEL_NAME, "alias": "champion", "version": None}
    run_id = metadata.get("mlflow_run_id")
    try:
        champion = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
        registry["version"] = champion.version
        run_id = run_id or champion.run_id
    except MlflowException as exc:  # registry absent or alias not set yet
        registry["error"] = type(exc).__name__
    return {
        "model_name": metadata.get("model_name"),
        "model_version": metadata.get("model_version"),
        "model_type": metadata.get("model_type"),
        "scenario": metadata.get("scenario_name"),
        "config": metadata.get("config_name"),
        "artifact": _relative(PRODUCTION_PATH),
        "artifact_sha256": _sha256(PRODUCTION_PATH),
        "trained_on_dataset_sha256": metadata.get("dataset", {}).get("sha256"),
        "mlflow_run_id": run_id,
        "registry": registry,
    }


def register_candidate(candidate: Pipeline, decision_status: str) -> dict[str, object]:
    """Register the candidate in the MLflow registry under the challenger alias."""
    registry: dict[str, object] = {"name": REGISTERED_MODEL_NAME, "alias": "challenger", "version": None}
    try:
        info = mlflow.sklearn.log_model(
            candidate, artifact_path="model", registered_model_name=REGISTERED_MODEL_NAME
        )
        version = str(info.registered_model_version)
        client = MlflowClient()
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "challenger", version)
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "decision", decision_status)
        registry["version"] = version
        registry["model_uri"] = f"models:/{REGISTERED_MODEL_NAME}/{version}"
    except MlflowException as exc:  # registry must not block the decision journal
        registry["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return registry


def build_decision_record(
    *,
    candidate_metadata: dict,
    candidate_metrics: dict[str, float],
    production: dict[str, object],
    production_metrics: dict[str, float],
    decision,
    candidate_run_id: str | None,
    candidate_registry: dict[str, object],
) -> dict[str, object]:
    """Assemble one self-contained, audit-ready promotion decision."""
    status = "promoted" if decision.promote else "rejected"
    deltas = {
        metric: float(candidate_metrics[metric]) - float(production_metrics.get(metric, 0.0))
        for metric in candidate_metrics
    }
    rules = list(decision.rules)
    dataset = candidate_metadata.get("dataset", {})
    training = candidate_metadata.get("training", {})
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "decision_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "decision": {"status": status, "promote": decision.promote, "reason": decision.reason},
        "candidate": {
            "model_name": candidate_metadata.get("model_name"),
            "model_version": candidate_metadata.get("model_version"),
            "model_type": candidate_metadata.get("model_type"),
            "artifact": _relative(CANDIDATE_PATH),
            "artifact_sha256": _sha256(CANDIDATE_PATH),
            "promoted_artifact": _relative(PROMOTED_PATH) if decision.promote else None,
            "mlflow_experiment": MLFLOW_EXPERIMENT,
            "mlflow_run_id": candidate_run_id,
            "registry": candidate_registry,
            "hyperparameters": candidate_metadata.get("configuration", {}).get("model_parameters"),
            "training_rows": training.get("rows"),
            "new_feedbacks": training.get("new_feedbacks"),
            "feedback_request_ids": training.get("feedback_request_ids", []),
            "excluded_feedbacks_replaying_reference": training.get("excluded_feedbacks_replaying_reference", 0),
        },
        "production": production,
        "data": {
            "reference_set": _relative(REFERENCE_SET),
            "reference_set_sha256": _sha256(REFERENCE_SET),
            "training_dataset": dataset.get("path"),
            "training_dataset_sha256": dataset.get("sha256"),
            "excluded_reference_rows": dataset.get("excluded_reference_rows"),
        },
        "metrics": {
            "evaluated_on": "reference_set",
            "candidate": candidate_metrics,
            "production": production_metrics,
            "delta_candidate_minus_production": deltas,
        },
        "quality_gate": {
            "status": "PASS" if all(rule["status"] == "PASS" for rule in rules) else "FAIL",
            "passed": sum(rule["status"] == "PASS" for rule in rules),
            "failed": sum(rule["status"] == "FAIL" for rule in rules),
            "rules": rules,
        },
    }


def log_decision(record: dict[str, object]) -> None:
    """Append every promotion decision, including rejected candidates."""
    DECISION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DECISION_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _semantic_version_parts(version: object) -> tuple[int, int, int] | None:
    if not isinstance(version, str):
        return None
    match = re.fullmatch(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", version)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def next_promoted_version(previous_version: object, base_version: object) -> str:
    """Return the next patch version, migrating legacy timestamp tags from the base version."""
    parts = _semantic_version_parts(previous_version) or _semantic_version_parts(base_version)
    if parts is None:
        return "v1.0.0"
    major, minor, patch = parts
    return f"v{major}.{minor}.{patch + 1}"


def write_promoted_metadata(
    candidate_metrics: dict[str, float],
    feedback_count: int,
) -> str:
    """Persist promotion metadata without changing the production artifact."""
    metadata = json.loads(PRODUCTION_METADATA_PATH.read_text(encoding="utf-8"))
    candidate_metadata = json.loads(CANDIDATE_METADATA_PATH.read_text(encoding="utf-8"))
    previous_promoted_metadata = (
        json.loads(PROMOTED_METADATA_PATH.read_text(encoding="utf-8"))
        if PROMOTED_METADATA_PATH.exists()
        else {}
    )
    previous_version = previous_promoted_metadata.get("model_version")
    base_version = metadata.get("model_version")
    promoted_version = next_promoted_version(previous_version, base_version)
    previous_semantic_version = (
        previous_version
        if _semantic_version_parts(previous_version) is not None
        else base_version
    )
    metadata["configuration"] = candidate_metadata["configuration"]
    metadata["candidate_dataset"] = candidate_metadata.get("dataset")
    metadata["candidate_training"] = candidate_metadata.get("training")
    metadata["model_version"] = promoted_version
    metadata["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["promotion"] = {
        "status": "promoted",
        "version": promoted_version,
        "previous_version": previous_semantic_version,
        "new_feedbacks": feedback_count,
        "reference_set": str(REFERENCE_SET.relative_to(ROOT)),
        "metrics": candidate_metrics,
    }
    PROMOTED_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return promoted_version


def write_retrain_result(result: dict[str, object]) -> None:
    """Persist and print the structured outcome for CI and local runs."""
    result_path = Path(
        os.environ.get("RETRAIN_RESULT_PATH", str(ROOT / "retrain-result.json"))
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, indent=2, ensure_ascii=False)
    result_path.write_text(serialized, encoding="utf-8")

    status = str(result.get("status", "unknown"))
    safe_status = status if status in {"skipped_low_volume", "promoted", "rejected"} else "unknown"
    candidate_trained = int(status in {"promoted", "rejected"})
    metrics_path = Path(
        os.environ.get("RETRAIN_METRICS_PATH", str(ROOT / "reports" / "retrain.prom"))
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = "\n".join([
        "# HELP cisia_retrain_last_run_status_info Status label for the latest retrainer run",
        "# TYPE cisia_retrain_last_run_status_info gauge",
        f'cisia_retrain_last_run_status_info{{status="{safe_status}"}} 1',
        "# HELP cisia_retrain_last_run_new_feedbacks New validated feedbacks available to the run",
        "# TYPE cisia_retrain_last_run_new_feedbacks gauge",
        f"cisia_retrain_last_run_new_feedbacks {int(result.get('new_feedbacks', 0))}",
        "# HELP cisia_retrain_last_run_min_feedback Feedback threshold configured for the run",
        "# TYPE cisia_retrain_last_run_min_feedback gauge",
        f"cisia_retrain_last_run_min_feedback {int(result.get('min_feedback', 0))}",
        "# HELP cisia_retrain_candidate_trained 1 when a candidate model was actually trained",
        "# TYPE cisia_retrain_candidate_trained gauge",
        f"cisia_retrain_candidate_trained {candidate_trained}",
        "# HELP cisia_retrain_last_run_timestamp_seconds Unix timestamp of the latest retrainer run",
        "# TYPE cisia_retrain_last_run_timestamp_seconds gauge",
        f"cisia_retrain_last_run_timestamp_seconds {datetime.now(timezone.utc).timestamp()}",
        "",
    ])
    temporary_metrics_path = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
    temporary_metrics_path.write_text(metrics, encoding="utf-8")
    temporary_metrics_path.replace(metrics_path)
    print(serialized)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-feedback", type=int, default=200)
    args = parser.parse_args()
    if args.min_feedback < 1:
        parser.error("--min-feedback must be positive")

    feedbacks = load_unconsumed_feedbacks()
    read_request_ids = feedbacks["request_id"].tolist() if not feedbacks.empty else []
    feedbacks, excluded_count = drop_reference_overlap(feedbacks)
    if excluded_count:
        print(f"Excluded {excluded_count} feedbacks replaying reference-set observations")
    feedback_count = len(feedbacks)
    if not should_retrain(feedback_count, args.min_feedback):
        write_retrain_result(
            {
                "status": "skipped_low_volume",
                "new_feedbacks": feedback_count,
                "excluded_feedbacks": excluded_count,
                "min_feedback": args.min_feedback,
            }
        )
        return 0

    X_train, y_train = build_training_data(feedbacks)
    candidate = build_candidate()
    candidate.fit(
        X_train,
        y_train,
        classifier__sample_weight=y_train.map(CLASS_WEIGHTS).to_numpy(dtype=float),
    )

    probabilities = candidate.predict_proba(X_train)
    predictions = candidate.predict(X_train)
    if probabilities.shape[1] != 3 or not np.isfinite(probabilities).all():
        raise ValueError("Candidate must expose finite probabilities for 3 classes")

    metrics = {
        "accuracy": float(accuracy_score(y_train, predictions)),
        "f1_macro": float(f1_score(y_train, predictions, average="macro", zero_division=0)),
        "recall_classe_2": float(
            recall_score(y_train, predictions, labels=[2], average=None, zero_division=0)[0]
        ),
    }
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate, CANDIDATE_PATH, compress=3)
    write_candidate_metadata(
        candidate,
        len(X_train),
        feedback_count,
        metrics,
        feedback_request_ids=feedbacks["request_id"].tolist(),
        excluded_feedbacks=excluded_count,
    )

    production = joblib.load(PRODUCTION_PATH)
    candidate_reference_metrics = evaluate_on_reference(candidate)
    production_reference_metrics = evaluate_on_reference(production)
    decision = decide_promotion(candidate_reference_metrics, production_reference_metrics)
    status = "promoted" if decision.promote else "rejected"
    candidate_metadata = json.loads(CANDIDATE_METADATA_PATH.read_text(encoding="utf-8"))
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    production_info = production_identity()
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=candidate_metadata["model_version"]) as run:
        mlflow.log_params({f"model.{key}": value for key, value in XGB_PARAMS.items()})
        mlflow.log_params({"training_rows": len(X_train), "new_feedbacks": feedback_count})
        mlflow.log_metrics({f"train.{key}": value for key, value in metrics.items()})
        mlflow.log_metrics(
            {f"reference.candidate.{key}": value for key, value in candidate_reference_metrics.items()}
        )
        mlflow.log_metrics(
            {f"reference.production.{key}": value for key, value in production_reference_metrics.items()}
        )
        mlflow.set_tags(
            {
                "decision": status,
                "decision_reason": decision.reason,
                "model_version": candidate_metadata["model_version"],
                "production_model_version": str(production_info["model_version"]),
                "production_mlflow_run_id": str(production_info["mlflow_run_id"]),
                "reference_set_sha256": str(_sha256(REFERENCE_SET)),
            }
        )
        mlflow.log_artifact(str(CANDIDATE_METADATA_PATH), artifact_path="model")
        candidate_registry = register_candidate(candidate, status)
        record = build_decision_record(
            candidate_metadata=candidate_metadata,
            candidate_metrics=candidate_reference_metrics,
            production=production_info,
            production_metrics=production_reference_metrics,
            decision=decision,
            candidate_run_id=run.info.run_id,
            candidate_registry=candidate_registry,
        )
        mlflow.set_tag("decision_id", record["decision_id"])
        mlflow.log_dict(record, "decision/decision_record.json")
    log_decision(record)

    promoted_model_version = None
    if decision.promote:
        shutil.copy2(CANDIDATE_PATH, PROMOTED_PATH)
        promoted_model_version = write_promoted_metadata(
            candidate_reference_metrics,
            feedback_count,
        )

    mark_feedbacks_consumed(read_request_ids)
    write_retrain_result(
        {
            "status": status,
            "promoted_model_version": promoted_model_version,
            "decision_id": record["decision_id"],
            "mlflow_run_id": record["candidate"]["mlflow_run_id"],
            "registry": candidate_registry,
            "candidate_path": str(CANDIDATE_PATH),
            "promoted_path": str(PROMOTED_PATH) if decision.promote else None,
            "new_feedbacks": feedback_count,
            "excluded_feedbacks": excluded_count,
            "min_feedback": args.min_feedback,
            "decision": decision.reason,
            "metrics_train": metrics,
            "metrics_reference": candidate_reference_metrics,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
