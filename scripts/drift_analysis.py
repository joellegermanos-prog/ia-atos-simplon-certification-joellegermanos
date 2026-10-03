"""Statistical drift analysis for CISIA reference and production samples.

The report separates:
- data drift: input distributions changed;
- concept drift: the feature/target relationship changed when labels exist;
- confidence drift: predicted probability distributions changed.

Usage:
    python scripts/drift_analysis.py --reference data/reference_set.csv \
        --current data/current_scored.csv --output reports/drift.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

TARGET = "classe_retour_emploi"
PROBABILITY_COLUMNS = ["proba_0", "proba_1", "proba_2"]
NUMERIC_FEATURES = ["anciennete_poste_ans"]
CATEGORICAL_FEATURES = ["niveau_diplome", "code_rome_vise", "code_insee_commune"]


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index using reference quantile bins."""
    reference = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    current = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    if np.ptp(reference) == 0:
        return 0.0 if np.ptp(current) == 0 and reference[0] == current[0] else float("inf")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)
    ref_rate = np.clip(ref_hist / len(reference), 1e-6, None)
    cur_rate = np.clip(cur_hist / len(current), 1e-6, None)
    return float(np.sum((cur_rate - ref_rate) * np.log(cur_rate / ref_rate)))


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """Population Stability Index from categorical value frequencies."""
    reference = reference.astype(object).where(reference.notna(), "<missing>").astype(str)
    current = current.astype(object).where(current.notna(), "<missing>").astype(str)
    if reference.empty or current.empty:
        return float("nan")
    categories = sorted(set(reference) | set(current))
    ref_rate = np.clip(
        reference.value_counts().reindex(categories, fill_value=0).to_numpy()
        / len(reference),
        1e-6,
        None,
    )
    cur_rate = np.clip(
        current.value_counts().reindex(categories, fill_value=0).to_numpy()
        / len(current),
        1e-6,
        None,
    )
    return float(np.sum((cur_rate - ref_rate) * np.log(cur_rate / ref_rate)))


def ks_result(reference: pd.Series, current: pd.Series) -> dict[str, float]:
    """Two-sample KS test for a numeric feature."""
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return {"statistic": float("nan"), "p_value": float("nan")}
    result = ks_2samp(ref, cur)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def chi2_result(reference: pd.Series, current: pd.Series) -> dict[str, float]:
    """Chi-squared independence test between source and current categories."""
    categories = sorted(set(reference.dropna().astype(str)) | set(current.dropna().astype(str)))
    if not categories:
        return {"statistic": float("nan"), "p_value": float("nan")}
    table = np.array([
        reference.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy(),
        current.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy(),
    ])
    statistic, p_value, _, _ = chi2_contingency(table)
    return {"statistic": float(statistic), "p_value": float(p_value)}


def _class_relationship(frame: pd.DataFrame, feature: str) -> dict[str, float]:
    if TARGET not in frame or feature not in frame:
        return {"statistic": float("nan"), "p_value": float("nan")}
    feature_values = frame[feature].astype(object).where(frame[feature].notna(), "<missing>")
    table = pd.crosstab(feature_values, frame[TARGET])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"statistic": float("nan"), "p_value": float("nan")}
    statistic, p_value, _, _ = chi2_contingency(table)
    return {"statistic": float(statistic), "p_value": float(p_value)}


def analyse(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Return data drift, concept drift and confidence drift diagnostics."""
    data_drift: dict[str, dict] = {"numeric": {}, "categorical": {}}
    for feature in NUMERIC_FEATURES:
        if feature in reference and feature in current:
            data_drift["numeric"][feature] = {
                "psi": psi(reference[feature], current[feature]),
                "ks": ks_result(reference[feature], current[feature]),
            }
    for feature in CATEGORICAL_FEATURES:
        if feature in reference and feature in current:
            data_drift["categorical"][feature] = {
                "psi": categorical_psi(reference[feature], current[feature]),
                "chi2": chi2_result(reference[feature], current[feature])
            }

    concept_drift = {"available": TARGET in reference and TARGET in current, "numeric": {}, "categorical": {}}
    if concept_drift["available"]:
        for feature in NUMERIC_FEATURES:
            if feature in current:
                ref_binned = pd.qcut(reference[feature], q=5, duplicates="drop")
                cur_binned = pd.qcut(current[feature], q=5, duplicates="drop")
                concept_drift["numeric"][feature] = {
                    "reference_relationship": _class_relationship(pd.DataFrame({feature: ref_binned, TARGET: reference[TARGET]}), feature),
                    "current_relationship": _class_relationship(pd.DataFrame({feature: cur_binned, TARGET: current[TARGET]}), feature),
                }
        for feature in CATEGORICAL_FEATURES:
            if feature in current:
                concept_drift["categorical"][feature] = {
                    "reference_relationship": _class_relationship(reference, feature),
                    "current_relationship": _class_relationship(current, feature),
                }

    confidence = {"available": False, "numeric": {}}
    for feature in PROBABILITY_COLUMNS:
        if feature in reference and feature in current:
            confidence["available"] = True
            confidence["numeric"][feature] = {
                "psi": psi(reference[feature], current[feature]),
                "ks": ks_result(reference[feature], current[feature]),
            }

    return {
        "thresholds": {"psi_warning": 0.25, "ks_p_value": 0.05, "chi2_p_value": 0.05},
        "data_drift": data_drift,
        "concept_drift": concept_drift,
        "confidence_drift": confidence,
        "triangulation": {
            "data_drift_requires": "PSI/KS/Chi2 significant on inputs",
            "concept_drift_requires": "label relationship changes with current labels",
            "diagnostic_note": "Statistical signals must be crossed with business and temporal context before retraining.",
        },
    }


def prometheus_metrics(report: dict) -> str:
    """Render the report as Prometheus gauges for the Grafana extension."""
    def finite(value) -> bool:
        return value is not None and bool(np.isfinite(value))

    lines = [
        "# HELP cisia_drift_psi Population Stability Index by feature",
        "# TYPE cisia_drift_psi gauge",
        "# HELP cisia_drift_ks_pvalue KS test p-value by feature",
        "# TYPE cisia_drift_ks_pvalue gauge",
        "# HELP cisia_drift_chi2_pvalue Chi-squared test p-value by feature",
        "# TYPE cisia_drift_chi2_pvalue gauge",
        "# HELP cisia_confidence_drift_psi PSI of predicted probabilities",
        "# TYPE cisia_confidence_drift_psi gauge",
    ]
    psi_watch = 0.10
    psi_warning = report["thresholds"]["psi_warning"]
    alert = 0
    severity = 0
    for feature, values in report["data_drift"]["numeric"].items():
        psi_value = values["psi"]
        ks_value = values["ks"]["p_value"]
        if finite(psi_value):
            lines.append(f'cisia_drift_psi{{feature="{feature}"}} {psi_value}')
            if psi_value >= psi_warning:
                alert = 1
                severity = max(severity, 2)
            elif psi_value >= psi_watch:
                severity = max(severity, 1)
        if finite(ks_value):
            lines.append(f'cisia_drift_ks_pvalue{{feature="{feature}"}} {ks_value}')
            if ks_value < 0.05:
                alert = 1
                severity = max(severity, 1)
    for feature, values in report["data_drift"]["categorical"].items():
        psi_value = values["psi"]
        p_value = values["chi2"]["p_value"]
        if finite(psi_value):
            lines.append(f'cisia_drift_psi{{feature="{feature}"}} {psi_value}')
            if psi_value >= psi_warning:
                alert = 1
                severity = max(severity, 2)
            elif psi_value >= psi_watch:
                severity = max(severity, 1)
        if finite(p_value):
            lines.append(f'cisia_drift_chi2_pvalue{{feature="{feature}"}} {p_value}')
            if p_value < 0.05:
                alert = 1
                severity = max(severity, 1)
    for feature, values in report["confidence_drift"]["numeric"].items():
        psi_value = values["psi"]
        class_name = feature.removeprefix("proba_")
        if finite(psi_value):
            lines.append(f'cisia_confidence_drift_psi{{class="{class_name}"}} {psi_value}')
            if psi_value >= psi_warning:
                alert = 1
                severity = max(severity, 2)
            elif psi_value >= psi_watch:
                severity = max(severity, 1)
    generated_at = report.get("generated_at_epoch", time.time())
    lines.extend([
        "# HELP cisia_drift_alert 1 when a drift signal needs investigation",
        "# TYPE cisia_drift_alert gauge",
        f"cisia_drift_alert {alert}",
        "# HELP cisia_drift_alert_severity 0 none, 1 warning, 2 critical",
        "# TYPE cisia_drift_alert_severity gauge",
        f"cisia_drift_alert_severity {severity}",
        "# HELP cisia_drift_report_timestamp_seconds Unix timestamp of the latest drift report",
        "# TYPE cisia_drift_report_timestamp_seconds gauge",
        f"cisia_drift_report_timestamp_seconds {generated_at}",
        "# HELP cisia_drift_alert_timestamp_seconds Unix timestamp of the latest detected alert",
        "# TYPE cisia_drift_alert_timestamp_seconds gauge",
        f"cisia_drift_alert_timestamp_seconds {generated_at if alert else 0}",
        "# HELP cisia_drift_alert_owner_info Owner responsible for drift investigation",
        "# TYPE cisia_drift_alert_owner_info gauge",
        'cisia_drift_alert_owner_info{owner="ML Platform"} 1',
    ])
    return "\n".join(lines) + "\n"


def json_safe(value):
    """Replace non-finite statistical values with JSON null."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/drift.json"))
    parser.add_argument("--prometheus-output", type=Path, default=None)
    args = parser.parse_args()
    report = analyse(pd.read_csv(args.reference), pd.read_csv(args.current))
    report["generated_at_epoch"] = time.time()
    report = json_safe(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    if args.prometheus_output:
        args.prometheus_output.parent.mkdir(parents=True, exist_ok=True)
        args.prometheus_output.write_text(prometheus_metrics(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
