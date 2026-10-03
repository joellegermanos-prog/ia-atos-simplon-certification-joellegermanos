"""Évaluation continue multiclasses du modèle CISIA + tracking MLflow.

À chaque release : recalcule les métriques cibles sur un jeu de référence
figé, **trace le run dans MLflow**, compare aux seuils, et **sort un code
retour non-zéro** si dégradation (→ bloque la release en CI).

Usage cible::

    python scripts/evaluate_model.py --freeze-baseline             # une fois, au gel du jeu
    python scripts/evaluate_model.py --release-tag v2.0.0
    python scripts/evaluate_model.py --release-tag bad --degrade   # test du rouge
    mlflow ui    # comparer les runs

⚠️ **Le piège central du brief.** La tentation est de comparer vos métriques à
la baseline holdout annoncée en M1 (`metrics_holdout` dans le `.json`). Ne le
faites pas : le holdout et votre jeu de référence n'ont ni la même taille ni la
même composition. Vous mesureriez l'écart entre **deux populations**, pas la
dégradation du **modèle** — et votre garde-fou se déclencherait tout seul.
La baseline du garde-fou, c'est le **golden run** : les métriques mesurées sur
**votre** jeu de référence, au moment où vous le gelez.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).parent.parent
SERVICE_ROOT = ROOT / "services" / "model"
sys.path.insert(0, str(SERVICE_ROOT))
from preprocess import create_features

MODELS_DIR = SERVICE_ROOT / "models"
REFERENCE_SET = ROOT / "data" / "reference_set.csv"
REFERENCE_BASELINE = ROOT / "data" / "reference_baseline.json"
SOURCE_DATASET = ROOT / "data" / "dataset_trajectoire_emploi.csv"
MODEL_META = MODELS_DIR / "cisia_emploi_xgboost_multimodal_ethique_best_class_2_ethique.json"
DEFAULT_PROMETHEUS_OUTPUT = ROOT / "reports" / "evaluation.prom"

# Seuils conservateurs pour bloquer une release uniquement sur une vraie
# dégradation du modèle sur le jeu de référence gelé.
# - plancher absolu : on ne lâche pas sous un minimum acceptable
# - tolérance relative : on tolère une baisse limitée par rapport au golden run
# Ces valeurs sont choisies pour rester plus larges que le bruit de mesure du
# jeu de référence (ici ~0.02). On garde donc des marges sûres sans se
# déclencher sur du bruit de sampling.
THRESHOLDS: dict[str, dict[str, float]] = {
    "f1_macro": {"absolute_min": 0.60, "max_drop_vs_baseline": 0.04},
    "f1_classe_2": {"absolute_min": 0.50, "max_drop_vs_baseline": 0.05},
    "recall_classe_2": {"absolute_min": 0.59, "max_drop_vs_baseline": 0.06},
    "roc_auc_ovr_macro": {"absolute_min": 0.70, "max_drop_vs_baseline": 0.04},
}


def compute_metrics(model, df: pd.DataFrame, meta: dict) -> dict[str, float]:
    """Calcule les métriques cibles sur le jeu de référence."""
    target_col = meta["target"]["column"]
    X = create_features(df.drop(columns=[target_col]))
    y_true = df[target_col].astype(int)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_classe_2": float(f1_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]),
        "recall_classe_2": float(recall_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]),
        "precision_classe_2": float(precision_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]),
        "roc_auc_ovr_macro": float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")),
    }


def check_thresholds(metrics: dict[str, float], baseline: dict) -> list[str]:
    """Retourne la liste des violations de seuil (vide = release OK)."""
    baseline_metrics = baseline["metrics"]
    violations: list[str] = []

    for metric_name, cfg in THRESHOLDS.items():
        current = float(metrics.get(metric_name, -1.0))
        baseline_value = float(baseline_metrics.get(metric_name, 0.0))
        abs_min = float(cfg["absolute_min"])
        max_drop = float(cfg["max_drop_vs_baseline"])

        if current < abs_min:
            violations.append(
                f"{metric_name}={current:.4f} < absolute_min={abs_min:.4f}"
            )

        drop = baseline_value - current
        if drop > max_drop:
            violations.append(
                f"{metric_name} drop={drop:.4f} > max_drop_vs_baseline={max_drop:.4f} "
                f"(baseline={baseline_value:.4f}, current={current:.4f})"
            )

    return violations


def render_prometheus_metrics(
    metrics: dict[str, float],
    baseline: dict,
    violations: list[str],
    generated_at: float | None = None,
) -> str:
    """Render the latest multi-metric release-gate result for Prometheus."""
    baseline_metrics = baseline["metrics"]
    lines = [
        "# HELP cisia_evaluation_gate_status 1 when the latest release gate passed",
        "# TYPE cisia_evaluation_gate_status gauge",
        f"cisia_evaluation_gate_status {int(not violations)}",
        "# HELP cisia_evaluation_violations_total Number of failed release-gate checks",
        "# TYPE cisia_evaluation_violations_total gauge",
        f"cisia_evaluation_violations_total {len(violations)}",
        "# HELP cisia_evaluation_metric_value Latest metric measured on the reference set",
        "# TYPE cisia_evaluation_metric_value gauge",
        "# HELP cisia_evaluation_metric_absolute_min Absolute quality floor per metric",
        "# TYPE cisia_evaluation_metric_absolute_min gauge",
        "# HELP cisia_evaluation_metric_baseline Golden-run metric value",
        "# TYPE cisia_evaluation_metric_baseline gauge",
        "# HELP cisia_evaluation_metric_drop_vs_baseline Baseline minus latest metric value",
        "# TYPE cisia_evaluation_metric_drop_vs_baseline gauge",
        "# HELP cisia_evaluation_metric_max_drop_vs_baseline Maximum permitted metric drop",
        "# TYPE cisia_evaluation_metric_max_drop_vs_baseline gauge",
        "# HELP cisia_evaluation_timestamp_seconds Unix time of the latest completed evaluation",
        "# TYPE cisia_evaluation_timestamp_seconds gauge",
    ]
    for metric_name, thresholds in THRESHOLDS.items():
        metric_value = float(metrics[metric_name])
        baseline_value = float(baseline_metrics[metric_name])
        labels = f'{{metric="{metric_name}"}}'
        lines.extend([
            f"cisia_evaluation_metric_value{labels} {metric_value}",
            f"cisia_evaluation_metric_absolute_min{labels} {thresholds['absolute_min']}",
            f"cisia_evaluation_metric_baseline{labels} {baseline_value}",
            f"cisia_evaluation_metric_drop_vs_baseline{labels} {baseline_value - metric_value}",
            f"cisia_evaluation_metric_max_drop_vs_baseline{labels} {thresholds['max_drop_vs_baseline']}",
        ])
    timestamp = generated_at if generated_at is not None else time.time()
    lines.append(f"cisia_evaluation_timestamp_seconds {timestamp}")
    return "\n".join(lines) + "\n"


def write_prometheus_metrics(
    output_path: Path,
    metrics: dict[str, float],
    baseline: dict,
    violations: list[str],
) -> None:
    """Atomically publish the latest gate result for the backend scrape endpoint."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        render_prometheus_metrics(metrics, baseline, violations),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def load_baseline() -> dict:
    """Charge le golden run (baseline mesurée sur le jeu de référence)."""
    if not REFERENCE_BASELINE.exists():
        raise SystemExit(
            f"{REFERENCE_BASELINE} est absent. Lancez d'abord : "
            "python scripts/evaluate_model.py --freeze-baseline"
        )

    baseline = json.loads(REFERENCE_BASELINE.read_text(encoding="utf-8"))
    if "metrics" not in baseline:
        raise SystemExit(f"{REFERENCE_BASELINE} ne contient pas de bloc 'metrics'.")
    return baseline


def freeze_baseline(model, df: pd.DataFrame, meta: dict) -> dict:
    """Mesure et gèle le golden run sur le jeu de référence.

    Le fichier JSON produit est la baseline de référence (golden run) que les
    futures releases devront comparer à leur sortie.
    """
    target_col = meta["target"]["column"]
    metrics = compute_metrics(model, df, meta)

    payload = {
        "model_version": meta["model_version"],
        "reference_set": (
            "data/reference_set.csv"
            if REFERENCE_SET.exists()
            else "data/dataset_trajectoire_emploi.csv:test_indices"
        ),
        "n_reference": len(df),
        "metrics": metrics,
        "target_column": target_col,
        "target_values": meta["target"]["expected_values"],
    }

    REFERENCE_BASELINE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Golden run saved to {REFERENCE_BASELINE}")
    print(json.dumps(payload, indent=2))
    return payload


def load_reference_set() -> pd.DataFrame:
    """Charge le jeu de référence, avec un garde-fou sur sa validité.

    Le jeu de référence est VOTRE instrument de mesure : vous le construisez
    à partir du holdout M1 (cf. `data/README.md`). Le fichier
    `reference_set_TEMPLATE.csv` livré dans le repo est un **exemple de
    format** de 20 lignes, pas un jeu de référence utilisable.
    """
    target_column = "classe_retour_emploi"
    df = pd.read_csv(REFERENCE_SET) if REFERENCE_SET.exists() else pd.DataFrame()

    # Le fichier historique reference_set.csv peut encore être le jeu bancaire.
    # Dans ce cas, reconstruire le holdout CISIA depuis les indices versionnés.
    if target_column not in df.columns:
        if not SOURCE_DATASET.exists() or not MODEL_META.exists():
            raise SystemExit(
                "Impossible de construire le jeu de référence CISIA : "
                "dataset ou métadonnées manquants."
            )
        source = pd.read_csv(SOURCE_DATASET)
        meta = json.loads(MODEL_META.read_text(encoding="utf-8"))
        df = source.loc[meta["dataset"]["test_indices"]].copy()

    if len(df) < 100 or df[target_column].nunique() < 3:
        raise SystemExit(
            f"{REFERENCE_SET} contient {len(df)} ligne(s) et "
            f"{df.get(target_column, pd.Series(dtype=int)).nunique()} classe(s) de cible.\n"
            "Un instrument de mesure multiclasses doit contenir les TROIS classes.\n"
            "Avez-vous copié reference_set_TEMPLATE.csv ? C'est un exemple de "
            "format, pas un jeu de référence — cf. data/README.md."
        )
    return df


def extract_model_hyperparameters(meta: dict) -> dict[str, object]:
    """Read hyperparameters from both legacy and CISIA metadata layouts."""
    legacy = meta.get("hyperparameters")
    if isinstance(legacy, dict):
        return legacy
    configuration = meta.get("configuration")
    if isinstance(configuration, dict):
        parameters = configuration.get("model_parameters")
        if isinstance(parameters, dict):
            return parameters
    return {}


def build_mlflow_params(meta: dict, release_tag: str, n_reference: int) -> dict:
    """Construit les params MLflow à partir du JSON du modèle, pas à la main."""
    params: dict[str, object] = {
        "model_version": meta.get("model_version", "unknown"),
        "release_tag": release_tag,
        "reference_set": str(REFERENCE_SET.name),
        "n_reference": int(n_reference),
        "target_column": meta.get("target", {}).get(
            "column",
            meta.get("target_column", "unknown"),
        ),
        "dataset_sha256": meta.get("dataset", {}).get(
            "sha256",
            meta.get("dataset_sha256", "unknown"),
        ),
    }

    hyperparams = extract_model_hyperparameters(meta)
    for key, value in hyperparams.items():
        params[f"hyperparameters.{key}"] = (
            value if isinstance(value, (str, int, float, bool)) else json.dumps(value)
        )

    return params


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-tag", default="dev")
    parser.add_argument("--degrade", action="store_true")
    parser.add_argument("--freeze-baseline", action="store_true")
    parser.add_argument("--prometheus-output", type=Path, default=DEFAULT_PROMETHEUS_OUTPUT)
    args = parser.parse_args()

    model = joblib.load(MODEL_META.with_suffix(".joblib"))
    meta = json.loads(MODEL_META.read_text(encoding="utf-8"))
    df = load_reference_set()

    if args.freeze_baseline:
        print(json.dumps(freeze_baseline(model, df, meta), indent=2))
        return 0

    if args.degrade:
        # Simule un bug de preprocessing réaliste : les labels sont permutés
        # sans changer les features, ce qui casse l'alignement X / y.
        df = df.copy()
        df["classe_retour_emploi"] = (
            df["classe_retour_emploi"]
            .sample(frac=1.0, random_state=42)
            .to_numpy()
        )

    metrics = compute_metrics(model, df, meta)
    baseline = load_baseline()  # ← le golden run, PAS metrics_holdout
    violations = check_thresholds(metrics, baseline)
    write_prometheus_metrics(args.prometheus_output, metrics, baseline, violations)

    # --- Bloc MLflow — params lus depuis le JSON du modèle ------------------
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("cisia-emploi-eval-continue")
    with mlflow.start_run(run_name=args.release_tag):
        mlflow.log_params(build_mlflow_params(meta, args.release_tag, len(df)))
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_type", meta.get("model_type", "unknown"))
        mlflow.set_tag("config_name", meta.get("config_name", "unknown"))
        mlflow.log_dict(meta, "model_metadata.json")
        mlflow.set_tag("status", "failed" if violations else "passed")
        mlflow.set_tag("release_blocked", str(bool(violations)))
    # ------------------------------------------------------------------------

    print(json.dumps({"metrics": metrics, "violations": violations}, indent=2))
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
