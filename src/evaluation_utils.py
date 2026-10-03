from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def make_json_serializable(value: Any) -> Any:
    """Convertit récursivement les valeurs NumPy, Pandas et Path en JSON-safe."""
    if isinstance(value, dict):
        return {str(key): make_json_serializable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [make_json_serializable(item) for item in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, pd.Series):
        return make_json_serializable(value.to_dict())

    return value


def compute_sha256(file_path: Path) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path.resolve()}")

    return sha256(file_path.read_bytes()).hexdigest()


def print_banner(title: str, width: int = 90) -> None:
    """Affiche un titre de section homogène dans la console."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_metrics(metrics: dict[str, Any], report_text: str) -> None:
    """Affiche le résumé de métriques d'une évaluation."""
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"F1 macro          : {metrics['f1_macro']:.4f}")
    print(f"F1 weighted       : {metrics['f1_weighted']:.4f}")
    print(f"F1 classe 0       : {metrics['f1_class_0']:.4f}")
    print(f"F1 classe 1       : {metrics['f1_class_1']:.4f}")
    print(f"F1 classe 2       : {metrics['f1_class_2']:.4f}")
    print(f"Recall classe 2    : {metrics['recall_classe_2']:.4f}")
    print(f"Precision classe 2 : {metrics['precision_classe_2']:.4f}")
    print(
        "Erreur critique 2 -> 0 : "
        f"{metrics['erreur_critique_2_vers_0']}"
    )
    print(
        "Erreur critique 0 -> 2 : "
        f"{metrics['erreur_critique_0_vers_2']}"
    )
    print(f"Latence p95       : {metrics['latency_p95_ms']:.3f} ms")
    print(
        "Temps CPU / 1 000 prédictions : "
        f"{metrics['cpu_ms_1000_predictions'] / 1_000:.3f} ms"
    )

    roc_auc = metrics.get("roc_auc_ovr_macro")
    if roc_auc is None:
        print("ROC-AUC OVR macro : non disponible")
    else:
        print(f"ROC-AUC OVR macro : {roc_auc:.4f}")

    print("\nMatrice de confusion :")
    print(np.asarray(metrics["confusion_matrix"]))

    print("\nClassification report :")
    print(report_text)


def create_benchmark(results: list[dict[str, Any]], output_path: Path) -> pd.DataFrame:
    """Produit le benchmark de tous les modèles évalués."""
    rows: list[dict[str, Any]] = []

    for result in results:
        metrics = result["metrics"]
        rows.append(
            {
                "model_type": result["model_type"],
                "scenario": result["scenario_name"],
                "config": result["config_name"],
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "recall_classe_2": metrics.get("recall_classe_2", np.nan),
                "precision_classe_2": metrics.get("precision_classe_2", np.nan),
                "erreur_critique_2_vers_0": metrics.get(
                    "erreur_critique_2_vers_0",
                    np.nan,
                ),
                "erreur_critique_0_vers_2": metrics.get(
                    "erreur_critique_0_vers_2",
                    np.nan,
                ),
                #"f1_weighted": metrics["f1_weighted"],
                #"f1_classe_0": metrics["f1_class_0"],
                #"f1_classe_1": metrics["f1_class_1"],
                "f1_classe_2": metrics["f1_class_2"],
                "roc_auc_ovr_macro": metrics["roc_auc_ovr_macro"],
                "latency_p95_ms": metrics["latency_p95_ms"],
                "cpu_mean_ms": metrics["cpu_mean_ms"],
            }
        )

    benchmark = pd.DataFrame(rows)

    if benchmark.empty:
        return benchmark

    benchmark = (
        benchmark.sort_values(
            by=["f1_macro", "f1_classe_2", "roc_auc_ovr_macro"],
            ascending=[False, False, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    benchmark.insert(0, "rang", range(1, len(benchmark) + 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark.to_csv(output_path, index=False, encoding="utf-8-sig")

    return benchmark


def save_benchmark_chart(benchmark: pd.DataFrame, output_path: Path) -> None:
    """Génère un graphique comparatif des modèles et scénarios."""
    if benchmark.empty:
        return

    chart_source = benchmark.copy()
    chart_source["combinaison"] = chart_source["model_type"] + "\n" + chart_source["scenario"]

    chart_data = chart_source[
        ["combinaison", "accuracy", "f1_macro", "f1_classe_2", "roc_auc_ovr_macro"]
    ].melt(id_vars="combinaison", var_name="Métrique", value_name="Score")

    width = max(14, len(chart_source) * 1.4)

    plt.figure(figsize=(width, 8))
    sns.barplot(data=chart_data, x="combinaison", y="Score", hue="Métrique")

    plt.ylim(0, 1)
    plt.title(
        "Benchmark CISIA\n"
        "Random Forest, Logistic Regression, LightGBM et XGBoost",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Algorithme et scénario")
    plt.ylabel("Score")
    plt.xticks(rotation=35, ha="right")
    plt.legend(title="Métrique")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def save_failures(failures: list[dict[str, str]], output_path: Path) -> None:
    """Sauvegarde les combinaisons non évaluées."""
    if not failures:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures).to_csv(output_path, index=False, encoding="utf-8-sig")


def print_benchmark(benchmark: pd.DataFrame) -> None:
    """Affiche le benchmark dans la console."""
    if benchmark.empty:
        print("\nAucun modèle n'a été évalué.")
        return

    columns_to_display = [
        "rang",
        "model_type",
        "scenario",
        "accuracy",
        "f1_macro",
        "f1_classe_2",
        "recall_classe_2",
        "precision_classe_2",
        "erreur_critique_2_vers_0",
        "erreur_critique_0_vers_2",
        "roc_auc_ovr_macro",
        "latency_p95_ms",
        "cpu_mean_ms",
    ]

    print_banner("BENCHMARK GLOBAL CISIA", width=130)
    display_benchmark = benchmark[columns_to_display].copy()
    display_benchmark = display_benchmark.rename(
        columns={
            "latency_p95_ms": "Latence p95 (ms)",
            "cpu_mean_ms": "CPU moyen / prédiction (ms)",
        }
    )
    print(display_benchmark.round(4).to_string(index=False))
    print("=" * 130)

    best_row = benchmark.iloc[0]
    print("\nMeilleure combinaison selon le F1 macro :")
    print(f"  Algorithme    : {best_row['model_type']}")
    print(f"  Scénario      : {best_row['scenario']}")
    print(f"  Configuration : {best_row['config']}")
    print(f"  F1 macro     : {best_row['f1_macro']:.4f}")
    print(f"  F1 classe 2  : {best_row['f1_classe_2']:.4f}")
    print(f"  Latence p95  : {best_row['latency_p95_ms']:.3f} ms")
    print(
        "  CPU moyen / prédiction : "
        f"{best_row['cpu_mean_ms']:.3f} ms"
    )

    roc_auc = best_row["roc_auc_ovr_macro"]
    if pd.notna(roc_auc):
        print(f"  ROC-AUC      : {roc_auc:.4f}")
