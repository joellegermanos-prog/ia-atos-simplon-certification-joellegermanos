"""
Projet CISIA
Évaluation des modèles sur le holdout réservé par train.py.

Algorithmes :
    - random_forest
    - logistic_regression
    - lightgbm
    - xgboost

Scénarios :
    - multimodal_complet
    - multimodal_ethique
    - texte_seul
    - tabulaire_seul

Le script :
    - charge le modèle et ses métadonnées ;
    - vérifie que le dataset n'a pas changé ;
    - reconstruit le holdout depuis les test_indices ;
    - calcule les métriques ;
    - sauvegarde les résultats ;
    - génère un benchmark global.

Exemples :

    # Évaluer les 16 combinaisons
    python src/evaluate.py --model-type all --scenario all --config balanced

    # Évaluer uniquement le meilleur modèle
    python src/evaluate.py --model-type xgboost --scenario multimodal_complet --config balanced

    # Ignorer les modèles absents
    python src/evaluate.py --model-type all --scenario all --config balanced --skip-missing

    # Ne pas modifier les fichiers metadata JSON
    python src/evaluate.py --model-type all --scenario all --config balanced --no-update-meta
"""

from __future__ import annotations

import argparse
import json

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, process_time
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    AVAILABLE_CONFIGS,
    AVAILABLE_MODEL_TYPES,
    DATA_PATH,
    MODEL_NAME,
    MODELS_DIR,
    REPORTS_DIR,
    SCENARIOS,
)
from evaluation_utils import (
    compute_sha256,
    make_json_serializable,
    print_banner,
    print_metrics,
)
from reporting import (
    create_benchmark,
    print_benchmark,
    save_benchmark_chart,
    save_failures,
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    recall_score,
)

from preprocess import (
    TARGET_COLUMN,
    TEXT_FEATURE,
    load_dataset,
    validate_target,
)


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

CLASS_LABELS = [0, 1, 2]

CLASS_NAMES = [
    "Retour rapide",
    "Retour moyen",
    "Risque longue durée",
]

AVAILABLE_MODEL_TYPES = AVAILABLE_MODEL_TYPES
SCENARIOS = tuple(SCENARIOS.keys())
AVAILABLE_CONFIGS = AVAILABLE_CONFIGS
DEFAULT_DATA_PATH = DATA_PATH
DEFAULT_MODELS_DIR = MODELS_DIR
DEFAULT_REPORTS_DIR = REPORTS_DIR


def build_artifact_name(
    model_type: str,
    scenario_name: str,
    config_name: str,
    prefix: str = MODEL_NAME,
) -> str:
    """Construit un nom d'artefact cohérent pour les fichiers de sortie."""
    return f"{prefix}_{model_type}_{scenario_name}_{config_name}"


def print_section(title: str) -> None:
    """Compatibilité : affiche un titre de section cohérent pour la console."""
    print_banner(title, width=90)


def ensure_parent_directory(path: Path) -> None:
    """Crée le dossier parent d'un fichier si nécessaire."""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json_file(path: Path, payload: Any) -> None:
    """Écrit un objet Python en JSON sans modifier la sémantique de sortie."""
    ensure_parent_directory(path)
    path.write_text(
        json.dumps(
            make_json_serializable(payload),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# =============================================================================
# 2. CHARGEMENT DU MODÈLE ET DES MÉTADONNÉES
# =============================================================================

def load_metadata(
    metadata_path: Path,
) -> dict[str, Any]:
    """Charge le fichier JSON créé par train.py."""

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Métadonnées introuvables : {metadata_path.resolve()}"
        )

    try:
        return json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as error:
        raise ValueError(
            f"JSON invalide : {metadata_path.resolve()}"
        ) from error


def load_model(
    model_path: Path,
):
    """Charge le pipeline entraîné."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path.resolve()}"
        )

    try:
        return joblib.load(
            model_path
        )

    except Exception as error:
        raise RuntimeError(
            f"Impossible de charger le modèle : {model_path.resolve()}"
        ) from error


def validate_dataset(
    data_path: Path,
    metadata: dict[str, Any],
) -> None:
    """Vérifie que le dataset est identique à celui du train."""

    dataset_metadata = metadata.get("dataset", {})
    expected_hash = dataset_metadata.get("sha256")

    if expected_hash is None:
        raise KeyError("Le metadata de train n'inclut pas le hash du dataset.")

    current_hash = compute_sha256(data_path)

    if current_hash != expected_hash:
        raise ValueError(
            "Le dataset a changé depuis l'entraînement.\n"
            f"Hash attendu : {expected_hash}\n"
            f"Hash obtenu  : {current_hash}"
        )

    print("Intégrité du dataset : OK")


# =============================================================================
# 4. CHARGEMENT DES DONNÉES
# =============================================================================

def load_text_dataset(
    data_path: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Charge uniquement la colonne textuelle et la cible.

    La structure correspond à celle utilisée dans train.py :
    un DataFrame contenant exactement une colonne texte.
    """

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {data_path.resolve()}"
        )

    df = pd.read_csv(
        data_path,
        dtype={
            "usager_id": "string",
            "code_rome_vise": "string",
            "code_insee_commune": "string",
        },
    )

    required_columns = {
        TEXT_FEATURE,
        TARGET_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise KeyError(
            f"Colonnes absentes : {sorted(missing_columns)}"
        )

    X = df[
        [TEXT_FEATURE]
    ].copy()

    X[TEXT_FEATURE] = (
        X[TEXT_FEATURE]
        .fillna("")
        .astype(str)
    )

    y = validate_target(
        df[TARGET_COLUMN]
    )

    return X, y


def load_scenario_data(
    data_path: Path,
    metadata: dict[str, Any],
):
    """Charge les données selon le scénario enregistré."""

    scenario = metadata[
        "scenario"
    ]

    if scenario.get(
        "text_only",
        False,
    ):
        return load_text_dataset(
            data_path
        )

    return load_dataset(
        path=data_path,
        include_sensitive=scenario.get(
            "include_sensitive",
            True,
        ),
        include_text=scenario.get(
            "include_text",
            True,
        ),
        features=scenario.get(
            "features"
        ),
    )


# =============================================================================
# 5. RECONSTRUCTION DU HOLDOUT
# =============================================================================

def rebuild_holdout(
    data_path: Path,
    metadata: dict[str, Any],
):
    """
    Reconstruit exactement le holdout créé dans train.py.

    Aucun nouveau train_test_split n'est réalisé.
    """

    X, y = load_scenario_data(
        data_path=data_path,
        metadata=metadata,
    )

    test_indices = [
        int(index)
        for index in metadata[
            "dataset"
        ][
            "test_indices"
        ]
    ]

    X_holdout = X.loc[
        test_indices
    ].copy()

    y_holdout = y.loc[
        test_indices
    ].copy()

    expected_rows = int(
        metadata[
            "dataset"
        ][
            "test_rows"
        ]
    )

    if len(X_holdout) != expected_rows:
        raise ValueError(
            "Nombre incorrect de lignes dans le holdout. "
            f"Attendu : {expected_rows}. "
            f"Obtenu : {len(X_holdout)}."
        )

    print(
        f"Lignes évaluées : {len(X_holdout)}"
    )

    print(
        "Source          : holdout réservé par train.py"
    )

    return X_holdout, y_holdout


# =============================================================================
# 6. CALCUL DES MÉTRIQUES
# =============================================================================

def measure_inference_latency(
    model,
    X_holdout: pd.DataFrame,
    max_samples: int = 100,
) -> dict[str, float | int]:
    """Mesure la latence d'une prédiction individuelle du pipeline complet."""
    if X_holdout.empty:
        raise ValueError("Impossible de mesurer la latence sur un holdout vide.")

    number_of_samples = min(max_samples, len(X_holdout))
    latencies_ms: list[float] = []
    cpu_times_ms: list[float] = []

    model.predict(X_holdout.iloc[[0]])

    for index in range(number_of_samples):
        observation = X_holdout.iloc[[index]]
        start_time = perf_counter()
        cpu_start_time = process_time()
        model.predict(observation)
        latencies_ms.append((perf_counter() - start_time) * 1000)
        cpu_times_ms.append((process_time() - cpu_start_time) * 1000)

    latency_array = np.asarray(latencies_ms)
    cpu_array = np.asarray(cpu_times_ms)

    return {
        "latency_mean_ms": float(latency_array.mean()),
        "latency_median_ms": float(np.median(latency_array)),
        "latency_p95_ms": float(np.percentile(latency_array, 95)),
        "cpu_mean_ms": float(cpu_array.mean()),
        "cpu_p95_ms": float(np.percentile(cpu_array, 95)),
        "cpu_ms_1000_predictions": float(cpu_array.mean() * 1_000),
        "latency_samples": int(number_of_samples),
    }

def evaluate_model(
    model,
    X_holdout,
    y_holdout: pd.Series,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray | None,
    str,
]:
    """Calcule les métriques sur le holdout."""

    latency_metrics = measure_inference_latency(
        model=model,
        X_holdout=X_holdout,
    )

    y_pred = np.asarray(
        model.predict(
            X_holdout
        )
    )

    y_proba: np.ndarray | None = None

    if hasattr(
        model,
        "predict_proba",
    ):
        y_proba = np.asarray(
            model.predict_proba(
                X_holdout
            )
        )

    f1_classes = f1_score(
        y_holdout,
        y_pred,
        labels=CLASS_LABELS,
        average=None,
        zero_division=0,
    )

    report_dictionary = classification_report(
        y_holdout,
        y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_holdout,
        y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    confusion = confusion_matrix(
        y_holdout,
        y_pred,
        labels=CLASS_LABELS,
    )

    metrics: dict[str, Any] = {
        **latency_metrics,

        "accuracy": float(
            accuracy_score(
                y_holdout,
                y_pred,
            )
        ),

        "f1_macro": float(
            f1_score(
                y_holdout,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "f1_weighted": float(
            f1_score(
                y_holdout,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),

        "f1_class_0": float(
            f1_classes[0]
        ),

        "f1_class_1": float(
            f1_classes[1]
        ),

        "f1_class_2": float(
            f1_classes[2]
        ),

        "recall_classe_2": float(
            recall_score(
                y_holdout,
                y_pred,
                labels=[2],
                average=None,
                zero_division=0,
            )[0]
        ),

        "precision_classe_2": float(
            precision_score(
                y_holdout,
                y_pred,
                labels=[2],
                average=None,
                zero_division=0,
            )[0]
        ),

        "erreur_critique_2_vers_0": int(
            confusion[2, 0]
        ),

        "erreur_critique_0_vers_2": int(
            confusion[0, 2]
        ),

        "confusion_matrix": confusion.tolist(),

        "classification_report": (
            report_dictionary
        ),
    }

    metrics[
        "roc_auc_ovr_macro"
    ] = None

    if y_proba is not None:
        try:
            metrics[
                "roc_auc_ovr_macro"
            ] = float(
                roc_auc_score(
                    y_holdout,
                    y_proba,
                    labels=CLASS_LABELS,
                    multi_class="ovr",
                    average="macro",
                )
            )

        except ValueError as error:
            metrics[
                "roc_auc_warning"
            ] = str(error)

    return (
        metrics,
        y_pred,
        y_proba,
        report_text,
    )


# =============================================================================
# 7. SAUVEGARDE DES RÉSULTATS
# =============================================================================

def save_outputs(
    artifact_name: str,
    model_type: str,
    scenario_name: str,
    reports_dir: Path,
    metrics: dict[str, Any],
    y_holdout: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
) -> dict[str, Path]:
    """
    Sauvegarde :
        - matrice de confusion ;
        - classification report ;
        - prédictions.
    """

    reports_dir.mkdir(parents=True, exist_ok=True)

    confusion_path = (
        reports_dir
        / f"{artifact_name}_confusion_matrix.png"
    )

    report_path = (
        reports_dir
        / f"{artifact_name}_classification_report.csv"
    )

    predictions_path = (
        reports_dir
        / f"{artifact_name}_predictions.csv"
    )

    # Matrice de confusion
    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(
        np.asarray(
            metrics[
                "confusion_matrix"
            ]
        ),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )

    plt.title(
        "Matrice de confusion\n"
        f"{model_type} | {scenario_name}"
    )

    plt.xlabel(
        "Classe prédite"
    )

    plt.ylabel(
        "Classe réelle"
    )

    plt.tight_layout()

    plt.savefig(
        confusion_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    # Classification report
    pd.DataFrame(
        metrics[
            "classification_report"
        ]
    ).transpose().to_csv(
        report_path,
        encoding="utf-8-sig",
    )

    # Prédictions
    predictions = pd.DataFrame(
        {
            "dataset_index": (
                y_holdout.index
            ),
            "y_true": (
                y_holdout.to_numpy()
            ),
            "y_pred": (
                y_pred
            ),
        }
    )

    predictions[
        "prediction_correcte"
    ] = (
        predictions[
            "y_true"
        ]
        == predictions[
            "y_pred"
        ]
    )

    if (
        y_proba is not None
        and y_proba.ndim == 2
        and y_proba.shape[1] == len(CLASS_LABELS)
    ):
        for position, class_label in enumerate(
            CLASS_LABELS
        ):
            predictions[
                f"probabilite_classe_{class_label}"
            ] = y_proba[
                :,
                position
            ]

    predictions.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "confusion_matrix": confusion_path,
        "classification_report": report_path,
        "predictions": predictions_path,
    }


# =============================================================================
# 8. AFFICHAGE
# =============================================================================

# =============================================================================
# 9. ÉVALUATION D'UNE COMBINAISON
# =============================================================================

def evaluate_combination(
    model_type: str,
    scenario_name: str,
    config_name: str,
    data_path: Path,
    models_dir: Path,
    reports_dir: Path,
    update_metadata: bool,
) -> dict[str, Any]:
    """Évalue une combinaison algorithme et scénario."""

    artifact_name = build_artifact_name(
        model_type=model_type,
        scenario_name=scenario_name,
        config_name=config_name,
    )

    model_path = models_dir / f"{artifact_name}.joblib"
    metadata_path = models_dir / f"{artifact_name}.json"
    evaluation_path = reports_dir / f"{artifact_name}_evaluation.json"

    print_section(f"ÉVALUATION : {model_type} | {scenario_name}")

    metadata = load_metadata(
        metadata_path
    )

    validate_dataset(
        data_path=data_path,
        metadata=metadata,
    )

    model = load_model(
        model_path
    )

    X_holdout, y_holdout = rebuild_holdout(
        data_path=data_path,
        metadata=metadata,
    )

    (
        metrics,
        y_pred,
        y_proba,
        report_text,
    ) = evaluate_model(
        model=model,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
    )

    output_paths = save_outputs(
        artifact_name=artifact_name,
        model_type=model_type,
        scenario_name=scenario_name,
        reports_dir=reports_dir,
        metrics=metrics,
        y_holdout=y_holdout,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    evaluation_result = {
        "status": "completed",

        "evaluated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "model_name": MODEL_NAME,

        "model_type": model_type,

        "scenario_name": scenario_name,

        "config_name": config_name,

        "evaluation_dataset": {
            "type": "holdout",
            "rows": int(
                len(X_holdout)
            ),
            "indices_source": (
                "metadata.dataset.test_indices"
            ),
            "sha256": metadata[
                "dataset"
            ][
                "sha256"
            ],
        },

        "metrics": metrics,

        "artifacts": {
            "model_path": str(
                model_path
            ),
            "metadata_path": str(
                metadata_path
            ),
            "evaluation": str(
                evaluation_path
            ),
            **{
                key: str(path)
                for key, path in output_paths.items()
            },
        },
    }

    write_json_file(
        path=evaluation_path,
        payload=evaluation_result,
    )

    if update_metadata:
        metadata.setdefault(
            "training",
            {}
        )

        metadata[
            "training"
        ][
            "evaluation_status"
        ] = "completed"

        metadata[
            "evaluation"
        ] = evaluation_result

        write_json_file(
            path=metadata_path,
            payload=metadata,
        )

    print_metrics(
        metrics=metrics,
        report_text=report_text,
    )

    print(
        f"\nÉvaluation JSON : {evaluation_path.resolve()}"
    )

    return evaluation_result


# =============================================================================
# 10. ÉVALUATION DES COMBINAISONS DEMANDÉES
# =============================================================================

def evaluate_requested_models(
    requested_model_type: str,
    requested_scenario: str,
    config_name: str,
    data_path: Path,
    models_dir: Path,
    reports_dir: Path,
    update_metadata: bool,
    skip_missing: bool,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, str]],
]:
    """Évalue toutes les combinaisons demandées."""

    model_types = (
        list(AVAILABLE_MODEL_TYPES)
        if requested_model_type == "all"
        else [requested_model_type]
    )

    scenario_names = (
        list(SCENARIOS)
        if requested_scenario == "all"
        else [requested_scenario]
    )

    combinations = [
        (
            model_type,
            scenario_name,
        )
        for model_type in model_types
        for scenario_name in scenario_names
    ]

    results: list[
        dict[str, Any]
    ] = []

    failures: list[
        dict[str, str]
    ] = []

    for position, (
        model_type,
        scenario_name,
    ) in enumerate(
        combinations,
        start=1,
    ):
        print(
            f"\nCombinaison "
            f"{position}/{len(combinations)}"
        )

        try:
            result = evaluate_combination(
                model_type=model_type,
                scenario_name=scenario_name,
                config_name=config_name,
                data_path=data_path,
                models_dir=models_dir,
                reports_dir=reports_dir,
                update_metadata=update_metadata,
            )

            results.append(
                result
            )

        except Exception as error:

            if not skip_missing:
                raise

            failures.append(
                {
                    "model_type": model_type,
                    "scenario_name": scenario_name,
                    "config_name": config_name,
                    "error": str(error),
                }
            )

            print(
                f"Combinaison ignorée : {error}"
            )

    return results, failures


# =============================================================================
# 11. INTERFACE EN LIGNE DE COMMANDE
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description=(
            "Évalue les modèles CISIA sur leur holdout."
        )
    )

    parser.add_argument(
        "--model-type",
        default="random_forest",
        choices=[
            *AVAILABLE_MODEL_TYPES,
            "all",
        ],
    )

    parser.add_argument(
        "--scenario",
        default="all",
        choices=[
            *SCENARIOS,
            "all",
        ],
    )

    parser.add_argument(
        "--config",
        default="balanced",
        choices=list(
            AVAILABLE_CONFIGS
        ),
    )

    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        type=Path,
    )

    parser.add_argument(
        "--models-dir",
        default=DEFAULT_MODELS_DIR,
        type=Path,
    )

    parser.add_argument(
        "--reports-dir",
        default=DEFAULT_REPORTS_DIR,
        type=Path,
    )

    parser.add_argument(
        "--no-update-meta",
        action="store_true",
    )

    parser.add_argument(
        "--skip-missing",
        action="store_true",
    )

    return parser.parse_args()


def run_evaluation(args: argparse.Namespace) -> None:
    """Orchestre l'évaluation complète pour les combinaisons demandées."""
    print("\nParamètres d'évaluation :")
    print(f"  Modèle        : {args.model_type}")
    print(f"  Scénario      : {args.scenario}")
    print(f"  Configuration : {args.config}")
    print(f"  Dataset       : {args.data}")

    results, failures = evaluate_requested_models(
        requested_model_type=args.model_type,
        requested_scenario=args.scenario,
        config_name=args.config,
        data_path=args.data,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        update_metadata=(not args.no_update_meta),
        skip_missing=args.skip_missing,
    )

    benchmark_name = build_artifact_name(
        model_type=args.model_type,
        scenario_name=args.scenario,
        config_name=args.config,
        prefix="benchmark",
    )

    benchmark_csv_path = args.reports_dir / f"{benchmark_name}.csv"
    benchmark_chart_path = args.reports_dir / f"{benchmark_name}.png"
    benchmark = create_benchmark(results=results, output_path=benchmark_csv_path)

    if not benchmark.empty:
        save_benchmark_chart(benchmark=benchmark, output_path=benchmark_chart_path)
        print_benchmark(benchmark)
        print(f"\nBenchmark CSV : {benchmark_csv_path.resolve()}")
        print(f"Graphique     : {benchmark_chart_path.resolve()}")

    if failures:
        failures_path = args.reports_dir / f"{benchmark_name}_failures.csv"
        save_failures(failures=failures, output_path=failures_path)
        print(f"\nCombinaisons ignorées : {len(failures)}")
        print(f"Détail : {failures_path.resolve()}")


def main() -> None:
    """Point d'entrée de evaluate.py."""
    run_evaluation(
        args=parse_args()
    )


if __name__ == "__main__":
    main()