"""
Projet CISIA
Entraînement de quatre algorithmes sur quatre scénarios.

Algorithmes disponibles :
    - random_forest
    - logistic_regression
    - lightgbm
    - xgboost

Scénarios disponibles :
    - multimodal_complet
    - multimodal_ethique
    - texte_seul
    - tabulaire_seul

Responsabilités de ce script :
    1. charger les données correspondant au scénario ;
    2. réserver un holdout stratifié de 20 % ;
    3. réaliser une validation croisée uniquement sur le train ;
    4. entraîner le pipeline final uniquement sur le train ;
    5. sauvegarder le pipeline complet ;
    6. sauvegarder les métadonnées nécessaires à evaluate.py.

Important :
    - les métriques de validation croisée sont calculées uniquement
      sur le jeu d'entraînement ;
    - aucune métrique n'est calculée sur le holdout dans train.py ;
    - le holdout est évalué uniquement dans evaluate.py.

Exemples :

    # Un modèle et un scénario
    python src/train.py --model-type random_forest --scenario multimodal_complet --config balanced

    # Tous les modèles et tous les scénarios
    python src/train.py --model-type all --scenario all --config balanced

    # Désactiver la validation croisée
    python src/train.py --model-type random_forest --scenario texte_seul --config balanced --cv-folds 0
"""

from __future__ import annotations

import argparse
import json
import platform

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, process_time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import (
    compute_sample_weight,
)


# =============================================================================
# IMPORTS OPTIONNELS
# =============================================================================

try:
    import lightgbm
    from lightgbm import LGBMClassifier

except ImportError:
    lightgbm = None
    LGBMClassifier = None


try:
    import xgboost
    from xgboost import XGBClassifier

except ImportError:
    xgboost = None
    XGBClassifier = None


# =============================================================================
# IMPORTS DU PROJET
# =============================================================================

from config import (
    AVAILABLE_CONFIGS as CONFIG_NAMES,
    AVAILABLE_MODEL_TYPES as MODEL_TYPES,
    DATA_PATH,
    MODEL_NAME,
    MODEL_VERSION,
    MODELS_DIR,
    RANDOM_STATE,
    SCENARIOS as SCENARIO_CONFIGS,
    TEST_SIZE,
)

from evaluation_utils import (
    compute_sha256,
    make_json_serializable,
)
from preprocess import (
    EXPECTED_TARGET_VALUES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    TARGET_COLUMN,
    TEXT_FEATURE,
    build_preprocessor,
    build_text_pipeline,
    get_all_features,
    get_categorical_features,
    load_dataset,
    validate_target,
)


# =============================================================================
# 1. CONFIGURATION GÉNÉRALE
# =============================================================================

DEFAULT_DATA_PATH: Path = Path(
    DATA_PATH
)

DEFAULT_OUTPUT_DIR: Path = Path(
    MODELS_DIR
)

AVAILABLE_MODEL_TYPES: tuple[str, ...] = tuple(
    MODEL_TYPES
)

AVAILABLE_CONFIGS: tuple[str, ...] = tuple(
    CONFIG_NAMES
)

SCENARIOS: dict[str, dict[str, Any]] = dict(
    SCENARIO_CONFIGS
)

CLASS_LABELS: list[int] = [
    0,
    1,
    2,
]


# =============================================================================
# 2. CONFIGURATIONS DES MODÈLES
# =============================================================================

MODEL_CONFIGS: dict[
    str,
    dict[str, dict[str, Any]],
] = {
    "random_forest": {
        "default": {
            "n_estimators": 300,
            "min_samples_leaf": 1,
            "class_weight": None,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },

        "balanced": {
            "n_estimators": 400,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },

        "balanced_tuned": {
           "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 1,
            "class_weight": None,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        },
        
        "best_class_2_text": {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 1,
            "class_weight": None,
            "max_features": "sqrt",
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
        },
        
        "tuned1": {
           "n_estimators": 300,
            "min_samples_leaf": 1,
            "class_weight": None,
            "max_depth": None,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": 1,
        },

        "tuned2": {
           "n_estimators": 500,
            "min_samples_leaf": 2,
            "class_weight": None,
            "max_depth": 18,
            "max_features": 0.35,
            "random_state": 42,
            "n_jobs": 1,
        },
        "critical_class_2": {
            "n_estimators": 400,
            "min_samples_leaf": 2,
            "class_weight": {
                0: 1.0,
                1: 1.0,
                2: 2.5,
            },
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },

        "regularized": {
            "n_estimators": 400,
            "max_depth": 15,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
    },

    "logistic_regression": {
        "default": {
            "solver": "lbfgs",
            "max_iter": 1_000,
            "C": 1.0,
            "class_weight": None,
            "random_state": RANDOM_STATE,
        },

        "balanced": {
            "solver": "lbfgs",
            "max_iter": 1_500,
            "C": 1.0,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        },

        "balanced_tuned": {
            "solver": "lbfgs",
            "max_iter": 2_000,
            "C": 0.1,
            "class_weight": None,
            "random_state": RANDOM_STATE,
        },


        "critical_class_2": {
            "solver": "lbfgs",
            "max_iter": 1_500,
            "C": 1.0,
            "class_weight": {
                0: 1.0,
                1: 1.0,
                2: 2.5,
            },
            "random_state": RANDOM_STATE,
        },

        "regularized": {
            "solver": "lbfgs",
            "max_iter": 2_000,
            "C": 0.5,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        },
    },

    "lightgbm": {
        "default": {
            "objective": "multiclass",
            "num_class": 3,
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
            "verbosity": -1,
        },

        "balanced": {
            "objective": "multiclass",
            "num_class": 3,
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.90,
            "subsample_freq": 1,
            "colsample_bytree": 0.90,
            "reg_lambda": 1.0,
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
            "verbosity": -1,
        },

        "best_class_2_complet": {
            "objective": "multiclass",
            "num_class": 3,
            "class_weight": {0: 1.0, 1: 1.0, 2: 2.5},
            "colsample_bytree": 0.9,
            "learning_rate": 0.03,
            "min_child_samples": 10,
            "n_estimators": 350,
            "num_leaves": 15,
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
            "verbosity": -1,
        },

        "best_class_2_tab": {
            "objective": "multiclass",
            "num_class": 3,
            "class_weight": None,
            "colsample_bytree": 0.9,
            "learning_rate": 0.03,
            "min_child_samples": 10,
            "n_estimators": 150,
            "num_leaves": 15,
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
            "verbosity": -1,
        },

        "critical_class_2": {
            "objective": "multiclass",
            "num_class": 3,
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.90,
            "subsample_freq": 1,
            "colsample_bytree": 0.90,
            "reg_lambda": 1.0,
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
            "verbosity": -1,
        },

        "regularized": {
            "objective": "multiclass",
            "num_class": 3,
            "n_estimators": 400,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 8,
            "min_child_samples": 30,
            "subsample": 0.80,
            "subsample_freq": 1,
            "colsample_bytree": 0.80,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
            "verbosity": -1,
        },
    },

    "xgboost": {
        "default": {
            "objective": "multi:softprob",
            "num_class": 3,
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        },

        "balanced": {
            "objective": "multi:softprob",
            "num_class": 3,
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 2.0,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        },

        "best_class_2_ethique": {
            "objective": "multi:softprob",
            "num_class": 3,
            "colsample_bytree": 0.9,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 3,
            "n_estimators": 400,
            "subsample": 0.9,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        },

        "critical_class_2": {
            "objective": "multi:softprob",
            "num_class": 3,
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 2.0,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        },

        "regularized": {
            "objective": "multi:softprob",
            "num_class": 3,
            "n_estimators": 400,
            "learning_rate": 0.03,
            "max_depth": 4,
            "min_child_weight": 5.0,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        },
    },
}


# =============================================================================
# 3. VALIDATION DE LA DEMANDE
# =============================================================================

def validate_training_request(
    model_type: str,
    scenario_name: str,
    config_name: str,
) -> None:
    """
    Vérifie que le modèle, le scénario et la configuration existent.
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(
            f"Modèle inconnu : '{model_type}'. "
            f"Modèles disponibles : {list(MODEL_CONFIGS)}"
        )

    if scenario_name not in SCENARIOS:
        raise ValueError(
            f"Scénario inconnu : '{scenario_name}'. "
            f"Scénarios disponibles : {list(SCENARIOS)}"
        )

    available_configs = MODEL_CONFIGS[
        model_type
    ]

    if config_name not in available_configs:
        raise ValueError(
            f"Configuration '{config_name}' indisponible "
            f"pour le modèle '{model_type}'. "
            f"Configurations disponibles : "
            f"{list(available_configs)}"
        )


# =============================================================================
# 4. UTILITAIRES
# =============================================================================

def check_library(
    model_type: str,
) -> None:
    """
    Vérifie que les bibliothèques optionnelles sont disponibles.
    """
    if (
        model_type == "lightgbm"
        and LGBMClassifier is None
    ):
        raise ImportError(
            "LightGBM n'est pas installé.\n"
            "Commande : python -m pip install lightgbm"
        )

    if (
        model_type == "xgboost"
        and XGBClassifier is None
    ):
        raise ImportError(
            "XGBoost n'est pas installé.\n"
            "Commande : python -m pip install xgboost"
        )


def build_artifact_name(
    model_type: str,
    scenario_name: str,
    config_name: str,
) -> str:
    """
    Construit un nom stable pour le modèle et ses métadonnées.
    """
    return (
        f"{MODEL_NAME}_"
        f"{model_type}_"
        f"{scenario_name}_"
        f"{config_name}"
    )


# =============================================================================
# 5. CHARGEMENT DES DONNÉES
# =============================================================================

def load_text_only_dataset(
    data_path: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Charge la colonne textuelle et la cible.

    X reste un DataFrame à une colonne pour assurer une structure stable
    entre train.py et evaluate.py.
    """
    data_path = Path(
        data_path
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : "
            f"{data_path.resolve()}"
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
            "Colonnes absentes pour le scénario texte seul : "
            f"{sorted(missing_columns)}"
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


def load_scenario_dataset(
    data_path: Path,
    scenario_name: str,
):
    """
    Charge les données du scénario demandé.
    """
    if scenario_name not in SCENARIOS:
        raise ValueError(
            f"Scénario inconnu : {scenario_name}"
        )

    scenario = SCENARIOS[
        scenario_name
    ]

    if scenario.get(
        "text_only",
        False,
    ):
        return load_text_only_dataset(
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
# 6. CONSTRUCTION DU MODÈLE
# =============================================================================

def build_classifier(
    model_type: str,
    config_name: str,
    parameter_overrides: dict[str, Any] | None = None,
):
    """
    Construit le classifieur correspondant à la demande.
    """
    check_library(
        model_type
    )

    parameters = MODEL_CONFIGS[
        model_type
    ][
        config_name
    ].copy()

    if parameter_overrides:
        parameters.update(
            parameter_overrides
        )

    classifiers = {
        "random_forest": (
            RandomForestClassifier
        ),
        "logistic_regression": (
            LogisticRegression
        ),
        "lightgbm": (
            LGBMClassifier
        ),
        "xgboost": (
            XGBClassifier
        ),
    }

    classifier_class = classifiers[
        model_type
    ]

    if classifier_class is None:
        raise ImportError(
            f"La classe du modèle '{model_type}' "
            "n'est pas disponible."
        )

    return classifier_class(
        **parameters
    )


def build_training_pipeline(
    model_type: str,
    scenario_name: str,
    config_name: str,
    text_min_df: int = 2,
    classifier_parameters: dict[str, Any] | None = None,
) -> Pipeline:
    """
    Construit le pipeline preprocessing et classifieur.

    Parameters
    ----------
    text_min_df:
        Fréquence documentaire minimale utilisée par TF-IDF.
    """
    scenario = SCENARIOS[
        scenario_name
    ]

    if scenario.get(
        "text_only",
        False,
    ):
        preprocessor = build_text_pipeline(
            max_features=1_000,
            ngram_range=(2, 3),
            min_df=text_min_df,
        )

    else:
        preprocessor = build_preprocessor(
            include_sensitive=scenario.get(
                "include_sensitive",
                True,
            ),
            include_text=scenario.get(
                "include_text",
                True,
            ),
            tfidf_max_features=1_000,
            tfidf_ngram_range=(2, 3),
            tfidf_min_df=text_min_df,
            features=scenario.get(
                "features"
            ),
        )

    classifier = build_classifier(
        model_type=model_type,
        config_name=config_name,
        parameter_overrides=classifier_parameters,
    )

    return Pipeline(
        steps=[
            (
                "preprocess",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


# =============================================================================
# 7. PONDÉRATION
# =============================================================================

def get_weighting_metadata(
    model_type: str,
    config_name: str,
) -> dict[str, Any]:
    """
    Décrit la stratégie de pondération utilisée.
    """
    if config_name == "default":
        return {
            "method": "none",
            "configuration": config_name,
        }

    if model_type in {
        "random_forest",
        "logistic_regression",
    }:
        return {
            "method": "class_weight",
            "configuration": config_name,
            "value": MODEL_CONFIGS[
                model_type
            ][
                config_name
            ].get(
                "class_weight"
            ),
        }

    return {
        "method": "sample_weight",
        "configuration": config_name,
        "value": (
            "balanced"
            if config_name in {
                "balanced",
                "regularized",
            }
            else {
                0: 1.0,
                1: 1.0,
                2: 2.5,
            }
        ),
    }


def build_fit_parameters(
    model_type: str,
    config_name: str,
    y_train: pd.Series,
) -> dict[str, Any]:
    """
    Construit sample_weight pour LightGBM et XGBoost.

    Random Forest et Logistic Regression utilisent class_weight
    directement dans leur constructeur.
    """
    if model_type not in {
        "lightgbm",
        "xgboost",
    }:
        return {}

    if config_name in {
        "balanced",
        "regularized",
    }:
        sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

    elif "class_2" in config_name.lower():
        model_parameters = MODEL_CONFIGS.get(
            model_type,
            {},
        ).get(
            config_name,
            {},
        )
        class_weights = model_parameters.get(
            "class_weight"
        )

        if not isinstance(class_weights, dict):
            class_weights = {
                0: 1.0,
                1: 1.0,
                2: 2.5,
            }

        sample_weight = (
            y_train
            .map(
                class_weights
            )
            .to_numpy(
                dtype=float
            )
        )

    else:
        return {}

    return {
        "classifier__sample_weight": (
            sample_weight
        )
    }


def build_stratification_target(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.Series:
    """
    Retourne le libellé de stratification à utiliser.

    Le projet stratifie uniquement sur la cible pour rester compatible
    avec les contraintes de scikit-learn et éviter les groupes trop petits
    sur des sous-classes spécifiques.
    """
    return y.copy()


def split_train_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Sépare les données en train/holdout avec une stratification sur la cible.

    Parameters
    ----------
    X, y:
        Jeu de données et cible.
    test_size:
        Taille du holdout (par défaut 20 %).
    random_state:
        Graine utilisée pour la reproductibilité.
    stratify:
        Optional. Labels de stratification. Si None, la cible est utilisée.
    """
    if stratify is None:
        stratify = build_stratification_target(
            X=X,
            y=y,
        )

    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )


# =============================================================================
# 8. VALIDATION CROISÉE
# =============================================================================

def summarize_cv_scores(
    fold_scores: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """
    Agrège les scores par fold.
    """
    if not fold_scores:
        return {}

    metric_names = sorted(
        fold_scores[0].keys()
    )

    summary = {}

    for metric_name in metric_names:
        values = np.asarray(
            [
                score[metric_name]
                for score in fold_scores
            ],
            dtype=float,
        )

        summary[
            metric_name
        ] = {
            "mean": float(
                values.mean()
            ),
            "std": float(
                values.std(
                    ddof=0
                )
            ),
            "min": float(
                values.min()
            ),
            "max": float(
                values.max()
            ),
        }

    return summary


def fit_pipeline_with_text_fallback(
    model_type: str,
    scenario_name: str,
    config_name: str,
    X_train,
    y_train: pd.Series,
    classifier_parameters: dict[str, Any] | None = None,
) -> tuple[Pipeline, int]:
    """
    Entraîne un pipeline avec min_df=2.

    Si TF-IDF ne conserve aucun terme dans un petit fold,
    un second essai est réalisé avec min_df=1.
    """
    scenario = SCENARIOS[
        scenario_name
    ]

    uses_text = bool(
        scenario.get(
            "include_text",
            False,
        )
        or scenario.get(
            "text_only",
            False,
        )
    )

    min_df_values = (
        (2, 1)
        if uses_text
        else (2,)
    )

    last_error = None

    for min_df in min_df_values:
        pipeline = build_training_pipeline(
            model_type=model_type,
            scenario_name=scenario_name,
            config_name=config_name,
            text_min_df=min_df,
            classifier_parameters=classifier_parameters,
        )

        fit_parameters = build_fit_parameters(
            model_type=model_type,
            config_name=config_name,
            y_train=y_train,
        )

        try:
            pipeline.fit(
                X_train,
                y_train,
                **fit_parameters,
            )

            return pipeline, min_df

        except ValueError as error:
            last_error = error

            empty_vocabulary_messages = (
                "After pruning, no terms remain",
                "empty vocabulary",
            )

            is_empty_vocabulary = any(
                message in str(error)
                for message
                in empty_vocabulary_messages
            )

            if (
                min_df == 2
                and is_empty_vocabulary
            ):
                continue

            raise

    raise RuntimeError(
        "Impossible d'entraîner le pipeline texte."
    ) from last_error


def compute_stratified_cv_summary(
    model_type: str,
    scenario_name: str,
    config_name: str,
    X,
    y: pd.Series,
    n_splits: int = 5,
) -> dict[str, Any]:
    """
    Réalise une validation croisée stratifiée sur le train uniquement.
    La stratification reste fondée sur la cible pour éviter les sous-
    classes trop petites et conserver une répartition stable des labels.
    """
    if n_splits < 2:
        return {
            "enabled": False,
            "n_splits": 0,
            "metrics": {},
        }

    stratification_target = build_stratification_target(
        X=X,
        y=y,
    )

    class_counts = (
        stratification_target
        .value_counts()
    )

    min_class_count = int(
        class_counts.min()
    )

    effective_splits = min(
        n_splits,
        min_class_count,
    )

    if effective_splits < 2:
        raise ValueError(
            "La validation croisée exige au moins deux observations "
            "dans chaque strate retenue."
        )

    cv = StratifiedKFold(
        n_splits=effective_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_scores: list[
        dict[str, float]
    ] = []

    min_df_used_by_fold: list[int] = []

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        cv.split(
            X,
            stratification_target,
        ),
        start=1,
    ):
        X_fold_train = X.iloc[
            train_indices
        ]

        y_fold_train = y.iloc[
            train_indices
        ]

        X_fold_validation = X.iloc[
            validation_indices
        ]

        y_fold_validation = y.iloc[
            validation_indices
        ]

        pipeline, min_df_used = (
            fit_pipeline_with_text_fallback(
                model_type=model_type,
                scenario_name=scenario_name,
                config_name=config_name,
                X_train=X_fold_train,
                y_train=y_fold_train,
            )
        )

        min_df_used_by_fold.append(
            min_df_used
        )

        y_pred = pipeline.predict(
            X_fold_validation
        )

        f1_classes = f1_score(
            y_fold_validation,
            y_pred,
            labels=CLASS_LABELS,
            average=None,
            zero_division=0,
        )

        recall_classes = recall_score(
            y_fold_validation,
            y_pred,
            labels=CLASS_LABELS,
            average=None,
            zero_division=0,
        )

        # Erreur critique : une classe 2 (risque de chômage longue durée)
        # prédite à tort en classe 0 (retour rapide) prive l'usager d'un
        # accompagnement renforcé — c'est l'erreur la plus coûteuse métier.
        true_classe_2_mask = (
            y_fold_validation == 2
        ).to_numpy()
        n_true_classe_2 = int(
            true_classe_2_mask.sum()
        )
        erreur_critique_2_vers_0 = (
            float(
                (
                    y_pred[true_classe_2_mask]
                    == 0
                ).sum()
                / n_true_classe_2
            )
            if n_true_classe_2 > 0
            else 0.0
        )

        fold_scores.append(
            {
                "accuracy": float(
                    accuracy_score(
                        y_fold_validation,
                        y_pred,
                    )
                ),

                "f1_macro": float(
                    f1_score(
                        y_fold_validation,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    )
                ),

                "f1_weighted": float(
                    f1_score(
                        y_fold_validation,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    )
                ),

                "f1_class_2": float(
                    f1_classes[2]
                ),

                "recall_classe_2": float(
                    recall_classes[2]
                ),

                "erreur_critique_2_vers_0": (
                    erreur_critique_2_vers_0
                ),
            }
        )

        print(
            f"  Fold {fold_number}/{effective_splits} "
            f"| F1 macro : "
            f"{fold_scores[-1]['f1_macro']:.4f} "
            f"| F1 classe 2 : "
            f"{fold_scores[-1]['f1_class_2']:.4f} "
            f"| Recall classe 2 : "
            f"{fold_scores[-1]['recall_classe_2']:.4f} "
            f"| Erreur critique 2\u21920 : "
            f"{fold_scores[-1]['erreur_critique_2_vers_0']:.2%}"
        )

    return {
        "enabled": True,
        "method": "StratifiedKFold",
        "n_splits": effective_splits,
        "shuffle": True,
        "random_state": RANDOM_STATE,
        "metrics": summarize_cv_scores(
            fold_scores
        ),
        "tfidf_min_df_used_by_fold": (
            min_df_used_by_fold
        ),
    }


def compare_baseline_hyperparameters(
    model_type: str,
    scenario_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config_name: str = "balanced",
    parameter_sets: dict[str, dict[str, Any]] | None = None,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """
    Compare 2 à 3 jeux d'hyperparamètres avec une CV stratifiée manuelle.

    La fonction est destinée à la baseline documentée du notebook : elle
    retourne le score F1 macro moyen, son écart-type, le rappel de la classe
    2 et une ligne d'analyse par configuration. Le holdout n'est jamais
    utilisé.

    ``parameter_sets`` contient des surcharges du modèle de base, par exemple:

        {
            "baseline_50": {"n_estimators": 50, "max_depth": None},
            "baseline_200": {"n_estimators": 200, "max_depth": 10},
        }
    """
    validate_training_request(
        model_type=model_type,
        scenario_name=scenario_name,
        config_name=config_name,
    )

    if parameter_sets is None:
        default_parameter_sets_by_model = {
            "random_forest": {
                "balanced": {
                    "n_estimators": 400,
                    "max_depth": None,
                    "min_samples_leaf": 2,
                    "max_features": "sqrt",
                    "class_weight": "balanced",
                },
                "balanced_robust": {
                    "n_estimators": 200,
                    "max_depth": None,
                    "min_samples_leaf": 2,
                    "max_features": "sqrt",
                    "class_weight": "balanced",
                },
                "critical_class_2": {
                    "n_estimators": 400,
                    "max_depth": 15,
                    "min_samples_leaf": 5,
                    "max_features": "log2",
                    "class_weight": {
                        0: 1.0,
                        1: 1.0,
                        2: 2.5,
                    },
                },
            },
            "logistic_regression": {
                "baseline_1": {"C": 0.1},
                "baseline_2": {"C": 1.0},
                "baseline_3": {"C": 10.0},
            },
            "lightgbm": {
                "baseline_1": {"num_leaves": 15},
                "baseline_2": {"num_leaves": 31},
                "baseline_3": {"num_leaves": 63},
            },
            "xgboost": {
                "baseline_1": {"max_depth": 3},
                "baseline_2": {"max_depth": 6},
                "baseline_3": {"max_depth": 9},
            },
        }
        default_parameter_sets = (
            default_parameter_sets_by_model[model_type]
        )
        parameter_sets = default_parameter_sets

    if not 2 <= len(parameter_sets) <= 3:
        raise ValueError(
            "La baseline doit comparer 2 ou 3 jeux d'hyperparamètres."
        )

    class_counts = y_train.value_counts()
    effective_folds = min(
        cv_folds,
        int(class_counts.min()),
    )

    if effective_folds < 2:
        raise ValueError(
            "La validation croisée exige au moins deux observations "
            "dans chaque classe."
        )

    cv = StratifiedKFold(
        n_splits=effective_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows: list[dict[str, Any]] = []

    for set_name, parameters in parameter_sets.items():
        fold_f1: list[float] = []
        fold_recall_class_2: list[float] = []

        for train_indices, validation_indices in cv.split(X_train, y_train):
            pipeline, _ = fit_pipeline_with_text_fallback(
                model_type=model_type,
                scenario_name=scenario_name,
                config_name=config_name,
                X_train=X_train.iloc[train_indices],
                y_train=y_train.iloc[train_indices],
                classifier_parameters=parameters,
            )

            y_validation = y_train.iloc[validation_indices]
            y_pred = pipeline.predict(
                X_train.iloc[validation_indices]
            )

            fold_f1.append(
                float(
                    f1_score(
                        y_validation,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    )
                )
            )

            recall_by_class = recall_score(
                y_validation,
                y_pred,
                labels=CLASS_LABELS,
                average=None,
                zero_division=0,
            )
            fold_recall_class_2.append(
                float(recall_by_class[2])
            )

        rows.append(
            {
                "scenario_name": scenario_name,
                "model_type": model_type,
                "config_name": config_name,
                "hyperparameter_set": set_name,
                "hyperparameters": repr(parameters),
                "f1_macro_mean": float(np.mean(fold_f1)),
                "f1_macro_std": float(np.std(fold_f1, ddof=0)),
                "recall_classe_2_mean": float(
                    np.mean(fold_recall_class_2)
                ),
                "recall_classe_2_std": float(
                    np.std(fold_recall_class_2, ddof=0)
                ),
            }
        )

    results = pd.DataFrame(rows).sort_values(
        ["f1_macro_mean", "recall_classe_2_mean"],
        ascending=[False, False],
        ignore_index=True,
    )
    best_set = results.iloc[0]["hyperparameter_set"]
    results["selected"] = results["hyperparameter_set"] == best_set
    results["analysis"] = results.apply(
        lambda row: (
            "Retenu : meilleur F1 macro CV."
            if row["selected"]
            else (
                "À comparer : F1 macro inférieur de "
                f"{results.iloc[0]['f1_macro_mean'] - row['f1_macro_mean']:.3f}."
            )
        ),
        axis=1,
    )

    return results


def _recall_class_2(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> float:
    """Calcule le rappel de la classe 2 pour un scorer sklearn."""
    values = recall_score(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        average=None,
        zero_division=0,
    )
    return float(values[2])




def _f1_class_2(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> float:
    """Calcule le F1 de la classe 2 pour un scorer sklearn."""
    values = f1_score(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        average=None,
        zero_division=0,
    )
    return float(values[2])


def _critical_error_2_to_0(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> float:
    """Mesure la proportion de classes 2 prédites à tort comme classe 0."""
    true_class_2 = np.asarray(y_true) == 2
    count_class_2 = int(true_class_2.sum())

    if count_class_2 == 0:
        return 0.0

    return float(
        (np.asarray(y_pred)[true_class_2] == 0).sum()
        / count_class_2
    )


def _critical_error_0_to_2(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> float:
    """Mesure la proportion de classes 0 prédites à tort comme classe 2."""
    true_class_0 = np.asarray(y_true) == 0
    count_class_0 = int(true_class_0.sum())

    if count_class_0 == 0:
        return 0.0

    return float(
        (np.asarray(y_pred)[true_class_0] == 2).sum()
        / count_class_0
    )


_WEIGHT_CRITICAL_ERROR_2_TO_0 = 3.0
_WEIGHT_FALSE_ALERT_0_TO_2 = 1.0


def _select_best_grid_search_candidate(
    cv_results: dict[str, Any],
) -> int:
    """Minimise le coût métier pondéré, puis départage par F1 macro."""
    error_2_to_0 = cv_results[
        "mean_test_error_2_to_0"
    ]
    error_0_to_2 = cv_results[
        "mean_test_error_0_to_2"
    ]
    f1_macro_scores = cv_results["mean_test_f1_macro"]

    def combined_cost(index: int) -> float:
        cost = (
            _WEIGHT_CRITICAL_ERROR_2_TO_0 * (-error_2_to_0[index])
            + _WEIGHT_FALSE_ALERT_0_TO_2 * (-error_0_to_2[index])
        )
        return -cost

    return max(
        range(len(f1_macro_scores)),
        key=lambda index: (
            combined_cost(index),
            f1_macro_scores[index],
        ),
    )


def _summarize_grid_search_results(
    grid_search: GridSearchCV,
) -> pd.DataFrame:
    """Trie par macro F1, class-2 F1, puis erreur critique croissante."""
    results_df = pd.DataFrame(grid_search.cv_results_)
    summary_cols = [
        "params",
        "mean_test_f1_macro",
        "mean_test_f1_class_2",
        "mean_test_recall_class_2",
        "mean_test_error_2_to_0",
        "mean_test_error_0_to_2",
        "rank_test_f1_macro",
        "rank_test_f1_class_2",
    ]
    summary = results_df[summary_cols].copy()
    summary["mean_test_error_2_to_0"] *= -1
    summary["mean_test_error_0_to_2"] *= -1
    summary["mean_test_combined_error_cost"] = (
        _WEIGHT_CRITICAL_ERROR_2_TO_0 * summary["mean_test_error_2_to_0"]
        + _WEIGHT_FALSE_ALERT_0_TO_2 * summary["mean_test_error_0_to_2"]
    )
    return summary.sort_values(
        [
            "mean_test_combined_error_cost",
            "mean_test_f1_macro",
        ],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)


def grid_search_random_forest_class_2(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameter_grid: list[dict[str, list[Any]]] | None = None,
    scenario_name: str | None = None,
    cv_folds: int = 5,
    n_jobs: int = -1,
) -> tuple[GridSearchCV, pd.DataFrame]:
    """Recherche les hyperparamètres d'un Random Forest pour la classe 2.

    La validation croisée est réalisée uniquement sur ``X_train`` et
    ``y_train``. Lorsque ``scenario_name`` est fourni, le pipeline de
    prétraitement du projet est utilisé avant le Random Forest.
    """
    if parameter_grid is None:
        parameter_grid = [
            {
                "n_estimators": [300, 500],
                "max_depth": [None, 18],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt"],
                "class_weight": [
                    None,
                    "balanced",
                    {0: 1.0, 1: 1.0, 2: 2.5},
                    {0: 1.0, 1: 1.0, 2: 4.0},
                ],
                "random_state": [RANDOM_STATE],
                "n_jobs": [1],
            },
        ]

    if cv_folds < 2:
        raise ValueError("cv_folds doit être supérieur ou égal à 2.")

    class_counts = pd.Series(y_train).value_counts()
    effective_folds = min(cv_folds, int(class_counts.min()))
    if effective_folds < 2:
        raise ValueError(
            "La validation croisée exige au moins deux observations "
            "dans chaque classe."
        )

    scoring = {
        "f1_macro": make_scorer(
            f1_score,
            average="macro",
            zero_division=0,
        ),
        "recall_class_2": make_scorer(_recall_class_2),
        "f1_class_2": make_scorer(_f1_class_2),
        "error_2_to_0": make_scorer(
            _critical_error_2_to_0,
            greater_is_better=False,
        ),
        "error_0_to_2": make_scorer(
            _critical_error_0_to_2,
            greater_is_better=False,
        ),
    }
    cv = StratifiedKFold(
        n_splits=effective_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    if scenario_name is None:
        estimator: Any = RandomForestClassifier()
    else:
        if scenario_name not in SCENARIOS:
            raise ValueError(
                f"Scénario inconnu : '{scenario_name}'. "
                f"Scénarios disponibles : {list(SCENARIOS)}"
            )
        estimator = build_training_pipeline(
            model_type="random_forest",
            scenario_name=scenario_name,
            config_name="default",
        )
        parameter_grid = [
            {
                f"classifier__{name}": values
                for name, values in parameters.items()
            }
            for parameters in parameter_grid
        ]

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=parameter_grid,
        scoring=scoring,
        refit=_select_best_grid_search_candidate,
        cv=cv,
        n_jobs=n_jobs,
        error_score="raise",
    )
    grid_search.fit(X_train, y_train)

    return grid_search, _summarize_grid_search_results(grid_search)


def _grid_search_pipeline_class_2(
    model_type: str,
    scenario_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameter_grid: list[dict[str, list[Any]]],
    cv_folds: int,
    n_jobs: int,
) -> tuple[GridSearchCV, pd.DataFrame]:
    """Exécute un GridSearchCV avec les métriques dédiées à la classe 2."""
    if scenario_name not in SCENARIOS:
        raise ValueError(
            f"Scénario inconnu : '{scenario_name}'. "
            f"Scénarios disponibles : {list(SCENARIOS)}"
        )

    class_counts = pd.Series(y_train).value_counts()
    effective_folds = min(cv_folds, int(class_counts.min()))
    if effective_folds < 2:
        raise ValueError(
            "La validation croisée exige au moins deux observations "
            "dans chaque classe."
        )

    estimator = build_training_pipeline(
        model_type=model_type,
        scenario_name=scenario_name,
        config_name="default",
    )
    prefixed_grid = [
        {
            f"classifier__{name}": values
            for name, values in parameters.items()
        }
        for parameters in parameter_grid
    ]
    scoring = {
        "f1_macro": make_scorer(
            f1_score,
            average="macro",
            zero_division=0,
        ),
        "recall_class_2": make_scorer(_recall_class_2),
        "f1_class_2": make_scorer(_f1_class_2),
        "error_2_to_0": make_scorer(
            _critical_error_2_to_0,
            greater_is_better=False,
        ),
        "error_0_to_2": make_scorer(
            _critical_error_0_to_2,
            greater_is_better=False,
        ),
    }
    cv = StratifiedKFold(
        n_splits=effective_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=prefixed_grid,
        scoring=scoring,
        refit=_select_best_grid_search_candidate,
        cv=cv,
        n_jobs=n_jobs,
        error_score="raise",
    )
    grid_search.fit(X_train, y_train)

    return grid_search, _summarize_grid_search_results(grid_search)


def grid_search_logistic_regression_class_2(
    scenario_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameter_grid: list[dict[str, list[Any]]] | None = None,
    cv_folds: int = 5,
    n_jobs: int = -1,
) -> tuple[GridSearchCV, pd.DataFrame]:
    """Optimise Logistic Regression pour la détection de la classe 2."""
    if parameter_grid is None:
        parameter_grid = [
            {
                "solver": ["lbfgs"],
                "C": [0.1, 1.0],
                "class_weight": [
                    None,
                    "balanced",
                    {0: 1.0, 1: 1.0, 2: 2.5},
                    {0: 1.0, 1: 1.0, 2: 4.0},
                ],
                "max_iter": [2_000],
            },
        ]

    return _grid_search_pipeline_class_2(
        model_type="logistic_regression",
        scenario_name=scenario_name,
        X_train=X_train,
        y_train=y_train,
        parameter_grid=parameter_grid,
        cv_folds=cv_folds,
        n_jobs=n_jobs,
    )


def grid_search_lightgbm_class_2(
    scenario_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameter_grid: list[dict[str, list[Any]]] | None = None,
    cv_folds: int = 5,
    n_jobs: int = -1,
) -> tuple[GridSearchCV, pd.DataFrame]:
    """Optimise LightGBM pour la détection de la classe 2."""
    if parameter_grid is None:
        parameter_grid = [
            {
                "n_estimators": [150, 350],
                "learning_rate": [0.03, 0.08],
                "num_leaves": [15, 31],
                "min_child_samples": [10, 30],
                "colsample_bytree": [0.9],
                "class_weight": [
                    None,
                    "balanced",
                    {0: 1.0, 1: 1.0, 2: 2.5},
                    {0: 1.0, 1: 1.0, 2: 4.0},
                ],
            },
        ]

    return _grid_search_pipeline_class_2(
        model_type="lightgbm",
        scenario_name=scenario_name,
        X_train=X_train,
        y_train=y_train,
        parameter_grid=parameter_grid,
        cv_folds=cv_folds,
        n_jobs=n_jobs,
    )

def grid_search_xgboost_class_2(
    scenario_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameter_grid: list[dict[str, list[Any]]] | None = None,
    cv_folds: int = 5,
    n_jobs: int = -1,
) -> tuple[GridSearchCV, pd.DataFrame]:
    """Optimise XGBoost pour la détection de la classe 2.

    Le pipeline de prétraitement du scénario est inclus dans la recherche.
    La validation croisée utilise uniquement ``X_train`` et ``y_train``.
    """
    if XGBClassifier is None:
        raise ImportError("XGBoost n'est pas installé.")

    if scenario_name not in SCENARIOS:
        raise ValueError(
            f"Scénario inconnu : '{scenario_name}'. "
            f"Scénarios disponibles : {list(SCENARIOS)}"
        )

    if parameter_grid is None:
        parameter_grid = [
            {
                "n_estimators": [200, 400],
                "learning_rate": [0.03, 0.1],
                "max_depth": [3, 6],
                "min_child_weight": [1, 3],
                "subsample": [0.9],
                "colsample_bytree": [0.9],
            },
        ]

    class_counts = pd.Series(y_train).value_counts()
    effective_folds = min(cv_folds, int(class_counts.min()))
    if effective_folds < 2:
        raise ValueError(
            "La validation croisée exige au moins deux observations "
            "dans chaque classe."
        )

    estimator = build_training_pipeline(
        model_type="xgboost",
        scenario_name=scenario_name,
        config_name="default",
    )
    estimator.set_params(
        classifier__n_jobs=1
    )
    sample_weights = np.where(
        np.asarray(y_train) == 2,
        2.5,
        1.0,
    )
    prefixed_grid = [
        {
            f"classifier__{name}": values
            for name, values in parameters.items()
        }
        for parameters in parameter_grid
    ]
    scoring = {
        "f1_macro": make_scorer(
            f1_score,
            average="macro",
            zero_division=0,
        ),
        "recall_class_2": make_scorer(_recall_class_2),
        "f1_class_2": make_scorer(_f1_class_2),
        "error_2_to_0": make_scorer(
            _critical_error_2_to_0,
            greater_is_better=False,
        ),
        "error_0_to_2": make_scorer(
            _critical_error_0_to_2,
            greater_is_better=False,
        ),
    }
    cv = StratifiedKFold(
        n_splits=effective_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=prefixed_grid,
        scoring=scoring,
        refit=_select_best_grid_search_candidate,
        cv=cv,
        n_jobs=n_jobs,
        error_score="raise",
    )
    grid_search.fit(
        X_train,
        y_train,
        classifier__sample_weight=sample_weights,
    )

    return grid_search, _summarize_grid_search_results(grid_search)






# =============================================================================
# 9. DESCRIPTION DES FEATURES
# =============================================================================

def build_feature_metadata(
    scenario_name: str,
) -> dict[str, Any]:
    """
    Décrit les variables utilisées par le scénario.
    """
    scenario = SCENARIOS[
        scenario_name
    ]

    if scenario.get(
        "text_only",
        False,
    ):
        return {
            "numeric": [],
            "ordinal": {},
            "categorical": [],
            "text": [
                TEXT_FEATURE
            ],
            "all_model_inputs": [
                TEXT_FEATURE
            ],
        }

    return {
        "numeric": list(
            NUMERIC_FEATURES
        ),

        "ordinal": {
            column: categories
            for column, categories
            in ORDINAL_FEATURES.items()
        },

        "categorical": (
            get_categorical_features(
                include_sensitive=scenario.get(
                    "include_sensitive",
                    True,
                )
            )
        ),

        "text": (
            [TEXT_FEATURE]
            if scenario.get(
                "include_text",
                True,
            )
            else []
        ),

        "all_model_inputs": (
            get_all_features(
                include_sensitive=scenario.get(
                    "include_sensitive",
                    True,
                ),
                include_text=scenario.get(
                    "include_text",
                    True,
                ),
            )
        ),
    }


# =============================================================================
# 10. MÉTADONNÉES
# =============================================================================

def measure_training_inference_cpu(
    model,
    X_train: pd.DataFrame,
    max_samples: int = 100,
) -> dict[str, float | int]:
    """Measure per-row prediction CPU time using training rows only."""
    if X_train.empty:
        raise ValueError("Impossible de mesurer les prédictions sur un train vide.")

    number_of_samples = min(max_samples, len(X_train))
    cpu_times_ms: list[float] = []

    model.predict(X_train.iloc[[0]])

    for index in range(number_of_samples):
        start_time = process_time()
        model.predict(X_train.iloc[[index]])
        cpu_times_ms.append((process_time() - start_time) * 1_000)

    return {
        "cpu_ms_1000_predictions": float(np.mean(cpu_times_ms) * 1_000),
        "inference_samples": number_of_samples,
    }

def build_metadata(
    model_type: str,
    scenario_name: str,
    config_name: str,
    data_path: Path,
    model_path: Path,
    metadata_path: Path,
    X_train,
    X_holdout,
    y_train: pd.Series,
    y_holdout: pd.Series,
    training_seconds: float,
    training_inference_metrics: dict[str, float | int],
    cv_summary: dict[str, Any],
    final_tfidf_min_df: int,
) -> dict[str, Any]:
    """
    Construit les métadonnées du modèle.
    """
    metadata = {
        "model_name": MODEL_NAME,
        "model_type": model_type,
        "model_version": MODEL_VERSION,
        "scenario_name": scenario_name,
        "config_name": config_name,

        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "versions": {
            "python": (
                platform.python_version()
            ),

            "scikit_learn": (
                sklearn.__version__
            ),

            "lightgbm": (
                lightgbm.__version__
                if lightgbm is not None
                else None
            ),

            "xgboost": (
                xgboost.__version__
                if xgboost is not None
                else None
            ),
        },

        "dataset": {
            "path": str(
                data_path
            ),

            "sha256": (
                compute_sha256(
                    data_path
                )
            ),

            "total_rows": int(
                len(X_train)
                + len(X_holdout)
            ),

            "train_rows": int(
                len(X_train)
            ),

            "test_rows": int(
                len(X_holdout)
            ),

            "test_size": TEST_SIZE,

            "random_state": RANDOM_STATE,

            "train_indices": [
                int(index)
                for index in X_train.index
            ],

            "test_indices": [
                int(index)
                for index in X_holdout.index
            ],

            "class_distribution_train": (
                y_train
                .value_counts(
                    normalize=True
                )
                .sort_index()
                .round(6)
                .to_dict()
            ),

            "class_distribution_test": (
                y_holdout
                .value_counts(
                    normalize=True
                )
                .sort_index()
                .round(6)
                .to_dict()
            ),
        },

        "scenario": dict(
            SCENARIOS[
                scenario_name
            ]
        ),

        "configuration": {
            "model_parameters": (
                MODEL_CONFIGS[
                    model_type
                ][
                    config_name
                ]
            ),

            "weighting": (
                get_weighting_metadata(
                    model_type=model_type,
                    config_name=config_name,
                )
            ),

            "tfidf_min_df_final": (
                final_tfidf_min_df
            ),
        },

        "feature_columns": (
            build_feature_metadata(
                scenario_name
            )
        ),

        "target": {
            "column": TARGET_COLUMN,

            "expected_values": sorted(
                EXPECTED_TARGET_VALUES
            ),

            "meaning": {
                "0": (
                    "Retour rapide, "
                    "moins de 6 mois"
                ),

                "1": (
                    "Retour moyen, "
                    "entre 6 et 12 mois"
                ),

                "2": (
                    "Risque de chômage longue durée, "
                    "plus de 12 mois"
                ),
            },
        },

        "training": {
            "training_seconds": float(
                training_seconds
            ),

            "cpu_ms_1000_predictions": float(
                training_inference_metrics[
                    "cpu_ms_1000_predictions"
                ]
            ),

            "inference_measurement_dataset": "train_split_only",

            "inference_samples": int(
                training_inference_metrics[
                    "inference_samples"
                ]
            ),

            "trained_on": (
                "train_split_only"
            ),

            "cross_validation": (
                cv_summary
            ),

            "evaluation_status": (
                "not_evaluated"
            ),

            "evaluation_script": (
                "src/evaluate.py"
            ),

            "holdout_used_during_training": (
                False
            ),
        },

        "artifacts": {
            "model_path": str(
                model_path
            ),

            "metadata_path": str(
                metadata_path
            ),
        },
    }

    return make_json_serializable(
        metadata
    )


# =============================================================================
# 11. ENTRAÎNEMENT D'UNE COMBINAISON
# =============================================================================

def train_scenario(
    model_type: str,
    scenario_name: str,
    config_name: str,
    data_path: Path,
    output_dir: Path,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """
    Entraîne un modèle pour un scénario.

    Le holdout est réservé et n'est jamais utilisé pour la validation
    croisée ni pour l'entraînement.
    """
    validate_training_request(
        model_type=model_type,
        scenario_name=scenario_name,
        config_name=config_name,
    )

    data_path = Path(
        data_path
    )

    output_dir = Path(
        output_dir
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : "
            f"{data_path.resolve()}"
        )

    #print("\n" + "=" * 78)
    #print("ENTRAÎNEMENT CISIA")
    print("=" * 78)

    print(
        f"Algorithme    : {model_type}"
    )

    print(
        f"Scénario      : {scenario_name}"
    )

    print(
        f"Configuration : {config_name}"
    )

    print(
        f"Description   : "
        f"{SCENARIOS[scenario_name].get('description', '')}"
    )

    print("=" * 78)

    X, y = load_scenario_dataset(
        data_path=data_path,
        scenario_name=scenario_name,
    )

    observed_classes = set(
        y.unique().tolist()
    )

    if observed_classes != EXPECTED_TARGET_VALUES:
        raise ValueError(
            "La cible ne contient pas exactement les classes "
            f"{sorted(EXPECTED_TARGET_VALUES)}. "
            f"Classes observées : {sorted(observed_classes)}"
        )

    stratification_target = build_stratification_target(
        X=X,
        y=y,
    )

    (
        X_train,
        X_holdout,
        y_train,
        y_holdout,
    ) = split_train_holdout(
        X=X,
        y=y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratification_target,
    )

    if not X_train.index.equals(
        y_train.index
    ):
        raise ValueError(
            "Les index de X_train et y_train "
            "ne sont pas alignés."
        )

    if not X_holdout.index.equals(
        y_holdout.index
    ):
        raise ValueError(
            "Les index de X_holdout et y_holdout "
            "ne sont pas alignés."
        )

    #print(
    #    f"\nDimensions totales : {X.shape}"
    #)

    print(
        f"Dimensions train   : {X_train.shape}"
    )

    #print(
    #    f"Dimensions holdout : {X_holdout.shape}"
    #)

    #print(
    #    "Le holdout est réservé à evaluate.py."
    #)

    cv_summary = {
        "enabled": False,
        "n_splits": 0,
        "metrics": {},
    }

    if cv_folds > 1:
        effective_folds = min(
            cv_folds,
            int(
                y_train
                .value_counts()
                .min()
            ),
        )

        print(
            "\nValidation croisée stratifiée "
        #    f"sur le train uniquement "
            f"({effective_folds} folds)..."
        )

        cv_summary = (
            compute_stratified_cv_summary(
                model_type=model_type,
                scenario_name=scenario_name,
                config_name=config_name,
                X=X_train,
                y=y_train,
                n_splits=cv_folds,
            )
        )

        f1_macro_summary = (
            cv_summary
            .get(
                "metrics",
                {},
            )
            .get(
                "f1_macro",
                {},
            )
        )

        recall_classe_2_summary = (
            cv_summary
            .get(
                "metrics",
                {},
            )
            .get(
                "recall_classe_2",
                {},
            )
        )

        erreur_critique_summary = (
            cv_summary
            .get(
                "metrics",
                {},
            )
            .get(
                "erreur_critique_2_vers_0",
                {},
            )
        )

        if f1_macro_summary:
            print(
                "\nRésumé F1 macro CV : "
                f"{f1_macro_summary['mean']:.4f} "
                f"+/- {f1_macro_summary['std']:.4f}"
            )

        if recall_classe_2_summary:
            print(
                "Résumé Recall classe 2 CV : "
                f"{recall_classe_2_summary['mean']:.4f} "
                f"+/- {recall_classe_2_summary['std']:.4f}"
            )

        if erreur_critique_summary:
            print(
                "Résumé Erreur critique 2\u21920 CV : "
                f"{erreur_critique_summary['mean']:.2%} "
                f"+/- {erreur_critique_summary['std']:.2%}"
            )

    #print(
    #    "\nEntraînement final du pipeline "
    #    "sur l'intégralité du train..."
    #)

    start = perf_counter()

    (
        pipeline,
        final_tfidf_min_df,
    ) = fit_pipeline_with_text_fallback(
        model_type=model_type,
        scenario_name=scenario_name,
        config_name=config_name,
        X_train=X_train,
        y_train=y_train,
    )

    training_seconds = (
        perf_counter()
        - start
    )

    print(
        "Entraînement terminé en "
        f"{training_seconds:.4f} seconde(s)."
    )

    training_inference_metrics = measure_training_inference_cpu(
        model=pipeline,
        X_train=X_train,
    )

    #print(
    #    "Temps CPU / 1 000 prédictions (train) : "
    #    f"{training_inference_metrics['cpu_ms_1000_predictions'] / 1_000:.3f} ms"
    #)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_name = build_artifact_name(
        model_type=model_type,
        scenario_name=scenario_name,
        config_name=config_name,
    )

    model_path = (
        output_dir
        / f"{artifact_name}.joblib"
    )

    metadata_path = (
        output_dir
        / f"{artifact_name}.json"
    )

    joblib.dump(
        pipeline,
        model_path,
        compress=3,
    )

    metadata = build_metadata(
        model_type=model_type,
        scenario_name=scenario_name,
        config_name=config_name,
        data_path=data_path,
        model_path=model_path,
        metadata_path=metadata_path,
        X_train=X_train,
        X_holdout=X_holdout,
        y_train=y_train,
        y_holdout=y_holdout,
        training_seconds=training_seconds,
        training_inference_metrics=training_inference_metrics,
        cv_summary=cv_summary,
        final_tfidf_min_df=(
            final_tfidf_min_df
        ),
    )

    model_size_bytes = int(
        model_path.stat().st_size
    )

    metadata[
        "artifacts"
    ][
        "model_size_bytes"
    ] = model_size_bytes

    metadata[
        "artifacts"
    ][
        "model_size_mb"
    ] = round(
        model_size_bytes
        / (1024 ** 2),
        6,
    )

    metadata_path.write_text(
        json.dumps(
            make_json_serializable(
                metadata
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    #print(
    #    f"\nPipeline     : "
    #    f"{model_path.resolve()}"
    #)

    #print(
    #    f"Métadonnées : "
    #    f"{metadata_path.resolve()}"
    #)


    return {
        "model_type": model_type,
        "scenario_name": scenario_name,
        "config_name": config_name,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "training_seconds": (
            training_seconds
        ),
        "cpu_ms_1000_predictions": (
            training_inference_metrics[
                "cpu_ms_1000_predictions"
            ]
        ),
        "cv_summary": cv_summary,
    }


# =============================================================================
# 12. EXÉCUTION DES COMBINAISONS
# =============================================================================

def train_requested_models(
    requested_model_type: str,
    requested_scenario: str,
    config_name: str,
    data_path: Path,
    output_dir: Path,
    cv_folds: int = 5,
) -> list[dict[str, Any]]:
    """
    Entraîne toutes les combinaisons demandées.
    """
    model_types = (
        list(
            AVAILABLE_MODEL_TYPES
        )
        if requested_model_type
        == "all"
        else [
            requested_model_type
        ]
    )

    scenario_names = (
        list(
            SCENARIOS.keys()
        )
        if requested_scenario
        == "all"
        else [
            requested_scenario
        ]
    )

    combinations = [
        (
            model_type,
            scenario_name,
        )
        for model_type
        in model_types
        for scenario_name
        in scenario_names
    ]

    results = []

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

        result = train_scenario(
            model_type=model_type,
            scenario_name=scenario_name,
            config_name=config_name,
            data_path=data_path,
            output_dir=output_dir,
            cv_folds=cv_folds,
        )

        results.append(
            result
        )

    return results


# =============================================================================
# 13. SYNTHÈSE
# =============================================================================

def print_training_summary(
    results: list[dict[str, Any]],
) -> None:
    """
    Affiche les scores moyens et leur écart-type CV sous forme de tableau.
    """
    if isinstance(results, pd.DataFrame):
        fine_tuning_results = results.copy()
    elif results and isinstance(results[0], pd.DataFrame):
        fine_tuning_results = pd.concat(
            results,
            ignore_index=True,
        )
    else:
        fine_tuning_results = None

    if fine_tuning_results is not None:
        metric_columns = [
            "scenario",
            "class_weight",
            "class_2_weight",
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "max_features",
            "learning_rate",
            "num_leaves",
            "min_child_samples",
            "colsample_bytree",
            "f1_macro",
            "f1_macro_std",
            "recall_c2",
            "critical_2_to_0",
            "selected",
            "analysis",
        ]
        available_columns = [
            column
            for column in metric_columns
            if column in fine_tuning_results.columns
        ]
        sorted_fine_tuning_results = fine_tuning_results.sort_values(
            [
                column
                for column in [
                    "scenario",
                    "f1_macro",
                    "critical_2_to_0",
                    "recall_c2",
                ]
                if column in fine_tuning_results.columns
            ],
            ascending=[
                True,
                False,
                True,
                False,
            ][:sum(
                column in fine_tuning_results.columns
                for column in [
                    "scenario",
                    "f1_macro",
                    "critical_2_to_0",
                    "recall_c2",
                ]
            )],
        )
        print("\nSYNTHÈSE DU FINE-TUNING")
        print(
            sorted_fine_tuning_results[
                available_columns
            ].to_string(index=False)
        )
        return

    scenario_labels = {
        "multimodal_complet": "Multimodal complet",
        "multimodal_ethique": "Multimodal éthique",
        "texte_seul": "Texte seul",
        "tabulaire_seul": "Tabulaire seul",
    }

    model_labels = {
        "random_forest": "Random Forest",
        "logistic_regression": "Régression logistique",
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
    }

    def format_score(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "N/A"

    def format_percentage(value: float | None) -> str:
        return f"{value * 100:.2f} %" if value is not None else "N/A"

    headers = [
        "Scénario",
        "Modèle",
        "Configuration",
        "F1-macro CV",
        "σ(F1)",
        "Recall classe 2",
        "σ(Recall C2)",
        "Erreur critique 2→0",
    ]

    table_rows: list[list[str]] = []

    sorted_results = sorted(
        results,
        key=lambda result: (
            result["scenario_name"],
            result["model_type"],
        ),
    )

    for result in sorted_results:
        cv_metrics = (
            result
            .get(
                "cv_summary",
                {},
            )
            .get(
                "metrics",
                {},
            )
        )

        f1_macro_cv = (
            cv_metrics
            .get(
                "f1_macro",
                {},
            )
            .get(
                "mean"
            )
        )

        recall_classe_2_cv = (
            cv_metrics
            .get(
                "recall_classe_2",
                {},
            )
            .get(
                "mean"
            )
        )

        f1_macro_std = (
            cv_metrics
            .get(
                "f1_macro",
                {},
            )
            .get(
                "std"
            )
        )

        recall_classe_2_std = (
            cv_metrics
            .get(
                "recall_classe_2",
                {},
            )
            .get(
                "std"
            )
        )

        erreur_critique_cv = (
            cv_metrics
            .get(
                "erreur_critique_2_vers_0",
                {},
            )
            .get(
                "mean"
            )
        )

        table_rows.append(
            [
                str(
                    scenario_labels.get(
                        result["scenario_name"],
                        result["scenario_name"],
                    )
                ),
                str(
                    model_labels.get(
                        result["model_type"],
                        result["model_type"],
                    )
                ),
                str(result["config_name"]),
                format_score(f1_macro_cv),
                format_score(f1_macro_std),
                format_score(recall_classe_2_cv),
                format_score(recall_classe_2_std),
                format_percentage(erreur_critique_cv),
            ]
        )

    table_rows = [
        [str(cell) for cell in row]
        for row in table_rows
    ]

    column_widths = [
        max(
            len(header),
            *(len(row[index]) for row in table_rows),
        )
        for index, header in enumerate(headers)
    ]

    def format_row(
        values: list[str],
        numeric: bool = False,
    ) -> str:
        cells = [
            (
                value.rjust(column_widths[index])
                if numeric and index >= 3
                else value.ljust(column_widths[index])
            )
            for index, value in enumerate(values)
        ]
        return "| " + " | ".join(cells) + " |"

    print("\nSYNTHÈSE DES ENTRAÎNEMENTS")
    print(format_row(headers))
    print(
        "| "
        + " | ".join(
            (
                "-" * column_width
                if index < 3
                else ":" + "-" * (column_width - 1)
            )
            for index, column_width in enumerate(column_widths)
        )
        + " |"
    )

    for row in table_rows:
        print(format_row(row, numeric=True))

    total_seconds = sum(
        result[
            "training_seconds"
        ]
        for result in sorted_results
    )

    print(
        f"Nombre de modèles : {len(results)}"
    )

    print(
        f"Temps de fit final cumulé : "
        f"{total_seconds:.3f} seconde(s)"
    )

    #print(
    #    "Évaluation finale du holdout : "
    #    "à réaliser avec src/evaluate.py"
    #)


# =============================================================================
# 14. ROBUSTESSE (ÉCART-TYPE CV)
# =============================================================================

def build_robustness_table(
    results: list[dict[str, Any]],
    metric: str = "f1_macro",
) -> pd.DataFrame:
    """
    Construit le tableau d'écart-type CV par scénario et par modèle.

    Un écart-type faible indique que la métrique varie peu entre les folds
    et correspond donc à un modèle plus robuste pour le scénario concerné.
    """
    rows: list[dict[str, Any]] = []

    for result in results:
        metric_summary = (
            result
            .get("cv_summary", {})
            .get("metrics", {})
            .get(metric, {})
        )

        rows.append(
            {
                "scenario_name": result["scenario_name"],
                "model_type": result["model_type"],
                "config_name": result["config_name"],
                "metric": metric,
                "mean": metric_summary.get("mean"),
                "std": metric_summary.get("std"),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "scenario_name",
            "model_type",
            "config_name",
            "metric",
            "mean",
            "std",
        ],
    ).sort_values(
        ["scenario_name", "model_type"],
        ignore_index=True,
    )


class ModelRobustnessAnalyzer:
    """
    Calcule la robustesse de chaque combinaison modèle/scénario à partir de
    l'écart-type de la validation croisée (plus l'écart-type est faible,
    plus le score est stable d'un fold à l'autre, donc plus le modèle est
    considéré robuste).

    Le tableau public est trié par scénario puis par modèle, pas par
    robustesse : c'est un tableau de lecture, pas un palmarès.
    """

    def __init__(
        self,
        results: list[dict[str, Any]],
        metric: str = "f1_macro",
    ) -> None:
        self.results = results
        self.metric = metric

    def compute(self) -> list[dict[str, Any]]:
        """
        Retourne une ligne par combinaison, avec la moyenne et l'écart-type
        CV de la métrique demandée, triée par scenario_name puis model_type.
        """
        table = build_robustness_table(
            results=self.results,
            metric=self.metric,
        )

        return table[
            [
                "model_type",
                "scenario_name",
                "config_name",
                "mean",
                "std",
            ]
        ].to_dict(
            orient="records"
        )

    def print_report(self) -> None:
        """Affiche le tableau de robustesse (écart-type CV de `self.metric`)."""
        rows = self.compute()

        print("\n" + "=" * 100)
        print(
            f"ROBUSTESSE PAR MODÈLE / SCÉNARIO "
            f"(écart-type CV sur {self.metric})"
        )
        print("=" * 100)

        for row in rows:
            mean_text = (
                f"{row['mean']:.4f}"
                if row["mean"] is not None
                else "N/A"
            )

            std_text = (
                f"{row['std']:.4f}"
                if row["std"] is not None
                else "N/A"
            )

            print(
                f"{row['model_type']:20s}"
                f" | {row['scenario_name']:25s}"
                f" | config={row['config_name']:16s}"
                f" | {self.metric} moyenne={mean_text:8s}"
                f" | écart-type={std_text:8s}"
            )

        print("=" * 100)


# =============================================================================
# 15. INTERFACE EN LIGNE DE COMMANDE
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Analyse les arguments de la ligne de commande.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Entraîne les modèles CISIA "
            "et réalise une validation croisée "
            "sur le train uniquement."
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
            *SCENARIOS.keys(),
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
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help=(
            "Nombre de folds de validation croisée. "
            "Utiliser 0 ou 1 pour désactiver la CV."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Point d'entrée du script.
    """
    args = parse_args()

    print("\nParamètres d'entraînement :")

    print(
        f"  Modèle        : {args.model_type}"
    )

    print(
        f"  Scénario      : {args.scenario}"
    )

    print(
        f"  Configuration : {args.config}"
    )

    print(
        f"  CV folds      : {args.cv_folds}"
    )

    print(
        f"  Dataset       : {args.data}"
    )

    print(
        f"  Sortie        : {args.output}"
    )

    results = train_requested_models(
        requested_model_type=(
            args.model_type
        ),
        requested_scenario=(
            args.scenario
        ),
        config_name=args.config,
        data_path=args.data,
        output_dir=args.output,
        cv_folds=args.cv_folds,
    )

    print_training_summary(
        results
    )

    ModelRobustnessAnalyzer(
        results
    ).print_report()

    print(
        "\nCommandes d'évaluation finale :"
    )

    for result in results:
        print(
            "python src/evaluate.py "
            f'--model-type "{result["model_type"]}" '
            f'--scenario "{result["scenario_name"]}" '
            f'--config "{result["config_name"]}" '
            f'--data "{args.data}"'
        )


if __name__ == "__main__":
    main()