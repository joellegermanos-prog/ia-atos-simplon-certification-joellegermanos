from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "dataset_trajectoire_emploi.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_NAME = "cisia_emploi"
MODEL_VERSION = "v1.0.0"
RANDOM_STATE = 42
TEST_SIZE = 0.20

AVAILABLE_MODEL_TYPES = (
    "random_forest",
    "logistic_regression",
    "lightgbm",
    "xgboost",
)

AVAILABLE_CONFIGS = (
    "default",
    "balanced",
    "critical_class_2",
    "regularized",
)

SCENARIOS: dict[str, dict[str, Any]] = {
    "multimodal_complet": {
        "description": "Variables tabulaires, texte et variables sensibles.",
        "include_sensitive": True,
        "include_text": True,
        "text_only": False,
    },
    "multimodal_ethique": {
        "description": "Variables tabulaires et texte sans variables sensibles.",
        "include_sensitive": False,
        "include_text": True,
        "text_only": False,
    },
    "texte_seul": {
        "description": "Variable texte uniquement.",
        "include_sensitive": False,
        "include_text": True,
        "text_only": True,
    },
    "tabulaire_seul": {
        "description": "Variables tabulaires uniquement, sans texte.",
        "include_sensitive": True,
        "include_text": False,
        "text_only": False,
        "features": ["age", "departement", "niveau_diplome", "anciennete_poste_ans"],
    },
}
