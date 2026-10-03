"""
Contract test du modèle CISIA.

Ce script valide qu'un pipeline sauvegardé par train.py :

    - se recharge correctement ;
    - accepte les données attendues ;
    - produit des prédictions ;
    - produit des probabilités valides ;
    - respecte les classes attendues ;
    - conserve une stabilité prédictive.

Utilisation :

    python src/contract_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation_utils import compute_sha256
from preprocess import load_dataset

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "cisia_emploi_xgboost_multimodal_complet_balanced.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "cisia_emploi_xgboost_multimodal_complet_balanced.json"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "dataset_trajectoire_emploi.csv"
)


EXPECTED_CLASSES = {0, 1, 2}


def load_metadata(
    metadata_path: Path,
) -> dict:
    return json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )


def contract_test_model(
    model_path: Path,
    metadata_path: Path,
    data_path: Path,
    expected_first_prediction: int | None = None,
    expected_first_proba: list[float] | None = None,
) -> None:
    """
    Vérifie le contrat du modèle CISIA.
    """

    metadata = load_metadata(
        metadata_path
    )

    # ------------------------------------------------------------------
    # Vérification dataset
    # ------------------------------------------------------------------

    expected_hash = metadata[
        "dataset"
    ][
        "sha256"
    ]

    current_hash = compute_sha256(
        data_path
    )

    assert (
        current_hash
        == expected_hash
    ), (
        "Le dataset utilisé pour le test "
        "n'est pas celui utilisé pour le train."
    )

    # ------------------------------------------------------------------
    # Chargement modèle
    # ------------------------------------------------------------------

    pipeline = joblib.load(
        model_path
    )

    scenario = metadata[
        "scenario"
    ]

    X, _ = load_dataset(
        path=data_path,
        include_sensitive=scenario.get(
            "include_sensitive",
            True,
        ),
        include_text=scenario.get(
            "include_text",
            True,
        ),
    )

    x_sample = X.head(
        5
    )

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    prediction = pipeline.predict(
        x_sample
    )

    assert (
        prediction.shape
        == (5,)
    ), (
        f"Shape inattendue : "
        f"{prediction.shape}"
    )

    observed_classes = set(
        prediction.tolist()
    )

    assert (
        observed_classes
        <= EXPECTED_CLASSES
    ), (
        "Classes inattendues : "
        f"{observed_classes - EXPECTED_CLASSES}"
    )

    # ------------------------------------------------------------------
    # predict_proba
    # ------------------------------------------------------------------

    assert hasattr(
        pipeline,
        "predict_proba"
    ), (
        "Le pipeline ne supporte pas "
        "predict_proba."
    )

    proba = pipeline.predict_proba(
        x_sample
    )

    assert (
        proba.shape
        == (5, 3)
    ), (
        f"Shape predict_proba={proba.shape}, "
        "attendu (5,3)"
    )

    assert (
        (proba >= 0).all()
        and (proba <= 1).all()
    ), (
        "Probabilités hors [0,1]"
    )

    sums = proba.sum(
        axis=1
    )

    assert np.allclose(
        sums,
        1.0,
        atol=1e-5,
    ), (
        "Les probabilités ne somment "
        "pas à 1."
    )

    # ------------------------------------------------------------------
    # stabilité prédictive
    # ------------------------------------------------------------------

    if expected_first_prediction is not None:

        observed_prediction = int(
            prediction[0]
        )

        assert (
            observed_prediction
            == expected_first_prediction
        ), (
            "Dérive de prédiction. "
            f"Observé={observed_prediction}, "
            f"Référence={expected_first_prediction}"
        )

    if expected_first_proba is not None:

        observed_proba = (
            proba[0]
            .round(6)
            .tolist()
        )

        reference = [
            round(v, 6)
            for v in expected_first_proba
        ]

        assert (
            observed_proba
            == reference
        ), (
            "Dérive prédictive.\n"
            f"Observé: {observed_proba}\n"
            f"Référence: {reference}"
        )

    print(
        "\n✅ Contract test OK"
    )

    print(
        "  Pipeline chargé"
    )

    print(
        "  Shapes correctes"
    )

    print(
        "  Classes correctes"
    )

    print(
        "  Probabilités valides"
    )

    print(
        "  Dataset cohérent"
    )


def test_contract_model() -> None:
    """Valide le contrat fonctionnel du modèle enregistré."""
    contract_test_model(
        model_path=MODEL_PATH,
        metadata_path=METADATA_PATH,
        data_path=DATA_PATH,
        expected_first_prediction=None,
        expected_first_proba=None,
    )


if __name__ == "__main__":
    test_contract_model()