"""Tests de démarrage — verts sur le squelette, plus exigeants ensuite.

Lancez `pytest` dès le clone : tout doit passer. Les tests marqués
"débloqué par TODO n" sautent tant que la fonction n'est pas implémentée,
puis deviennent de vrais garde-fous une fois vos TODO complétés.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibration import (
    calibration_degraded,
    expected_calibration_error,
    reliability_table,
)
from drift_detection import (
    PSI_DRIFT,
    PSI_STABLE,
    population_stability_index,
    psi_verdict,
)
from recommendations import DriftDiagnosis, diagnose_drift_type, recommend


def test_data_files_present():
    """Les 3 CSV fournis sont là — sinon relisez le README (section données)."""
    for name in ("reference_set.csv", "prod_3months.csv", "predictions_log.csv"):
        assert (ROOT / "data" / name).exists(), f"data/{name} manquant"


def test_colonnes_reference_presentes_en_prod():
    """Toute feature de la référence doit exister en prod (comparaison possible).

    La prod peut avoir des colonnes en plus (ex. `timestamp`), jamais en moins.
    """
    ref = pd.read_csv(ROOT / "data" / "reference_set.csv", nrows=5)
    prod = pd.read_csv(ROOT / "data" / "prod_3months.csv", nrows=5)
    manquantes = set(ref.columns) - set(prod.columns)
    assert not manquantes, f"colonnes absentes de la prod : {sorted(manquantes)}"


def test_seuils_psi_coherents():
    """Les seuils du mini-cours 01 ne doivent pas être modifiés."""
    assert 0 < PSI_STABLE < PSI_DRIFT


def test_psi_identique_proche_de_zero():
    """Débloqué par TODO 1 (drift_detection) : PSI(x, x) ≈ 0."""
    serie = pd.Series(range(1000), dtype=float)
    try:
        psi = population_stability_index(serie, serie)
    except NotImplementedError:
        pytest.skip("TODO 1 de drift_detection.py à compléter")
    assert psi == pytest.approx(0.0, abs=1e-6)


def test_psi_verdict_borne():
    """Débloqué par TODO 2 (drift_detection) : verdicts aux bornes."""
    try:
        assert psi_verdict(0.0) == "stable"
        assert psi_verdict(0.5) == "dérive"
    except NotImplementedError:
        pytest.skip("TODO 2 de drift_detection.py à compléter")


def test_diagnose_drift_type_cas_canonique():
    """Débloqué par TODO 1 (recommendations) : cas d'école du mini-cours 02."""
    d = DriftDiagnosis(
        n_features_drift=2, auc_stable=True, calibration_degraded=True, f1_drop=0.06
    )
    try:
        assert diagnose_drift_type(d) == "data drift"
    except NotImplementedError:
        pytest.skip("TODO 1 de recommendations.py à compléter")


def test_calibration_ece_is_zero_when_probabilities_match_labels():
    proba = pd.Series([0.0, 0.0, 1.0, 1.0])
    labels = pd.Series([0, 0, 1, 1])

    assert expected_calibration_error(proba, labels, n_bins=10) == pytest.approx(0.0)
    table = reliability_table(proba, labels, n_bins=10)
    assert list(table.columns) == [
        "bin",
        "n",
        "confiance_moyenne",
        "taux_observe",
        "ecart",
    ]


def test_calibration_degraded_feeds_data_drift_recommendation():
    reference_proba = pd.Series([0.1, 0.1, 0.9, 0.9])
    reference_labels = pd.Series([0, 0, 1, 1])
    current_proba = pd.Series([0.9, 0.9, 0.9, 0.9])
    current_labels = pd.Series([0, 0, 1, 1])

    degraded = calibration_degraded(
        reference_proba,
        reference_labels,
        current_proba,
        current_labels,
        n_bins=10,
    )
    diagnosis = DriftDiagnosis(
        n_features_drift=1,
        auc_stable=True,
        calibration_degraded=degraded,
        f1_drop=0.056,
    )

    result = recommend(diagnosis)

    assert degraded is True
    assert result["drift_type"] == "data drift"
    assert result["action"] == "Réentraîner sur données récentes"
    assert result["urgence"] == "sous 3 semaines"
