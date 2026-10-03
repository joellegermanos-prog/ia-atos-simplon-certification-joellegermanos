import numpy as np
import pandas as pd

from scripts.drift_analysis import analyse, categorical_psi, chi2_result, ks_result, psi


def test_psi_detects_numeric_distribution_shift():
    reference = pd.Series(np.zeros(100))
    current = pd.Series(np.ones(100) * 10)
    assert psi(reference, current) > 0


def test_ks_and_chi2_return_significance_fields():
    ks = ks_result(pd.Series([0, 0, 0, 1]), pd.Series([1, 1, 1, 1]))
    chi2 = chi2_result(pd.Series(["a", "a", "b"]), pd.Series(["b", "b", "b"]))
    assert {"statistic", "p_value"} == set(ks)
    assert {"statistic", "p_value"} == set(chi2)


def test_categorical_psi_detects_distribution_shift():
    reference = pd.Series(["Bac", "Bac", "Bac+2", "Bac+2"])
    current = pd.Series(["Bac+5", "Bac+5", "Bac+5", "Bac+5"])
    assert categorical_psi(reference, current) > 0


def test_analyse_triangulates_inputs_labels_and_confidence():
    reference = pd.DataFrame(
        {
            "anciennete_poste_ans": [1, 1, 2, 2],
            "niveau_diplome": ["Bac", "Bac", "Bac+2", "Bac+2"],
            "code_rome_vise": ["A", "A", "B", "B"],
            "code_insee_commune": ["75001"] * 4,
            "classe_retour_emploi": [0, 0, 1, 1],
            "proba_0": [0.8, 0.7, 0.2, 0.1],
            "proba_1": [0.1, 0.2, 0.7, 0.8],
            "proba_2": [0.1, 0.1, 0.1, 0.1],
        }
    )
    current = reference.copy()
    current["anciennete_poste_ans"] += 20
    current["proba_2"] = 0.8
    report = analyse(reference, current)
    assert "anciennete_poste_ans" in report["data_drift"]["numeric"]
    assert report["concept_drift"]["available"] is True
    assert report["confidence_drift"]["available"] is True
