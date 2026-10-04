"""Calibration en exploitation .
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _calibration_frame(
    proba: pd.Series,
    true_label: pd.Series,
    n_bins: int,
) -> pd.DataFrame:
    if n_bins < 1:
        raise ValueError("n_bins doit être supérieur ou égal à 1")

    frame = pd.DataFrame({"proba": proba, "true_label": true_label}).dropna()
    if frame.empty:
        return frame.assign(bin=pd.Series(dtype="object"))
    if ((frame["proba"] < 0) | (frame["proba"] > 1)).any():
        raise ValueError("Les probabilités doivent être comprises entre 0 et 1")
    if (~frame["true_label"].isin([0, 1])).any():
        raise ValueError("Les labels doivent valoir 0 ou 1")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    frame["bin"] = pd.cut(frame["proba"], edges, include_lowest=True)
    return frame


def reliability_table(proba: pd.Series, true_label: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Table du reliability diagram : bin / n / confiance_moyenne / taux_observe / ecart."""
    frame = _calibration_frame(proba, true_label, n_bins)
    columns = ["bin", "n", "confiance_moyenne", "taux_observe", "ecart"]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    table = (
        frame.groupby("bin", observed=True)
        .agg(
            n=("true_label", "size"),
            confiance_moyenne=("proba", "mean"),
            taux_observe=("true_label", "mean"),
        )
        .reset_index()
    )
    table["ecart"] = table["confiance_moyenne"] - table["taux_observe"]
    return table[columns]


def expected_calibration_error(proba: pd.Series, true_label: pd.Series, n_bins: int = 10) -> float:
    """ECE = Σ (n_bin/N) * |confiance - taux observé|. 0 = parfaitement calibré."""
    table = reliability_table(proba, true_label, n_bins)
    if table.empty:
        return float("nan")
    weights = table["n"] / table["n"].sum()
    return float((weights * table["ecart"].abs()).sum())


def calibration_degraded(
    reference_proba: pd.Series,
    reference_label: pd.Series,
    current_proba: pd.Series,
    current_label: pd.Series,
    n_bins: int = 10,
    min_increase: float = 0.0,
) -> bool:
    """Indique si l'ECE courant dépasse celui de référence.

    Les deux ECE utilisent exactement le même binning. ``min_increase`` permet
    d'ignorer une variation inférieure à une tolérance métier.
    """
    reference_ece = expected_calibration_error(reference_proba, reference_label, n_bins)
    current_ece = expected_calibration_error(current_proba, current_label, n_bins)
    if pd.isna(reference_ece) or pd.isna(current_ece):
        return False
    return bool(current_ece - reference_ece > min_increase)
