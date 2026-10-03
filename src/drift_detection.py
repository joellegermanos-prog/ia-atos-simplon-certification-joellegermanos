"""Détection de dérive — PSI, KS, Chi² (SQUELETTE À COMPLÉTER).

Trois méthodes complémentaires. Mini-cours : `01_PSI_KS_Chi2_essentiel.md`.
N'inventez pas vos métriques : PSI (formule ci-dessous), KS et Chi² sont dans
scipy.stats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

PSI_STABLE = 0.10
PSI_DRIFT = 0.25
ALPHA = 0.05
EPSILON = 1e-6
MIN_EXPECTED_COUNT = 5.0

# PSI #

def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    n_bins: int = 10,
) -> float:
    """PSI entre référence et courant.
 
    PSI = Σ (p_cur - p_ref) * ln(p_cur / p_ref), bornes des bins = quantiles de
    la référence. Lissage anti-zéro par epsilon **puis renormalisation** : sans
    renormaliser, les proportions ne somment plus à 1 et le PSI est biaisé.
 
    Renvoie ``nan`` si un échantillon est vide, ``0.0`` si la référence est
    quasi constante (aucun bin exploitable).
    """
    ref_values = pd.Series(reference).dropna().to_numpy(dtype=float)
    cur_values = pd.Series(current).dropna().to_numpy(dtype=float)
    if ref_values.size == 0 or cur_values.size == 0:
        return float("nan")
 
    edges = np.unique(np.quantile(ref_values, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        return 0.0
    # bornes ouvertes : une valeur prod hors du support de la référence doit
    # tomber dans un bin, pas être ignorée silencieusement par np.histogram.
    edges[0], edges[-1] = -np.inf, np.inf
 
    p_ref = np.histogram(ref_values, edges)[0] / ref_values.size
    p_cur = np.histogram(cur_values, edges)[0] / cur_values.size
 
    p_ref, p_cur = p_ref + EPSILON, p_cur + EPSILON
    p_ref, p_cur = p_ref / p_ref.sum(), p_cur / p_cur.sum()
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))

def psi_verdict(psi: float) -> str:
    """Traduit un PSI en verdict (stable / suspect / dérive).
 
    Repères conventionnels du credit scoring, pas des lois statistiques :
    0.24 et 0.26 disent la même chose.
    """
    if pd.isna(psi):
        return "non calculable"
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_DRIFT:
        return "suspect"
    return "dérive"

# Tests statistiques #

def ks_pvalue(reference: pd.Series, current: pd.Series) -> float:
    """p-value du test de Kolmogorov-Smirnov (2 échantillons)."""
    ref_values = pd.Series(reference).dropna()
    cur_values = pd.Series(current).dropna()
    if ref_values.empty or cur_values.empty:
        return float("nan")
    return float(ks_2samp(ref_values, cur_values).pvalue)

def chi2_pvalue(reference: pd.Series, current: pd.Series) -> float:
    """p-value du Chi² sur les fréquences de modalités.
 
    Les modalités sont réindexées sur leur union avant construction de la table :
    sans alignement, une modalité présente d'un seul côté fait planter la table
    de contingence. Lissage +1 pour éviter les cases à effectif nul.
    """
    counts_ref = pd.Series(reference).value_counts()
    counts_cur = pd.Series(current).value_counts()
    if counts_ref.empty or counts_cur.empty:
        return float("nan")
 
    modalities = counts_ref.index.union(counts_cur.index)
    if len(modalities) < 2:
        return float("nan")  # une seule modalité : rien à tester
 
    table = np.vstack([
        counts_ref.reindex(modalities, fill_value=0).to_numpy() + 1,
        counts_cur.reindex(modalities, fill_value=0).to_numpy() + 1,
    ])
    return float(chi2_contingency(table)[1])

def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    """Tableau de synthèse : feature / type / psi / ks_pvalue / chi2_pvalue / verdict.
 
    Tri par sévérité décroissante (PSI) : la première ligne doit être la feature
    qui bouge le plus, pas la première dans l'ordre alphabétique.
    """
    records: list[dict[str, object]] = []
 
    for column in numeric_cols:
        psi = population_stability_index(reference[column], current[column])
        records.append({
            "feature": column,
            "type": "numerique",
            "psi": psi,
            "ks_pvalue": ks_pvalue(reference[column], current[column]),
            "chi2_pvalue": float("nan"),
            "verdict": psi_verdict(psi),
        })
 
    for column in categorical_cols:
        # PSI catégoriel : les bins sont les modalités. Sans lui, les
        # catégorielles n'auraient qu'une p-value et aucune mesure d'ampleur.
        p_ref = reference[column].value_counts(normalize=True)
        p_cur = current[column].value_counts(normalize=True)
        modalities = p_ref.index.union(p_cur.index)
        prop_ref = p_ref.reindex(modalities, fill_value=0.0).to_numpy() + EPSILON
        prop_cur = p_cur.reindex(modalities, fill_value=0.0).to_numpy() + EPSILON
        prop_ref, prop_cur = prop_ref / prop_ref.sum(), prop_cur / prop_cur.sum()
        psi = float(np.sum((prop_cur - prop_ref) * np.log(prop_cur / prop_ref)))
        records.append({
            "feature": column,
            "type": "categorielle",
            "psi": psi,
            "ks_pvalue": float("nan"),
            "chi2_pvalue": chi2_pvalue(reference[column], current[column]),
            "verdict": psi_verdict(psi),
        })
 
    columns = ["feature", "type", "psi", "ks_pvalue", "chi2_pvalue", "verdict"]
    if not records:
        return pd.DataFrame(columns=columns).set_index("feature")
 
    return pd.DataFrame(records).set_index("feature").sort_values("psi", ascending=False)
