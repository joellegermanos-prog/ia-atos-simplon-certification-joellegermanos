"""Logique de recommandation de remédiation (SQUELETTE À COMPLÉTER).

La remédiation doit être **proportionnée** au diagnostic : réentraîner coûte
cher, on ne le propose que quand ça vaut le coup. Mini-cours :
`02_Data_drift_vs_concept_drift_essentiel.md` (matrice features × AUC).
"""

from __future__ import annotations

from dataclasses import dataclass

F1_DROP_TOLERE = 0.03

@dataclass
class DriftDiagnosis:
    """Synthèse du diagnostic pour décider de la remédiation."""

    n_features_drift: int  # nb de features en "dérive" (PSI > 0.25)
    auc_stable: bool  # le pouvoir discriminant tient-il ?
    calibration_degraded: bool  # la confiance a-t-elle dérivé ?
    f1_drop: float  # baisse de F1 macro (early → late)


def diagnose_drift_type(d: DriftDiagnosis) -> str:
    """Oriente vers "data drift" / "concept drift" / "mixte".

    ⚠️ Heuristique d'orientation, pas une preuve : elle formalise la matrice
    du mini-cours 02 (features × AUC) pour produire une hypothèse principale.
    Le verdict final se construit en croisant features, AUC, calibration et
    temporalité — et doit énoncer ce qui manquerait pour trancher.
    """
    features_derivent = d.n_features_drift > 0
    if features_derivent and d.auc_stable:
        return "data drift"
    if not features_derivent and not d.auc_stable:
        return "concept drift"
    if features_derivent and not d.auc_stable:
        return "mixte"
    return "pas de signal"


def recommend(d: DriftDiagnosis) -> dict[str, str]:
    """Recommande une action proportionnée au diagnostic.
 
    Returns:
        dict avec les clés : action / justification / urgence / drift_type.
    """
    drift_type = diagnose_drift_type(d)
 
    if drift_type == "concept drift" or (drift_type == "mixte" and d.f1_drop > F1_DROP_TOLERE):
        action = "Réentraîner en urgence et investiguer la cause"
        justification = (
            "Le modèle ne classe plus aussi bien : la logique de risque elle-même "
            f"semble avoir changé (F1 -{d.f1_drop:.2f}). Attendre aggrave la perte."
        )
        urgence = "sous 1 semaine"
    elif drift_type == "data drift" and (d.calibration_degraded or d.f1_drop > F1_DROP_TOLERE):
        action = "Réentraîner sur données récentes"
        justification = (
            "Le modèle trie toujours correctement les dossiers, mais la clientèle "
            "entrante a changé : les probabilités annoncées ne sont plus fiables. "
            "Un réentraînement les recale sans revoir l'architecture."
        )
        urgence = "sous 3 semaines"
    elif drift_type == "data drift":
        action = "Surveiller, sans réentraîner"
        justification = (
            "Les données entrantes ont bougé, mais ni le tri des dossiers ni la "
            "fiabilité des probabilités n'en souffrent à ce stade. Réentraîner "
            "coûterait sans bénéfice mesurable."
        )
        urgence = "revue mensuelle"
    elif drift_type == "mixte":
        action = "Ajuster le seuil de décision et resserrer la surveillance"
        justification = (
            "La performance bouge mais reste dans la tolérance : un ajustement de "
            "seuil coûte quelques heures là où un réentraînement coûte des jours."
        )
        urgence = "sous 4 semaines"
    else:
        action = "Aucune action, maintenir la surveillance en place"
        justification = "Aucun signal statistique significatif sur la période observée."
        urgence = "revue trimestrielle"
 
    return {
        "action": action,
        "justification": justification,
        "urgence": urgence,
        "drift_type": drift_type,
    }