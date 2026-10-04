"""Métriques métier Prometheus — service model (fourni — votre exemple de référence).

En plus des métriques HTTP standard exposées par
``prometheus-fastapi-instrumentator`` (latence, RPS, codes retour), on
expose 2 métriques **métier** qui répondent à la 3ᵉ question de Sophie
Léger : *« le modèle prédit-il toujours bien ? »*

- ``cisia_emploi_predictions_total`` : compteur des prédictions, labellé par
  classe prédite (0 = retour rapide, 1 = retour moyen, 2 = longue durée). La dérive de la répartition
  0/1 dans le temps est un signal d'alerte (data drift / concept drift).
- ``cisia_emploi_prediction_proba`` : histogramme des probabilités de la classe prédite
  renvoyées. Un modèle sain produit une distribution étalée ; un pic à
  0.5 ou aux bornes signale un problème.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

MODEL_INFO = Gauge(
    "cisia_model_info",
    "Modele CISIA charge dans le service.",
    labelnames=("model_name", "model_version", "scenario", "calibration_method"),
)

PREDICTIONS_TOTAL = Counter(
    "cisia_emploi_predictions_total",
    "Nombre de prédictions servies, par classe prédite.",
    labelnames=("predicted_class",),
)

for prediction_class in (0, 1, 2):
    PREDICTIONS_TOTAL.labels(predicted_class=str(prediction_class))

PREDICTION_PROBA = Histogram(
    "cisia_emploi_prediction_proba",
    "Distribution des probabilités de la classe prédite.",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)


def observe_prediction(predicted_class: int, proba_default: float) -> None:
    """Enregistre une prédiction dans les métriques métier.

    Args:
        predicted_class: Classe prédite (0 = retour rapide, 1 = retour moyen, 2 = longue durée).
        proba_default: Probabilité de la classe prédite renvoyée par le modèle.
    """
    PREDICTIONS_TOTAL.labels(predicted_class=str(predicted_class)).inc()
    PREDICTION_PROBA.observe(proba_default)
