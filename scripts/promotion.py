"""Promotion policy for CISIA candidate models."""
from __future__ import annotations

from dataclasses import dataclass, field

QUALITY_FLOORS: dict[str, float] = {
    "f1_macro": 0.60,
    "f1_classe_2": 0.50,
    "recall_classe_2": 0.50,
    "roc_auc_ovr_macro": 0.70,
}

# The recall of class 2 is the critical business metric. A small measurement
# variation is tolerated, but a meaningful regression blocks promotion.
REGRESSION_TOLERANCES: dict[str, float] = {
    "f1_macro": 0.02,
    "f1_classe_2": 0.03,
    "recall_classe_2": 0.03,
    "roc_auc_ovr_macro": 0.02,
}

IMPROVEMENT_METRICS: tuple[str, ...] = ("f1_macro", "recall_classe_2")


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str
    rules: tuple[dict, ...] = field(default_factory=tuple)


def evaluate_quality_gate(
    candidate: dict[str, float],
    production: dict[str, float],
) -> list[dict]:
    """Evaluate every promotion rule and return one PASS/FAIL entry per rule."""
    rules: list[dict] = []
    for metric, floor in QUALITY_FLOORS.items():
        value = float(candidate.get(metric, 0.0))
        rules.append({
            "rule": f"floor.{metric}",
            "type": "quality_floor",
            "metric": metric,
            "value": value,
            "threshold": floor,
            "comparison": ">=",
            "status": "PASS" if value >= floor else "FAIL",
        })
    for metric, tolerance in REGRESSION_TOLERANCES.items():
        drop = float(production.get(metric, 0.0)) - float(candidate.get(metric, 0.0))
        rules.append({
            "rule": f"regression.{metric}",
            "type": "max_regression_vs_production",
            "metric": metric,
            "value": drop,
            "threshold": tolerance,
            "comparison": "<=",
            "status": "PASS" if drop <= tolerance else "FAIL",
        })
    gains = {
        metric: float(candidate.get(metric, 0.0)) - float(production.get(metric, 0.0))
        for metric in IMPROVEMENT_METRICS
    }
    rules.append({
        "rule": "improvement.any_of",
        "type": "minimum_gain_vs_production",
        "metric": "|".join(IMPROVEMENT_METRICS),
        "value": max(gains.values()),
        "threshold": 0.0,
        "comparison": ">",
        "status": "PASS" if max(gains.values()) > 0 else "FAIL",
        "details": gains,
    })
    return rules


def decide_promotion(
    candidate: dict[str, float],
    production: dict[str, float],
) -> PromotionDecision:
    """Allow promotion only if floors pass and no metric regresses too far."""
    rules = tuple(evaluate_quality_gate(candidate, production))
    failed = [rule for rule in rules if rule["status"] == "FAIL"]

    floors = [rule for rule in failed if rule["type"] == "quality_floor"]
    if floors:
        rule = floors[0]
        return PromotionDecision(
            False,
            f"quality floor not met: {rule['metric']}={rule['value']:.4f} < {rule['threshold']:.4f}",
            rules,
        )

    regressions = [
        f"{rule['metric']} drop={rule['value']:.4f} > {rule['threshold']:.4f}"
        for rule in failed
        if rule["type"] == "max_regression_vs_production"
    ]
    if regressions:
        return PromotionDecision(False, "critical regression: " + "; ".join(regressions), rules)

    improvement = rules[-1]
    if improvement["status"] == "FAIL":
        return PromotionDecision(
            False,
            "candidate does not improve f1_macro or recall_classe_2",
            rules,
        )

    gains = improvement["details"]
    return PromotionDecision(
        True,
        f"candidate accepted: f1_macro={gains['f1_macro']:+.4f}, "
        f"recall_classe_2={gains['recall_classe_2']:+.4f}",
        rules,
    )
