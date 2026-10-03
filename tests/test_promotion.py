from scripts.promotion import (
    QUALITY_FLOORS,
    REGRESSION_TOLERANCES,
    decide_promotion,
    evaluate_quality_gate,
)


def test_promotion_accepts_candidate_with_class_two_gain():
    decision = decide_promotion(
        {
            "f1_macro": 0.74,
            "f1_classe_2": 0.66,
            "recall_classe_2": 0.70,
            "roc_auc_ovr_macro": 0.88,
        },
        {
            "f1_macro": 0.72,
            "f1_classe_2": 0.63,
            "recall_classe_2": 0.65,
            "roc_auc_ovr_macro": 0.87,
        },
    )

    assert decision.promote is True


def test_promotion_rejects_class_two_regression():
    decision = decide_promotion(
        {
            "f1_macro": 0.74,
            "f1_classe_2": 0.60,
            "recall_classe_2": 0.60,
            "roc_auc_ovr_macro": 0.88,
        },
        {
            "f1_macro": 0.72,
            "f1_classe_2": 0.63,
            "recall_classe_2": 0.65,
            "roc_auc_ovr_macro": 0.87,
        },
    )

    assert decision.promote is False
    assert "regression" in decision.reason


def test_quality_gate_reports_each_rule():
    candidate = {"f1_macro": 0.74, "f1_classe_2": 0.60, "recall_classe_2": 0.60, "roc_auc_ovr_macro": 0.88}
    production = {"f1_macro": 0.72, "f1_classe_2": 0.63, "recall_classe_2": 0.65, "roc_auc_ovr_macro": 0.87}

    rules = {rule["rule"]: rule for rule in evaluate_quality_gate(candidate, production)}

    assert len(rules) == len(QUALITY_FLOORS) + len(REGRESSION_TOLERANCES) + 1
    assert rules["floor.recall_classe_2"]["status"] == "PASS"
    assert rules["regression.recall_classe_2"]["status"] == "FAIL"
    assert rules["improvement.any_of"]["status"] == "PASS"
    assert decide_promotion(candidate, production).rules
