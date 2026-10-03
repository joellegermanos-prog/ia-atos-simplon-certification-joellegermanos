from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import SCENARIOS
from preprocess import (
    SENSITIVE_FEATURES,
    TEXT_FEATURE,
    build_preprocessor,
    get_all_features,
    load_dataset,
)

DATA_PATH = PROJECT_ROOT / "data" / "dataset_trajectoire_emploi.csv"


def test_print_variables_scenario_1() -> None:
    """Affiche et vérifie les variables du scénario multimodal complet."""
    scenario = SCENARIOS["multimodal_complet"]
    expected_features = get_all_features(
        include_sensitive=scenario["include_sensitive"],
        include_text=scenario["include_text"],
    )
    observed_features, _ = load_dataset(
        DATA_PATH,
        include_sensitive=scenario["include_sensitive"],
        include_text=scenario["include_text"],
    )

    print("\nScénario 1 : multimodal_complet")
    print(f"Variables ({len(expected_features)}) :")
    for feature in expected_features:
        print(f"- {feature}")

    assert list(observed_features.columns) == expected_features


def test_print_variables_scenario_2() -> None:
    """Affiche et vérifie les variables du scénario multimodal éthique."""
    scenario = SCENARIOS["multimodal_ethique"]
    expected_features = get_all_features(
        include_sensitive=scenario["include_sensitive"],
        include_text=scenario["include_text"],
    )
    observed_features, _ = load_dataset(
        DATA_PATH,
        include_sensitive=scenario["include_sensitive"],
        include_text=scenario["include_text"],
    )

    print("\nScénario 2 : multimodal_ethique")
    print(f"Variables ({len(expected_features)}) :")
    for feature in expected_features:
        print(f"- {feature}")

    assert list(observed_features.columns) == expected_features
    for feature in SENSITIVE_FEATURES:
        assert feature not in observed_features.columns


def test_print_variables_scenario_3() -> None:
    """Affiche et vérifie les variables du scénario texte seul."""
    scenario = SCENARIOS["texte_seul"]
    expected_features = [TEXT_FEATURE]
    observed_features, _ = load_dataset(
        DATA_PATH,
        include_sensitive=scenario["include_sensitive"],
        include_text=scenario["include_text"],
    )
    observed_features = observed_features[[TEXT_FEATURE]]

    print("\nScénario 3 : texte_seul")
    print(f"Variables ({len(expected_features)}) :")
    for feature in expected_features:
        print(f"- {feature}")

    assert list(observed_features.columns) == expected_features


def test_print_variables_scenario_4() -> None:
    """Affiche et vérifie les variables du scénario tabulaire seul."""
    scenario = SCENARIOS["tabulaire_seul"]
    expected_features = get_all_features(
        include_sensitive=scenario["include_sensitive"],
        include_text=scenario["include_text"],
        features=scenario.get("features"),
    )
    observed_features, _ = load_dataset(
        DATA_PATH,
        include_sensitive=scenario["include_sensitive"],
        include_text=scenario["include_text"],
        features=scenario.get("features"),
    )

    print("\nScénario 4 : tabulaire_seul")
    print(f"Variables ({len(expected_features)}) :")
    for feature in expected_features:
        print(f"- {feature}")

    assert list(observed_features.columns) == expected_features
    assert TEXT_FEATURE not in observed_features.columns


def test_full_scenario_keeps_sensitive_features() -> None:
    X, y = load_dataset(DATA_PATH, include_sensitive=True, include_text=True)

    assert len(X) > 0
    assert len(y) == len(X)

    for feature in SENSITIVE_FEATURES:
        assert feature in X.columns, f"Feature sensible absente: {feature}"

    assert "synthese_entretien" in X.columns


def test_ethic_scenario_removes_sensitive_features() -> None:
    X, y = load_dataset(DATA_PATH, include_sensitive=False, include_text=True)

    assert len(X) > 0
    assert len(y) == len(X)

    for feature in SENSITIVE_FEATURES:
        assert feature not in X.columns, f"Feature sensible encore présente: {feature}"

    assert "synthese_entretien" in X.columns


def test_tabular_only_scenario_excludes_text() -> None:
    X, y = load_dataset(DATA_PATH, include_sensitive=True, include_text=False)

    assert len(X) > 0
    assert len(y) == len(X)
    assert "synthese_entretien" not in X.columns


def test_preprocessor_accepts_scenario_flags() -> None:
    X, _y = load_dataset(DATA_PATH, include_sensitive=False, include_text=True)

    preprocessor = build_preprocessor(include_sensitive=False, include_text=True)
    Xt = preprocessor.fit_transform(X)

    assert Xt.shape[0] == len(X)
    assert Xt.shape[1] > 0

    if hasattr(Xt, "toarray"):
        Xt_dense = Xt.toarray()
    else:
        Xt_dense = Xt

    assert np.isnan(Xt_dense).sum() == 0


def test_scenario_flags_are_consistent_with_sensitive_list() -> None:
    full_X, _ = load_dataset(DATA_PATH, include_sensitive=True, include_text=True)
    ethic_X, _ = load_dataset(DATA_PATH, include_sensitive=False, include_text=True)

    for feature in SENSITIVE_FEATURES:
        assert feature in full_X.columns
        assert feature not in ethic_X.columns


def test_print_variables_by_scenario() -> None:
    """Affiche et vérifie la liste des variables de chaque scénario."""
    for scenario_name, scenario in SCENARIOS.items():
        if scenario["text_only"]:
            expected_features = [TEXT_FEATURE]
        else:
            expected_features = get_all_features(
                include_sensitive=scenario["include_sensitive"],
                include_text=scenario["include_text"],
                features=scenario.get("features"),
            )

        observed_features, _ = load_dataset(
            DATA_PATH,
            include_sensitive=scenario["include_sensitive"],
            include_text=scenario["include_text"],
            features=scenario.get("features"),
        )
        if scenario["text_only"]:
            observed_features = observed_features[[TEXT_FEATURE]]

        print(f"\n{scenario_name}")
        print(f"  Variables ({len(expected_features)}):")
        for feature in expected_features:
            print(f"    - {feature}")

        assert list(observed_features.columns) == expected_features
