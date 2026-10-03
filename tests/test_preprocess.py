from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocess import build_preprocessor, load_dataset

DATA_PATH = PROJECT_ROOT / "data" / "dataset_trajectoire_emploi.csv"
TARGET_COLUMN = "classe_retour_emploi"


@pytest.fixture
def prepared_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Charge les données brutes et les features utilisées par le pipeline."""
    raw_df = pd.read_csv(DATA_PATH)
    X, y = load_dataset(
        DATA_PATH,
        include_sensitive=True,
        include_text=True,
    )
    return raw_df, X, y


@pytest.fixture
def transformed_data(
    prepared_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series],
) -> tuple[pd.DataFrame, object]:
    """Ajuste le préprocesseur et retourne les données transformées."""
    _, X, _ = prepared_data
    preprocessor = build_preprocessor(
        include_sensitive=True,
        include_text=True,
    )
    return X, preprocessor.fit_transform(X)


def test_dataset_is_not_empty(prepared_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series]) -> None:
    raw_df, _, _ = prepared_data
    print(f"Dataset complet non vide : {raw_df.shape[0]} lignes")
    assert raw_df.shape[0] > 0, "Dataset vide après chargement"


def test_target_has_no_missing_values(prepared_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series]) -> None:
    _, _, y = prepared_data
    print(f"Valeurs manquantes dans la cible : {y.isna().sum()}")
    assert y.isna().sum() == 0, "Manquants résiduels dans la cible"


def test_dataset_has_no_duplicates(prepared_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series]) -> None:
    raw_df, _, _ = prepared_data
    print(f"Doublons dans le dataset : {raw_df.duplicated().sum()}")
    assert raw_df.duplicated().sum() == 0, "Doublons résiduels dans le dataset"


def test_target_contains_expected_classes(prepared_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series]) -> None:
    _, _, y = prepared_data
    observed_classes = sorted(y.unique().tolist())
    print(f"Classes observées dans {TARGET_COLUMN} : {observed_classes}")
    assert set(y.unique()).issubset({0, 1, 2}), (
        f"Classes de cible inattendues dans {TARGET_COLUMN}: {observed_classes}"
    )


def test_preprocessing_preserves_rows(
    prepared_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series],
) -> None:
    _, X, y = prepared_data
    X_train, X_test, y_train, _y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(
        include_sensitive=True,
        include_text=True,
    )
    X_train_transformed = preprocessor.fit_transform(X_train, y_train)
    X_test_transformed = preprocessor.transform(X_test)

    expected_test_rows = int(len(X) * 0.2)
    expected_train_rows = len(X) - expected_test_rows

    print(
        "Répartition attendue/réelle : "
        f"train {expected_train_rows}/{X_train.shape[0]}, "
        f"test {expected_test_rows}/{X_test.shape[0]}"
    )

    assert X_train.shape[0] == expected_train_rows, (
        "Le nombre de lignes du train est incorrect"
    )
    assert X_test.shape[0] == expected_test_rows, (
        "Le nombre de lignes du test est incorrect"
    )

    print(
        "Lignes train avant/après preprocessing : "
        f"{X_train.shape[0]} / {X_train_transformed.shape[0]}"
    )
    print(
        "Lignes test avant/après preprocessing : "
        f"{X_test.shape[0]} / {X_test_transformed.shape[0]}"
    )

    assert X_train_transformed.shape[0] == X_train.shape[0], (
        "Le nombre de lignes du train a changé pendant le preprocessing"
    )
    assert X_test_transformed.shape[0] == X_test.shape[0], (
        "Le nombre de lignes du test a changé pendant le preprocessing"
    )
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0], (
        "Le split train/test est incomplet"
    )


def test_preprocessing_produces_features(
    transformed_data: tuple[pd.DataFrame, object],
) -> None:
    X, X_transformed = transformed_data
    print(f"Features produites : {X.shape[1]} -> {X_transformed.shape[1]}")
    assert X_transformed.shape[1] > 0, "Aucune feature produite après preprocessing"