"""
Projet CISIA
Orientation et tri multimodal des demandeurs d'emploi.

Ce module construit un pipeline scikit-learn de préparation des données
pour une classification multiclasses du délai de retour à l'emploi.

Fonctionnalités :
    - chargement et validation du dataset ;
    - feature engineering du code INSEE ;
    - imputation des valeurs manquantes ;
    - standardisation des variables numériques ;
    - encodage ordinal du niveau de diplôme ;
    - encodage One-Hot des variables catégorielles ;
    - vectorisation TF-IDF des synthèses d'entretien ;
    - gestion d'un scénario avec ou sans variable sensible ;
    - sauvegarde du préprocesseur avec joblib.

La cible est :
    classe_retour_emploi
        0 : retour rapide, moins de 6 mois
        1 : retour moyen, entre 6 et 12 mois
        2 : risque de chômage longue durée, plus de 12 mois

Le préprocesseur est conçu pour supporter des modèles de référence
linéaires comme la régression logistique ainsi que des modèles
booster et arborescents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# =============================================================================
# 1. CONFIGURATION GÉNÉRALE
# =============================================================================

TARGET_COLUMN: str = "classe_retour_emploi"

EXPECTED_TARGET_VALUES: set[int] = {0, 1, 2}

IDENTIFIER_COLUMNS: list[str] = [
    "usager_id",
]

RAW_REQUIRED_COLUMNS: list[str] = [
    "usager_id",
    "age",
    "niveau_diplome",
    "anciennete_poste_ans",
    "code_rome_vise",
    "code_insee_commune",
    "est_allocataire",
    "nationalite_hors_ue",
    "synthese_entretien",
    TARGET_COLUMN,
]


# =============================================================================
# 2. DÉFINITION DES FEATURES
# =============================================================================

# Variables numériques continues.
NUMERIC_FEATURES: list[str] = [
    "age",
    "anciennete_poste_ans",
]

# Variable catégorielle ordonnée.
# L'ordre a une signification métier.
ORDINAL_FEATURES: dict[str, list[str]] = {
    "niveau_diplome": [
        "Sans diplôme",
        "Bac",
        "Bac+2",
        "Bac+5",
    ]
}

# Variables catégorielles nominales.
BASE_CATEGORICAL_FEATURES: list[str] = [
    "code_rome_vise",
    "departement",
    "est_allocataire",
]

# Variables sensibles à partir du diagnostic éthique du notebook.
# La nationalité et l'âge sont les variables sensibles principales.
# Le niveau de diplôme n'est pas une variable sensible au sens RGPD,
# et le département est un proxy géographique à traiter séparément.
SENSITIVE_FEATURES: list[str] = [
    "nationalite_hors_ue",
    "age",
]

# Proxy géographique, à considérer séparément de la sensibilité stricte.
PROXY_FEATURES: list[str] = [
    "departement",
    "niveau_diplome",
]

# Colonne textuelle.
TEXT_FEATURE: str = "synthese_entretien"


# Mots-outils français sans valeur métier.
# Les négations sont conservées volontairement.
FRENCH_STOP_WORDS = [
    "a",
    "afin",
    "ainsi",
    "au",
    "aux",
    "avec",
    "ce",
    "ces",
    "cet",
    "cette",
    "comme",
    "dans",
    "de",
    "des",
    "du",
    "elle",
    "elles",
    "en",
    "est",
    "et",
    "il",
    "ils",
    "je",
    "la",
    "le",
    "les",
    "leur",
    "leurs",
    "lui",
    "mais",
    "mes",
    "mon",
    "nous",
    "on",
    "ou",
    "par",
    "pour",
    "que",
    "qui",
    "sa",
    "se",
    "ses",
    "son",
    "sur",
    "tu",
    "un",
    "une",
    "vos",
    "votre",
    "vous",
]

# =============================================================================
# 3. TRANSFORMATEUR DE NETTOYAGE DU TEXTE
# =============================================================================

class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Nettoie une colonne textuelle avant la vectorisation TF-IDF.

    Le ColumnTransformer transmet une colonne seule sous la forme d'une
    Series Pandas. Ce transformateur garantit une sortie unidimensionnelle
    constituée uniquement de chaînes de caractères.

    Traitements appliqués :
        - remplacement des valeurs manquantes par une chaîne vide ;
        - conversion en chaîne de caractères ;
        - suppression des espaces en début et fin de texte ;
        - remplacement des chaînes vides par une valeur neutre.

    Le remplacement final par "texte_manquant" évite que TfidfVectorizer
    reçoive un corpus entièrement vide dans certains sous-échantillons.
    """

    def fit(
        self,
        X: Any,
        y: Any = None,
    ) -> "TextCleaner":
        return self

    def transform(self, X: Any) -> pd.Series:
        # Cas d'une Series transmise par ColumnTransformer.
        if isinstance(X, pd.Series):
            text_series = X.copy()

        # Cas d'un DataFrame à une colonne.
        elif isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError(
                    "TextCleaner attend exactement une colonne textuelle."
                )
            text_series = X.iloc[:, 0].copy()

        # Cas d'un tableau NumPy ou d'une liste.
        else:
            array = np.asarray(X)

            if array.ndim == 2:
                if array.shape[1] != 1:
                    raise ValueError(
                        "TextCleaner attend exactement une colonne textuelle."
                    )
                array = array.ravel()

            text_series = pd.Series(array)

        text_series = (
            text_series
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", "texte_manquant")
        )

        return text_series

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """Retourne un nom stable pour la colonne textuelle."""
        if input_features is None:
            return np.array([TEXT_FEATURE], dtype=object)

        return np.asarray(input_features, dtype=object)


# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

def extract_department(code_insee: Any) -> str:
    """Extrait le département depuis un code INSEE de commune."""
    if pd.isna(code_insee):
        return "INCONNU"

    code = str(code_insee).strip().upper()
    if code.endswith(".0"):
        code = code[:-2]
    if not code or code.lower() == "nan":
        return "INCONNU"

    if code.startswith("2A"):
        return "2A"
    if code.startswith("2B"):
        return "2B"

    if code.isdigit():
        code = code.zfill(5)

    return code[:2] if len(code) >= 2 else "INCONNU"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée les variables dérivées nécessaires au preprocessing.

    Feature créée :
        departement, extrait de code_insee_commune.

    La variable groupe_age n'est volontairement pas utilisée comme feature
    du modèle. Elle sert à l'audit éthique et au calcul du Disparate Impact.
    Le modèle conserve l'âge sous sa forme numérique continue.

    Parameters
    ----------
    df:
        DataFrame contenant les données brutes.

    Returns
    -------
    pd.DataFrame
        Copie du DataFrame enrichie.
    """
    df_features = df.copy()

    if "code_insee_commune" not in df_features.columns:
        raise KeyError(
            "La colonne 'code_insee_commune' est nécessaire "
            "pour créer la variable 'departement'."
        )

    df_features["departement"] = (
        df_features["code_insee_commune"]
        .apply(extract_department)
    )

    return df_features


# =============================================================================
# 5. VALIDATION DES DONNÉES
# =============================================================================

def validate_raw_columns(df: pd.DataFrame) -> None:
    """
    Vérifie la présence des colonnes attendues dans les données brutes.

    Raises
    ------
    KeyError
        Si une ou plusieurs colonnes nécessaires sont absentes.
    """
    missing_columns = [
        column
        for column in RAW_REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Colonnes attendues absentes du CSV : "
            f"{missing_columns}"
        )


def validate_target(y: pd.Series) -> pd.Series:
    """
    Valide et convertit la cible en entier.
    La cible doit uniquement contenir les classes 0, 1 et 2.

    Parameters
    ----------
    y:
        Série contenant la cible brute.

    Returns
    -------
    pd.Series
        Cible validée avec le type entier.

    Raises
    ------
    ValueError
        Si la cible contient des valeurs manquantes ou inconnues.
    """
    if y.isna().any():
        number_missing = int(y.isna().sum())

        raise ValueError(
            f"La cible contient {number_missing} valeur(s) manquante(s)."
        )

    try:
        validated_y = pd.to_numeric(y, errors="raise").astype(int)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "La cible doit contenir uniquement les entiers 0, 1 et 2."
        ) from error

    observed_values = set(validated_y.unique().tolist())
    unknown_values = observed_values - EXPECTED_TARGET_VALUES

    if unknown_values:
        raise ValueError(
            "Valeurs de cible non reconnues : "
            f"{sorted(unknown_values)}. "
            "Valeurs attendues : 0, 1 et 2."
        )

    return validated_y


# =============================================================================
# 6. SÉLECTION DES FEATURES
# =============================================================================

def get_categorical_features(
    include_sensitive: bool = True,
) -> list:
    """Retourne la liste des variables catégorielles du scénario demandé.

    Parameters
    ----------
    include_sensitive:
        True pour inclure les variables sensibles.
        False pour le scénario éthique sans variable sensible.

    Returns
    -------
    list[str]
        Liste des variables catégorielles.
    """

    categorical_features = [
        feature
        for feature in BASE_CATEGORICAL_FEATURES
        if include_sensitive or feature not in SENSITIVE_FEATURES
    ]

    if include_sensitive:
        categorical_features.extend(
            feature
            for feature in SENSITIVE_FEATURES
            if feature not in NUMERIC_FEATURES
            and feature not in ORDINAL_FEATURES
            and feature not in categorical_features
        )

    return categorical_features


def get_numeric_features(
    include_sensitive: bool = True,
) -> list[str]:
    """Retourne les variables numériques selon le scénario.

    Les variables sensibles sont définies centralement dans SENSITIVE_FEATURES.
    Si include_sensitive est False, toutes les colonnes sensibles sont retirées
    du bloc numérique avant le préprocesseur.
    """
    numeric_features = NUMERIC_FEATURES.copy()

    if not include_sensitive:
        numeric_features = [
            feature for feature in numeric_features
            if feature not in SENSITIVE_FEATURES
        ]

    return numeric_features


def get_ordinal_features(
    include_sensitive: bool = True,
) -> dict[str, list[str]]:
    """Retourne les variables ordinales selon le scénario.

    Si include_sensitive est False, les colonnes ordinales marquées comme
    sensibles (cf. SENSITIVE_FEATURES) sont retirées.
    """
    return {
        feature: categories
        for feature, categories in ORDINAL_FEATURES.items()
        if include_sensitive or feature not in SENSITIVE_FEATURES
    }


def get_all_features(
    include_sensitive: bool = True,
    include_text: bool = True,
    features: list[str] | None = None,
) -> list:
    """
    Retourne la liste complète des features utilisées.

    Parameters
    ----------
    include_sensitive:
        Inclure ou exclure les variables sensibles.

    include_text:
        Inclure ou exclure synthese_entretien.

    features:
        Liste explicite et manuelle des features à utiliser, en
        remplacement total du calcul par include_sensitive/include_text.
        Permet de choisir librement les variables d'un scénario.

    Returns
    -------
    list[str]
        Liste ordonnée des features.
    """
    if features is not None:
        return list(features)

    numeric_features = get_numeric_features(include_sensitive=include_sensitive)

    features = (
        numeric_features
        + list(get_ordinal_features(include_sensitive=include_sensitive).keys())
        + get_categorical_features(include_sensitive)
    )

    if include_text:
        features.append(TEXT_FEATURE)

    return features


# =============================================================================
# 7. CHARGEMENT DU DATASET
# =============================================================================

def load_dataset(
    path: Path,
    include_sensitive: bool = True,
    include_text: bool = True,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Charge le dataset CISIA et retourne X et y.

    Les codes ROME et INSEE sont lus comme des chaînes afin de préserver :
        - les zéros initiaux ;
        - les codes corses 2A et 2B ;
        - leur nature catégorielle.

    Les colonnes non utilisées, notamment usager_id et code_insee_commune
    après extraction du département, sont exclues de X.

    Parameters
    ----------
    path:
        Chemin du fichier CSV.

    include_sensitive:
        Inclure nationalite_hors_ue dans X.

    include_text:
        Inclure synthese_entretien dans X.

    features:
        Liste explicite et manuelle des features à utiliser (cf.
        get_all_features), en remplacement de include_sensitive/include_text.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        X, y.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path.resolve()}"
        )

    df = pd.read_csv(
        path,
        dtype={
            "usager_id": "string",
            "code_rome_vise": "string",
            "code_insee_commune": "string",
        },
    )

    validate_raw_columns(df)

    df = create_features(df)

    y = validate_target(df[TARGET_COLUMN])

    all_features = get_all_features(
        include_sensitive=include_sensitive,
        include_text=include_text,
        features=features,
    )

    missing_features = [
        feature
        for feature in all_features
        if feature not in df.columns
    ]

    if missing_features:
        raise KeyError(
            "Features attendues absentes après feature engineering : "
            f"{missing_features}"
        )

    X = df[all_features].copy()

    return X, y


# =============================================================================
# 8. CONSTRUCTION DU PREPROCESSOR
# =============================================================================

def build_numeric_pipeline() -> Pipeline:
    """
    Construit la branche numérique.

    Étapes :
        1. imputation par la médiane ;
        2. standardisation.

    L'imputation médiane est choisie car anciennete_poste_ans présente
    une asymétrie et des valeurs extrêmes.
    """
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )


def build_ordinal_pipeline() -> Pipeline:
    """
    Construit la branche ordinale pour niveau_diplome.

    Les modalités inconnues en production reçoivent la valeur -1.
    """
    ordinal_categories = [
        ORDINAL_FEATURES[column]
        for column in ORDINAL_FEATURES
    ]

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    dtype=np.float64,
                ),
            ),
        ]
    )


def build_categorical_pipeline() -> Pipeline:
    """
    Construit la branche des variables catégorielles nominales.

    Étapes :
        1. imputation par la modalité la plus fréquente ;
        2. encodage One-Hot.

    handle_unknown="ignore" évite une erreur lorsqu'une nouvelle modalité
    apparaît dans les données de test ou en production.
    """
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float64,
                ),
            ),
        ]
    )


def build_text_pipeline(
    max_features: int = 1_000,
    ngram_range: tuple[int, int] = (2, 3),
    min_df: int = 2,
) -> Pipeline:
    """
    Construit la branche textuelle TF-IDF.

    Parameters
    ----------
    max_features: Nombre maximal de termes conservés.
    ngram_range:(1, 2) conserve les unigrammes et bigrammes.
    min_df: Un terme doit apparaître dans au moins min_df documents.

    Returns
    -------
    Pipeline
        Pipeline de nettoyage et vectorisation du texte.
    """
    return Pipeline(
        steps=[
            (
                "cleaner",
                TextCleaner(),
            ),
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words=FRENCH_STOP_WORDS,
                    max_features=max_features,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    sublinear_tf=True,
                    dtype=np.float64,
                ),
            ),
        ]
    )


def build_preprocessor(
    include_sensitive: bool = True,
    include_text: bool = True,
    tfidf_max_features: int = 1_000,
    tfidf_ngram_range: tuple[int, int] = (2, 3),
    tfidf_min_df: int = 2,
    features: list[str] | None = None,
) -> ColumnTransformer:
    """
    Construit le préprocesseur final pour les variables numériques,
    ordinales, catégorielles et textuelles.

    Parameters
    ----------
    include_sensitive: Inclure la variable sensible nationalite_hors_ue.
    include_text: Inclure la branche TF-IDF pour synthese_entretien.
    tfidf_max_features: Nombre maximal de termes conservés dans la branche texte.
    tfidf_ngram_range: Plage des n-grammes pour la vectorisation TF-IDF.
    tfidf_min_df: Fréquence minimale d'apparition pour conserver un terme.
    features: Liste explicite et manuelle des features à utiliser, en
        remplacement de include_sensitive/include_text. Chaque branche
        (numérique, ordinale, catégorielle, texte) ne conserve que les
        colonnes présentes dans cette liste.

    Returns
    -------
    ColumnTransformer
        Préprocesseur scikit-learn prêt à être ajusté sur X.
    """
    if features is not None:
        numerical_features = [
            feature for feature in NUMERIC_FEATURES if feature in features
        ]
        ordinal_features = [
            feature for feature in ORDINAL_FEATURES if feature in features
        ]
        categorical_features = [
            feature
            for feature in features
            if feature not in numerical_features
            and feature not in ordinal_features
            and feature != TEXT_FEATURE
        ]
        include_text = TEXT_FEATURE in features
    else:
        numerical_features = get_numeric_features(include_sensitive=include_sensitive)
        ordinal_features = list(
            get_ordinal_features(include_sensitive=include_sensitive).keys()
        )
        categorical_features = get_categorical_features(include_sensitive=include_sensitive)

    transformers: list[tuple[str, Any, list[str]]] = [
        ("num", build_numeric_pipeline(), numerical_features),
        ("ord", build_ordinal_pipeline(), ordinal_features),
        ("cat", build_categorical_pipeline(), categorical_features),
    ]

    if include_text:
        transformers.append(
            (
                "text",
                build_text_pipeline(
                    max_features=tfidf_max_features,
                    ngram_range=tfidf_ngram_range,
                    min_df=tfidf_min_df,
                ),
                [TEXT_FEATURE],
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


if __name__ == "__main__":

    X, y = load_dataset(Path("data/dataset_trajectoire_emploi.csv"))
    preprocessor = (build_preprocessor())
    X_transformed = (preprocessor.fit_transform(X))
    print(f"X : {X.shape}")
    print(f"X transformé : {X_transformed.shape}")
    print("\n✅ Préprocessing OK")