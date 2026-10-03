"""
Construction du jeu de référence (Reference Set) à partir du holdout.

Le fichier généré servira pour :
- le Golden Run
- l'évaluation continue
- les comparaisons futures de modèles

Sortie :
    data/reference_set.csv
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent

DATASET_PATH = ROOT / "data" / "dataset_trajectoire_emploi.csv"
METADATA_PATH = (
    ROOT
    / "services"
    / "model"
    / "models"
    / "cisia_emploi_xgboost_multimodal_ethique_best_class_2_ethique.json"
)

OUTPUT_PATH = ROOT / "data" / "reference_set.csv"


def main():

    print("Chargement du dataset...")
    df = pd.read_csv(DATASET_PATH)

    print("Chargement des métadonnées du modèle...")
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    test_indices = metadata["dataset"]["test_indices"]

    print(f"Nombre d'indices holdout : {len(test_indices)}")

    reference_df = df.loc[test_indices].copy()

    reference_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print("\nReference Set créé avec succès")
    print(f"Fichier : {OUTPUT_PATH}")
    print(f"Nombre de lignes : {len(reference_df)}")

    print("\nRépartition des classes :")
    print(
        reference_df["classe_retour_emploi"]
        .value_counts(normalize=True)
        .sort_index()
    )


if __name__ == "__main__":
    main()