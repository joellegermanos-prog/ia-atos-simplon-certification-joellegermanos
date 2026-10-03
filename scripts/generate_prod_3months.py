"""Generate a reproducible three-month CISIA production sample."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "dataset_trajectoire_emploi.csv"
OUTPUT = ROOT / "data" / "prod_3months.csv"
LOG_OUTPUT = ROOT / "data" / "predictions_log.csv"
SEED = 42
N_ROWS = 3000


def main() -> None:
    source = pd.read_csv(SOURCE, dtype={"code_insee_commune": str})
    rng = np.random.default_rng(SEED)
    sample = source.iloc[rng.integers(0, len(source), size=N_ROWS)].reset_index(drop=True)
    sample["usager_id"] = [f"PROD_{index:05d}" for index in range(N_ROWS)]

    sample["age"] = sample["age"].fillna(source["age"].median())
    sample["niveau_diplome"] = sample["niveau_diplome"].fillna(
        source["niveau_diplome"].mode().iat[0]
    )
    sample["est_allocataire"] = sample["est_allocataire"].fillna(
        source["est_allocataire"].mode().iat[0]
    )
    sample["synthese_entretien"] = sample["synthese_entretien"].fillna(
        "Information non renseignee"
    )

    # Simulate a gradual population shift during the observation window.
    progress = np.linspace(0.0, 1.0, N_ROWS)
    sample["age"] = (sample["age"] + 1.5 * progress).round(1)
    sample["anciennete_poste_ans"] = (
        sample["anciennete_poste_ans"] * (1.0 - 0.08 * progress)
    ).round(1)
    diploma_mask = rng.random(N_ROWS) < (0.05 + 0.10 * progress)
    sample.loc[diploma_mask, "niveau_diplome"] = "Bac+2"
    allocator_mask = rng.random(N_ROWS) < (0.03 + 0.07 * progress)
    sample.loc[allocator_mask, "est_allocataire"] = 1.0

    # Probabilities are scored outputs: the target is retained only for delayed
    # monitoring and calibration evaluation, not for real-time scoring.
    target = sample["classe_retour_emploi"].to_numpy(dtype=int)
    probabilities = np.full((N_ROWS, 3), 0.15, dtype=float)
    probabilities[np.arange(N_ROWS), target] = 0.70
    noise = rng.dirichlet(np.ones(3), size=N_ROWS) * 0.15
    probabilities = probabilities * 0.85 + noise
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    probabilities[:, 2] = 1.0 - probabilities[:, 0] - probabilities[:, 1]
    for class_index in range(3):
        sample[f"proba_{class_index}"] = probabilities[:, class_index]
    sample["prediction"] = probabilities.argmax(axis=1)
    sample["timestamp"] = pd.date_range(
        "2026-01-01", "2026-03-31 23:59:00", periods=N_ROWS
    )

    columns = [
        "usager_id",
        "timestamp",
        "age",
        "niveau_diplome",
        "anciennete_poste_ans",
        "code_rome_vise",
        "code_insee_commune",
        "est_allocataire",
        "nationalite_hors_ue",
        "synthese_entretien",
        "classe_retour_emploi",
        "prediction",
        "proba_0",
        "proba_1",
        "proba_2",
    ]
    sample[columns].to_csv(OUTPUT, index=False)
    log_columns = [
        "usager_id",
        "timestamp",
        "prediction",
        "proba_0",
        "proba_1",
        "proba_2",
        "classe_retour_emploi",
    ]
    prediction_log = sample[log_columns].rename(columns={"usager_id": "request_id"})
    prediction_log["proba_default"] = prediction_log["proba_2"]
    prediction_log["true_label"] = (
        prediction_log["classe_retour_emploi"] == 2
    ).astype(int)
    prediction_log.to_csv(LOG_OUTPUT, index=False)
    print(f"Wrote {len(sample)} rows to {OUTPUT}")
    print(f"Wrote {len(sample)} rows to {LOG_OUTPUT}")


if __name__ == "__main__":
    main()
