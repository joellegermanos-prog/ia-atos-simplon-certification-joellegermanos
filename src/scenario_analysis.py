"""Analyse comparative des scénarios CISIA.

Construit le tableau signature de soutenance à partir des résultats
produits par evaluate.py.

Usage depuis la racine du projet :
    python src/scenario_analysis.py --config balanced
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import REPORTS_DIR


SCENARIO_INFORMATION: dict[str, dict[str, str]] = {
    "multimodal_complet": {
        "label": "S1 - Multimodal complet",
        "explicabilite": "Moyenne",
        "dependance": "Aucune",
        "biais": (
            "Age identifie comme variable sensible en section 1.5. "
            "Risque mesure en section 3.6, non mitige par retrait. "
            "Le departement reste un proxy geographique potentiel."
        ),
        "verdict": (
            "Performance elevee, mais compromis ethique moins favorable."
        ),
    },
    "multimodal_ethique": {
        "label": "S2 - Multimodal ethique",
        "explicabilite": "Moyenne",
        "dependance": "Aucune",
        "biais": (
            "Age retire : biais direct mitige. "
            "Le departement reste surveille comme proxy geographique potentiel."
        ),
        "verdict": (
            "Meilleur compromis entre performance, robustesse et ethique."
        ),
    },
    "texte_seul": {
        "label": "S3 - Texte seul",
        "explicabilite": "Faible a moyenne",
        "dependance": "Aucune",
        "biais": (
            "Variables sensibles absentes. Risque residuel lie a la qualite "
            "et au vocabulaire des syntheses d'entretien."
        ),
        "verdict": (
            "Solution sobre, mais moins robuste si le texte est incomplet."
        ),
    },
    "tabulaire_seul": {
        "label": "S4 - Tabulaire seul",
        "explicabilite": "Moyenne a elevee",
        "dependance": "Aucune",
        "biais": (
            "Age conserve : risque direct residuel. "
            "Le departement reste un proxy geographique potentiel."
        ),
        "verdict": (
            "Rapide et simple, mais moins favorable sur le plan ethique."
        ),
    },
}


TABLE_COLUMNS = [
    "Scenario",
    "Modele",
    "Metrique principale",
    "Metrique secondaire",
    "Cout inference",
    "Latence p95",
    "Explicabilite",
    "Dependance fournisseur",
    "Biais et mitigation",
    "Verdict",
    "Empreinte estimee",
]


def load_benchmark(reports_dir: Path, config_name: str) -> pd.DataFrame:
    """Charge le benchmark et ajoute le rappel de la classe 2."""
    benchmark_path = reports_dir / f"benchmark_all_all_{config_name}.csv"

    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Benchmark introuvable : {benchmark_path}. "
            "Lance d'abord evaluate.py."
        )

    benchmark = pd.read_csv(benchmark_path)
    recall_rows: list[dict[str, Any]] = []

    for evaluation_path in reports_dir.glob(
        f"cisia_emploi_*_{config_name}_evaluation.json"
    ):
        with evaluation_path.open(encoding="utf-8") as file:
            evaluation = json.load(file)

        report = evaluation["metrics"]["classification_report"]
        class_2_metrics = next(
            values
            for label, values in report.items()
            if label.lower().startswith("risque longue")
        )

        recall_rows.append(
            {
                "model_type": evaluation["model_type"],
                "scenario": evaluation["scenario_name"],
                "config": evaluation["config_name"],
                "recall_classe_2": class_2_metrics["recall"],
            }
        )

    recall_df = pd.DataFrame(recall_rows)
    if recall_df.empty:
        raise ValueError("Aucun fichier JSON d'evaluation n'a ete trouve.")

    return benchmark.merge(
        recall_df,
        on=["model_type", "scenario", "config"],
        how="left",
        validate="one_to_one",
    )


def build_signature_table(
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Selectionne le meilleur modele de chaque scenario et construit le tableau."""
    unknown_scenarios = set(benchmark["scenario"]) - set(SCENARIO_INFORMATION)
    if unknown_scenarios:
        raise ValueError(f"Scenarios sans analyse metier : {sorted(unknown_scenarios)}")

    best_by_scenario = (
        benchmark
        .sort_values(
            ["scenario", "f1_macro", "recall_classe_2"],
            ascending=[True, False, False],
        )
        .groupby("scenario", as_index=False)
        .first()
    )

    if "cpu_ms_1000_predictions" not in best_by_scenario:
        legacy_cpu_column = "cpu_seconds_1000_predictions"
        if legacy_cpu_column not in best_by_scenario:
            raise KeyError("Benchmark missing CPU cost per 1,000 predictions.")
        best_by_scenario["cpu_ms_1000_predictions"] = (
            best_by_scenario[legacy_cpu_column] * 1_000
        )

    table = pd.DataFrame(
        {
            "Scenario": best_by_scenario["scenario"].map(
                lambda value: SCENARIO_INFORMATION[value]["label"]
            ),
            "Modele": best_by_scenario["model_type"],
            "Metrique principale": best_by_scenario["f1_macro"].map(
                lambda value: f"F1 macro holdout = {value:.3f}"
            ),
            "Metrique secondaire": best_by_scenario["recall_classe_2"].map(
                lambda value: f"Rappel classe 2 = {value:.3f}"
            ),
            "Cout inference": best_by_scenario[
                "cpu_ms_1000_predictions"
            ].map(
                lambda value: f"{value:.3f} CPU.ms / 1 000 predictions"
            ),
            "Latence p95": best_by_scenario["latency_p95_ms"].map(
                lambda value: f"{value:.3f} ms"
            ),
            "Explicabilite": best_by_scenario["scenario"].map(
                lambda value: SCENARIO_INFORMATION[value]["explicabilite"]
            ),
            "Dependance fournisseur": best_by_scenario["scenario"].map(
                lambda value: SCENARIO_INFORMATION[value]["dependance"]
            ),
            "Biais et mitigation": best_by_scenario["scenario"].map(
                lambda value: SCENARIO_INFORMATION[value]["biais"]
            ),
            "Verdict": best_by_scenario["scenario"].map(
                lambda value: SCENARIO_INFORMATION[value]["verdict"]
            ),
            "Empreinte estimee": (
                "Non mesuree ; a estimer avec Scaphandre ou Green Algorithms"
            ),
        }
    )

    return table[TABLE_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construit le tableau signature des scenarios CISIA."
    )
    parser.add_argument("--config", default="balanced")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    benchmark = load_benchmark(args.reports_dir, args.config)
    signature_table = build_signature_table(benchmark)
    output_path = args.reports_dir / "tableau_signature_scenarios.csv"
    signature_table.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("TABLEAU SIGNATURE - ANALYSE DES SCENARIOS")
    print("=" * 120)
    print(signature_table.to_string(index=False))
    print("=" * 120)
    print(f"Fichier sauvegarde : {output_path.resolve()}")


if __name__ == "__main__":
    main()
