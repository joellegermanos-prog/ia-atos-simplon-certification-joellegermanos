"""Évaluation continue + tracking MLflow (SQUELETTE M5-B2 À COMPLÉTER).

À chaque release : recalcule les métriques cibles sur un jeu de référence
figé, **trace le run dans MLflow**, compare aux seuils, et **sort un code
retour non-zéro** si dégradation (→ bloque la release en CI).

Renommez ce fichier en `scripts/evaluate_model.py` une fois complété.
Mini-cours : `07_MLflow_tracking_essentiel.md` + `08_Evaluation_continue_seuils`.

Usage cible::

    python scripts/evaluate_model.py --freeze-baseline             # une fois, au gel du jeu
    python scripts/evaluate_model.py --release-tag v2.0.0
    python scripts/evaluate_model.py --release-tag bad --degrade   # test du rouge
    mlflow ui    # comparer les runs

⚠️ **Le piège central du brief.** La tentation est de comparer vos métriques à
la baseline holdout annoncée en M1 (`metrics_holdout` dans le `.json`). Ne le
faites pas : le holdout et votre jeu de référence n'ont ni la même taille ni la
même composition. Vous mesureriez l'écart entre **deux populations**, pas la
dégradation du **modèle** — et votre garde-fou se déclencherait tout seul.
La baseline du garde-fou, c'est le **golden run** : les métriques mesurées sur
**votre** jeu de référence, au moment où vous le gelez.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import mlflow
import pandas as pd

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "services" / "model" / "models"
REFERENCE_SET = ROOT / "data" / "reference_set.csv"
REFERENCE_BASELINE = ROOT / "data" / "reference_baseline.json"

# TODO 1 — définir vos seuils (stratégie absolu / relatif / hybride).
#   Documentez-les ET justifiez-les dans evaluation_thresholds.md.
#   ⚠️ Une tolérance relative n'a de sens que si elle est **plus grande que le
#   bruit de mesure** de votre jeu de référence. Mesurez ce bruit (bootstrap,
#   cf. mini-cours 08) et prenez au moins 2 σ. Sous le bruit, le garde-fou se
#   déclenche sur du hasard et vous perdez confiance en lui.
THRESHOLDS: dict[str, dict[str, float]] = {
    # "f1_macro": {"absolute_min": ..., "max_drop_vs_baseline": ...},
}


def compute_metrics(model, df: pd.DataFrame, meta: dict) -> dict[str, float]:
    """Calcule les métriques cibles sur le jeu de référence."""
    # TODO 2 — construire X (feature_columns_*) et y (target + target_mapping),
    #   prédire, et calculer f1_macro / f1_default / roc_auc / recall_default.
    raise NotImplementedError


def check_thresholds(metrics: dict[str, float], baseline: dict) -> list[str]:
    """Retourne la liste des violations de seuil (vide = release OK)."""
    # TODO 3 — comparer chaque métrique à son plancher absolu ET à la baisse
    #   max tolérée vs baseline. Retourner les messages de violation.
    raise NotImplementedError


def load_baseline() -> dict:
    """Charge le golden run (baseline mesurée sur le jeu de référence)."""
    # TODO 3bis — lire REFERENCE_BASELINE et renvoyer ses métriques.
    #   Si le fichier n'existe pas : lever une erreur explicite qui dit de
    #   lancer `--freeze-baseline`. Surtout **pas** de repli silencieux sur
    #   `meta["metrics_holdout"]` : ce serait comparer deux populations.
    raise NotImplementedError


def freeze_baseline(model, df: pd.DataFrame, meta: dict) -> dict:
    """Mesure et gèle le golden run sur le jeu de référence."""
    # TODO 3ter — calculer les métriques sur le jeu de référence et les écrire
    #   dans REFERENCE_BASELINE (avec model_version, reference_set,
    #   n_reference). Ce fichier est **versionné** : c'est lui qui arbitre les
    #   releases. À regeler seulement si le jeu OU le modèle de référence change.
    raise NotImplementedError


def load_reference_set() -> pd.DataFrame:
    """Charge le jeu de référence, avec un garde-fou sur sa validité.

    Le jeu de référence est VOTRE instrument de mesure : vous le construisez
    à partir du holdout M1 (cf. `data/README.md`). Le fichier
    `reference_set_TEMPLATE.csv` livré dans le repo est un **exemple de
    format** de 20 lignes, pas un jeu de référence utilisable.
    """
    if not REFERENCE_SET.exists():
        raise SystemExit(
            f"{REFERENCE_SET} est absent.\n"
            "Ce fichier n'est pas fourni : c'est à vous de le construire à "
            "partir du holdout M1 (`data/lending_club_holdout.csv`).\n"
            "Mode d'emploi : data/README.md — étape 0."
        )
    df = pd.read_csv(REFERENCE_SET)
    if len(df) < 100 or df.iloc[:, -1].nunique() < 2:
        raise SystemExit(
            f"{REFERENCE_SET} contient {len(df)} ligne(s) et "
            f"{df.iloc[:, -1].nunique()} classe(s) de cible.\n"
            "Un instrument de mesure a besoin des DEUX classes et d'assez "
            "d'observations de la classe rare (~500 lignes attendues).\n"
            "Avez-vous copié reference_set_TEMPLATE.csv ? C'est un exemple de "
            "format, pas un jeu de référence — cf. data/README.md."
        )
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-tag", default="dev")
    parser.add_argument("--degrade", action="store_true")
    parser.add_argument("--freeze-baseline", action="store_true")
    args = parser.parse_args()

    model = joblib.load(MODELS_DIR / "cisia_emploi_xgboost_multimodal_ethique_best_class_2_ethique.joblib")
    meta = json.loads((MODELS_DIR / "cisia_emploi_xgboost_multimodal_ethique_best_class_2_ethique.json").read_text(encoding="utf-8"))
    df = load_reference_set()

    if args.freeze_baseline:
        print(json.dumps(freeze_baseline(model, df, meta), indent=2))
        return 0

    if args.degrade:
        # TODO 4 — simuler un bug de preprocessing réaliste (ex. désaligner
        #   X et y) pour PROUVER que le rouge bloque bien la release.
        pass

    metrics = compute_metrics(model, df, meta)
    baseline = load_baseline()  # ← le golden run, PAS metrics_holdout
    violations = check_thresholds(metrics, baseline)

    # --- Bloc MLflow PRÉ-CÂBLÉ — complétez params + metrics ------------------
    mlflow.set_experiment("cisia-emploi-eval-continue")
    with mlflow.start_run(run_name=args.release_tag):
        mlflow.log_params(
            {
                "model_version": meta["model_version"],
                "release_tag": args.release_tag,
                # TODO 5 — ajouter reference_set, n_reference…
            }
        )
        mlflow.log_metrics(metrics)  # ← les 4 métriques tracées
        mlflow.set_tag("release_blocked", str(bool(violations)))
    # ------------------------------------------------------------------------

    print(json.dumps({"metrics": metrics, "violations": violations}, indent=2))
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
