# CISIA Certification

Projet d'aide à l'orientation des demandeurs d'emploi. Le dépôt couvre le cycle de vie du modèle, de la préparation des données et l'évaluation jusqu'au service API, au suivi opérationnel et au réentraînement sur feedbacks.

## Parcours du modèle

1. `src/train.py` entraîne un pipeline sur le train split et conserve un holdout de référence.
2. `src/evaluate.py` valide le hash du dataset, reconstruit le même holdout et mesure les performances.
3. Les résultats et artefacts sont enregistrés dans `models/` et `reports/`.
4. `services/model` sert les prédictions; `services/backend` orchestre le scoring et les feedbacks; `services/frontend` fournit l'interface conseiller.
5. Le retrainer optionnel construit et évalue un candidat à partir des feedbacks éligibles avant toute promotion.

Les quatre scénarios sont `multimodal_complet`, `multimodal_ethique`, `texte_seul` et `tabulaire_seul`. Les algorithmes disponibles sont Random Forest, régression logistique, LightGBM et XGBoost.

## Structure

```text
certification/
├── data/                 # Dataset, jeu de référence et données d'audit
├── models/               # Artefacts et métadonnées du pipeline de recherche
├── notebooks/            # Analyse et restitution du projet
├── reports/              # Évaluations, benchmarks, décisions et métriques batch
├── src/                  # Prétraitement, entraînement, évaluation et analyse
├── scripts/              # Quality gate, drift, promotion et retraining
├── services/
│   ├── model/             # API modèle FastAPI
│   ├── backend/           # API scoring, feedback et historique
│   ├── frontend/          # Interface conseiller
│   └── retrainer/         # Image du retrainer Compose
├── prometheus/            # Configuration de collecte
├── grafana/               # Provisioning et tableaux de bord
├── .github/workflows/     # CI/CD et workflow de réentraînement
├── tests/                 # Tests du pipeline et de l'évaluation
├── tests-archive/         # Tests historiques
├── docker-compose.yml
├── requirements-dev.txt
└── README.md
```

## Installation locale

Python 3.11 ou plus récent est utilisé par la CI. Depuis la racine de `certification` :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

LightGBM est optionnel dans le pipeline Python; pour entraîner ou évaluer ce modèle, l'installer en plus :

```powershell
python -m pip install lightgbm
```

## Entraîner et évaluer

L'entraînement utilise le dataset défini dans `src/config.py`, fait une validation croisée uniquement sur le train split, entraîne le pipeline final et sauvegarde le modèle et ses métadonnées dans `models/`.

```powershell
python src/train.py --model-type xgboost --scenario multimodal_complet --config balanced
```

Pour entraîner toutes les combinaisons disponibles :

```powershell
python src/train.py --model-type all --scenario all --config balanced
```

L'évaluation vérifie le dataset, recharge le holdout via les indices stockés dans les métadonnées, puis produit les métriques et artefacts associés.

```powershell
python src/evaluate.py --model-type all --scenario all --config balanced --skip-missing
```

Utiliser `--help` pour afficher les options disponibles. Le holdout n'est pas utilisé pour la validation croisée ni pour l'entraînement.

## Résultats

Les modèles de recherche et leurs JSON de provenance sont écrits dans `models/`. Les évaluations produisent dans `reports/` un JSON par combinaison, une matrice de confusion, un rapport de classification, un fichier de prédictions et, lorsque plusieurs modèles sont évalués, un benchmark CSV et son graphique.

Le benchmark distingue `cpu_mean_ms` (temps CPU moyen par prédiction, en ms) de `latency_p95_ms` (latence murale au 95e percentile). Le premier mesure le temps processeur consommé; le second mesure le temps écoulé observé par l'appelant.

## Dérive des données

Le diagnostic compare un jeu de référence et un jeu courant de même schéma. Il calcule des indicateurs de dérive de données et, si les colonnes correspondantes sont présentes, de dérive de cible et de confiance.

```powershell
python scripts/drift_analysis.py `
	--reference data/reference_set.csv `
	--current data/current_scored.csv `
	--output reports/drift.json `
	--prometheus-output reports/drift.prom
```

`--prometheus-output` est facultatif. Le rapport JSON conserve les détails; le fichier Prometheus permet au backend et à Grafana d'exposer le dernier diagnostic.

## Services locaux

La stack complète se lance avec Docker Compose :

```powershell
docker compose up --build -d
docker compose ps
```

| Composant | URL locale | Rôle |
|---|---|---|
| Frontend | http://localhost:8088 | Interface conseiller |
| Backend API | http://localhost:8001/docs | Scoring, feedbacks et historique |
| Model API | http://localhost:8000/docs | Prédiction, informations du modèle et métriques |
| MLflow | http://localhost:5000 | Suivi des runs et artefacts |
| Prometheus | http://localhost:9090 | Collecte des métriques |
| Grafana | http://localhost:3001 | Visualisation et dashboards |

Le démarrage requiert l'artefact configuré par `MODEL_ARTIFACT` dans `services/model/models/`. Les valeurs par défaut Compose pointent vers le modèle multimodal éthique livré dans ce dossier. Les endpoints `/health` permettent de vérifier l'état des services.

## Feedback et réentraînement

Le backend associe chaque feedback à un `request_id` de prédiction, valide sa cohérence et le conserve dans le volume Docker `feedback_data`. Le retrainer est un profil optionnel; par défaut, il exige 200 feedbacks non consommés.

```powershell
docker compose --profile retrain run --rm retrainer python scripts/retrain.py --min-feedback 200
```

Sous Windows, `scripts/run_retrain.ps1 -MinFeedback 200` lance la même commande. Le candidat est évalué sur le jeu de référence et n'est pas activé si les règles de promotion échouent. Le workflow GitHub Actions `retrain.yml` peut être lancé manuellement et est planifié mensuellement; il nécessite son runner et sa configuration de déploiement.

## Quality gate et CI/CD

Le jeu de référence gelé (`data/reference_set.csv`) sert à comparer une release au golden baseline, pas à entraîner le modèle. Après avoir gelé la baseline une première fois, le quality gate s'exécute ainsi :

```powershell
python scripts/evaluate_model.py --release-tag local
```

`--degrade` permet de vérifier le chemin de rejet; `--freeze-baseline` sert uniquement à créer ou renouveler explicitement la baseline de référence.

Le workflow `ci.yml` exécute Ruff, les tests API et d'évaluation, puis le quality gate. Les images des services sont construites pour les PR; leur publication GHCR est réservée aux pushes sur `main` et aux tags `v*`. Le déploiement sur l'hôte cible ne s'exécute que sur `main` et si les paramètres de déploiement sont configurés.

## Tests

Depuis la racine du dépôt :

```powershell
python -m pytest -q
```

Pour le lint utilisé en CI :

```powershell
ruff check services scripts tests
```

`Usefultesting.md` décrit les vérifications manuelles Docker, API, monitoring, feedback, retraining, promotion et rollback.
