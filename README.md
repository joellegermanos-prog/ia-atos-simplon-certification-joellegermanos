# ia-atos-simplon-certification

Ce projet met en place une chaîne complète de production pour un modèle d'aide à l'orientation des demandeurs d'emploi : entraînement, évaluation de qualité, API de scoring, feedback conseiller, observabilité, réentraînement et promotion contrôlée.

L'objectif est de sécuriser le cycle de vie du modèle depuis le développement jusqu'au déploiement opérationnel, avec un contrôle explicite sur la qualité, les dérives de données et la robustesse de l'architecture.

## 1. Vue d'ensemble

Le dépôt couvre :

- la préparation et l'entraînement de modèles multimodaux et tabulaires ;
- la validation des performances sur un holdout de référence ;
- l'exposition d'une API de prédiction ;
- l'orchestration backend, le stockage des prédictions et des feedbacks ;
- la surveillance avec Prometheus et Grafana ;
- la détection de dérive des données ;
- le réentraînement contrôlé à partir des annotations conseillers ;
- la promotion d'un candidat seulement si les seuils de qualité sont respectés.

## 2. Architecture

```text
Navigateur
    |
    v
Frontend :8088
    |
    v
Backend FastAPI :8001
    |
    +--> Model FastAPI :8000
    |
    +--> SQLite feedback_data
    |
    +--> Prometheus :9090
                 |
                 v
               Grafana :3001
```

Les responsabilités sont séparées :

- le service model charge l'artefact et retourne une prédiction ;
- le backend valide les requêtes, gère le request_id et stocke les historiques ;
- le frontend est une interface simple pour le scoring et les feedbacks ;
- Prometheus et Grafana assurent la supervision système et métier ;
- les scripts d'évaluation et de drift sécurisent la décision de promotion ou de rejet.

## 3. Structure du dépôt

```text
certification/
├── data/                     # jeux de données, référence, drift et données de production
├── grafana/                 # configuration et dashboards Grafana
├── models/                  # artefacts de modèles et métadonnées
├── notebooks/               # analyses exploratoires et restitution
├── prometheus/              # configuration Prometheus
├── reports/                 # évaluations, rapports et exports
├── scripts/                 # drift, promotion, retraining, evaluation et utilitaires
├── services/
│   ├── model/               # API de prédiction et métriques
│   ├── backend/             # orchestration scoring et feedbacks
│   ├── frontend/            # interface web / Nginx
│   └── retrainer/           # service optionnel activé via profile retrain
├── src/                     # code de training, evaluation et preprocessing
├── tests/                   # tests fonctionnels et de qualité
├── docker-compose.yml       # orchestration de la stack locale
├── pytest.ini               # configuration pytest
├── requirements-dev.txt     # dépendances de dev / test
├── retrain-result.json      # résultat de retraining récent
├── drift_summary.md         # synthèse de dérive
├── README.md                # ce document
└── mlruns/                  # tracking MLflow local
```

## 4. Prérequis

- Python 3.11+
- Docker Desktop / Docker Engine
- Docker Compose
- PowerShell (pour les scripts Windows) ou bash/zsh

Vérification rapide :

```powershell
python --version
docker --version
docker compose version
```

## 5. Installation locale

Depuis la racine du dossier `certification` :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Si le projet nécessite un modèle spécifique ou un package complémentaire :

```powershell
python -m pip install lightgbm
```

## 6. Démarrage rapide

### 6.1 Démarrer la stack locale

```powershell
docker compose up -d --build
docker compose ps
```

### 6.2 Vérifier les services

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8001/health
Invoke-WebRequest http://localhost:8088
```

### 6.3 Vérifier les endpoints

| Service | URL | Description |
|---|---|---|
| Model | http://localhost:8000/docs | API de prédiction, santé et métriques |
| Backend | http://localhost:8001/docs | score, feedback et historique |
| Frontend | http://localhost:8088 | interface du conseiller |
| Prometheus | http://localhost:9090 | monitoring des services |
| Grafana | http://localhost:3001 | dashboard observabilité |
| MLflow | http://localhost:5000 | tracking des runs |

## 7. Tests et validation

### Tests Python

```powershell
python -m pytest -q
```

### Vérification du code

```powershell
python -m py_compile scripts/retrain.py scripts/promotion.py
python -m py_compile src/calibration.py src/recommendations.py
```

### Vérification de la configuration Docker

```powershell
docker compose config
```

Cette étape permet de détecter rapidement une erreur de YAML, une variable manquante ou un service mal défini avant d'exécuter le stack complet.

## 8. Évaluation et quality gate

Le quality gate repose sur un jeu de référence figé. Le but est de vérifier qu'un modèle candidat ou un artefact mis en production ne dégrade pas la qualité métier.

Exemple de commande :

```powershell
python scripts/evaluate_model.py --release-tag test
python scripts/evaluate_model.py --release-tag degraded --degrade
```

Le comportement attendu :

- release conforme → code de sortie 0 ;
- version dégradée ou non conforme → code de sortie non nul ;
- les écarts doivent être documentés et justifiés dans les seuils de promotion.

## 9. Dérive de données et calibration

Le projet inclut des scripts de diagnostic pour la dérive et la calibration :

```powershell
python scripts/drift_analysis.py `
  --reference data/reference_set.csv `
  --current data/current_scored.csv `
  --output reports/drift.json `
  --prometheus-output reports/drift.prom
```

Cela permet de mesurer :

- PSI sur les variables et probabilités ;
- tests KS / Chi² ;
- dérive des probabilités et des features ;
- dégradation éventuelle de calibration ;
- signal de concept drift ou data drift.

## 10. Feedback et retraining

Le pipeline de feedback est central pour le cycle continu d'amélioration du modèle :

1. un score est généré et associé à un request_id ;
2. le conseiller valide ou corrige la classe réelle ;
3. le feedback est stocké en base SQLite ;
4. le retrainer détecte le seuil de nouveaux feedbacks non consommés ;
5. un candidat est entraîné, évalué et comparé au modèle de production ;
6. la décision de promotion ou de rejet est journalisée ;
7. un artefact n'est déployé que si les règles de qualité sont satisfaites.

Exemple :

```powershell
docker compose --profile retrain run --rm retrainer
```

Ou, sous Windows :

```powershell
./scripts/run_retrain.ps1 -MinFeedback 200
```

## 11. Promotion et déploiement

Le système est conçu pour éviter de promouvoir un modèle sans seuil de qualité. Les décisions sont automatisées par des scripts de promotion et de déploiement :

```powershell
./scripts/deploy_promoted.ps1 -Build
```

Le modèle promu remplace l'artefact courant ou, suivant la configuration, le service est recréé pour charger le nouveau modèle. Cela permet de supporter une version stable et un rollback contrôlé.

## 12. Observabilité

Le dépôt fournit une supervision par défaut :

- Prometheus sur le port 9090 ;
- Grafana sur le port 3001 ;
- dashboards pour la santé, la vitesse et le comportement du système ;
- métriques de service et de modèle, incluant les erreurs backend et les métriques métier.

Vérifications utiles :

```powershell
Invoke-WebRequest http://localhost:9090/-/healthy
Invoke-WebRequest http://localhost:3001/api/health
```

## 13. Commandes utiles

### Démarrer complètement la stack

```powershell
docker compose up -d --build
```

### Arrêter la stack

```powershell
docker compose down
```

### Redémarrer un service

```powershell
docker compose restart model backend frontend
```

### Consulter les logs

```powershell
docker compose logs -f backend
docker compose logs -f model
```

### Vérifier le statut des conteneurs

```powershell
docker compose ps
```

## 14. Bonnes pratiques

- garder les jeux de référence figés et versionnés ;
- ne pas promouvoir un modèle sur la base d'un seul signal ;
- tenir compte de la qualité des labels avant de conclure à un concept drift ;
- utiliser les données de production pour la surveillance, pas pour l'entraînement sans garde-fou ;
- vérifier le contrat métier et le contrat technique avant toute publication.

## 15. Résumé

Ce projet illustre un MLOps réaliste pour un système d'IA de décision : entraînement fiable, contrôle qualité, API de service, supervision opérationnelle, feedback de terrain et boucle de réentraînement. L'enjeu central n'est pas seulement la performance du modèle, mais aussi sa robustesse, son traçabilité et la sécurité des décisions de promotion.

Pour aller plus loin, il faut combiner :

- l'évaluation de référence ;
- les métriques de production ;
- la calibration et la dérive ;
- la politique de promotion ;
- le runbook d'astreinte.