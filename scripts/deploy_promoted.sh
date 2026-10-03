#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f services/model/models/cisia_emploi_promoted.joblib || \
      ! -f services/model/models/cisia_emploi_promoted.json ]]; then
  echo "Promoted model artifacts are missing. Run the retrainer and pass its promotion gate first." >&2
  exit 1
fi

export MODEL_ARTIFACT=/app/models/cisia_emploi_promoted.joblib
docker compose up -d --force-recreate model backend frontend
docker compose ps model backend frontend
