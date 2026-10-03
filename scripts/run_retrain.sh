#!/usr/bin/env bash
set -euo pipefail

MIN_FEEDBACK="${1:-200}"
docker compose --profile retrain run --rm retrainer python scripts/retrain.py --min-feedback "${MIN_FEEDBACK}"
