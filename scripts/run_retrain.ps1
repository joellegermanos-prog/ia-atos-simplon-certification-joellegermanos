param(
    [int]$MinFeedback = 200
)

$ErrorActionPreference = "Stop"
docker compose --profile retrain run --rm retrainer python scripts/retrain.py --min-feedback $MinFeedback
