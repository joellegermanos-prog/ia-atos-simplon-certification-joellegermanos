param(
    [switch]$Build
)

$ErrorActionPreference = "Stop"
$artifact = "services/model/models/cisia_emploi_promoted.joblib"
$metadata = "services/model/models/cisia_emploi_promoted.json"

if (-not (Test-Path $artifact) -or -not (Test-Path $metadata)) {
    throw "Promoted model artifacts are missing. Run the retrainer and pass its promotion gate first."
}

$env:MODEL_ARTIFACT = "/app/models/cisia_emploi_promoted.joblib"
$composeArgs = @("compose", "up", "-d", "--force-recreate")
if ($Build) {
    $composeArgs += "--build"
}
$composeArgs += @("model", "backend", "frontend")
& docker @composeArgs
if ($LASTEXITCODE -ne 0) {
    throw "Docker Compose deployment failed."
}

docker compose ps model backend frontend