$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
docker compose --profile train run --rm trainer python -m src.pipeline.finalize

