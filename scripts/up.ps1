param(
    [switch]$Detached
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if ($Detached) {
    docker compose up --build -d
} else {
    docker compose up --build
}

