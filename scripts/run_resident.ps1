# AI-Loop-Trade003 resident loop (demo-live)
# Keep FxPro MT5 terminal logged in before starting.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$env:PYTHONUNBUFFERED = "1"
if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }

$stamp = Get-Date -Format "yyyyMMdd"
$log = Join-Path $Root "logs\resident_$stamp.log"

Write-Host "Starting Trade003 resident loop -> $log"
& python main.py loop 2>&1 | Tee-Object -FilePath $log -Append
