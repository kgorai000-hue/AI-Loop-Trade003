# AI-Loop-Trade003 resident loop (demo-live)
# Keep FxPro MT5 terminal logged in before starting.
#
# Note: Python logging writes to stderr. Do NOT use ErrorAction Stop around
# the python process, or PowerShell will treat the first log line as fatal.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$env:PYTHONUNBUFFERED = "1"
if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }

$stamp = Get-Date -Format "yyyyMMdd"
$log = Join-Path $Root "logs\resident_$stamp.log"

Write-Host "Starting Trade003 resident loop -> $log"
Write-Host "MT5 demo must stay logged in. Stop with Ctrl+C."

# Continue: native stderr from python must not abort the host pipeline.
$ErrorActionPreference = "Continue"
& python -u main.py loop 2>&1 | ForEach-Object {
    if ($_ -is [System.Management.Automation.ErrorRecord]) {
        $_.Exception.Message
    } else {
        $_
    }
} | Tee-Object -FilePath $log -Append

exit $LASTEXITCODE
