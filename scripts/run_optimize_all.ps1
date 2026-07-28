# AI-Loop-Trade003: optimize all tradeable symbols (or a subset).
# Keep FxPro MT5 optional here; optimize uses OHLCV store + LLM/grid.
#
# Python logging writes to stderr. Do NOT use ErrorAction Stop around
# the python process, or PowerShell will treat the first log line as fatal.

param(
    [string[]]$Symbol = @(),
    [switch]$IncludeGold,
    [switch]$All
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$env:PYTHONUNBUFFERED = "1"
if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }

$stamp = Get-Date -Format "yyyyMMdd_HHmm"
$log = Join-Path $Root "logs\optimize_all_$stamp.log"

$pyArgs = @("-u", "main.py", "optimize")
if ($All) {
    # no --symbol => all Asset Groups
} elseif ($Symbol.Count -gt 0) {
    foreach ($s in $Symbol) {
        $pyArgs += "--symbol"
        $pyArgs += $s
    }
} else {
    # Default: all tradeable except GOLD (already optimized on demo VPS)
    $defaults = @(
        "#US30", "#USSPX500", "#USNDAQ100", "#Japan225",
        "#Germany40", "#UK100",
        "SILVER",
        "EURUSD", "GBPUSD", "USDJPY",
        "WTI"
    )
    if ($IncludeGold) {
        $defaults = @("GOLD") + $defaults
    }
    foreach ($s in $defaults) {
        $pyArgs += "--symbol"
        $pyArgs += $s
    }
}

Write-Host "Starting Trade003 optimize -> $log"
Write-Host "Args: python $($pyArgs -join ' ')"
Write-Host "This may take many hours. Stop with Ctrl+C."

# Continue: native stderr from python must not abort the host pipeline.
$ErrorActionPreference = "Continue"
& python @pyArgs 2>&1 | ForEach-Object {
    if ($_ -is [System.Management.Automation.ErrorRecord]) {
        $_.Exception.Message
    } else {
        $_
    }
} | Tee-Object -FilePath $log -Append

exit $LASTEXITCODE
