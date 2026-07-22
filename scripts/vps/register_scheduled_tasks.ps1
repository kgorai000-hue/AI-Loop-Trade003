# Register VPS scheduled tasks for MT5_loop paper dry_run monitoring.
# Run in elevated PowerShell:  .\scripts\vps\register_scheduled_tasks.ps1

param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string]$HourlyStart = "00:05",
    [string]$DailyVerboseAt = "07:00",
    [string]$WeeklyReportsDay = "MON",
    [string]$WeeklyReportsAt = "08:00"
)

$ErrorActionPreference = "Stop"

function Register-Task {
    param(
        [string]$Name,
        [string]$BatPath,
        [string]$Schedule,
        [string[]]$ExtraArgs = @()
    )
    if (-not (Test-Path $BatPath)) {
        throw "Batch file not found: $BatPath"
    }
    $args = @("/create", "/tn", $Name, "/tr", $BatPath, "/sc", $Schedule, "/rl", "HIGHEST", "/f") + $ExtraArgs
    & schtasks @args
    if ($LASTEXITCODE -ne 0) {
        throw "schtasks failed for $Name (exit $LASTEXITCODE)"
    }
    Write-Host "Registered: $Name -> $BatPath"
}

$cycleBat = Join-Path $ProjectRoot "scripts\vps\run_paper_cycle.bat"
$verboseBat = Join-Path $ProjectRoot "scripts\vps\run_paper_verbose.bat"
$weeklyBat = Join-Path $ProjectRoot "scripts\vps\run_weekly_reports.bat"

Write-Host "Project root: $ProjectRoot"
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "logs") | Out-Null

Register-Task -Name "MT5_PaperRun" -BatPath $cycleBat -Schedule "hourly" -ExtraArgs @("/mo", "1", "/st", $HourlyStart)
Register-Task -Name "MT5_PaperVerbose" -BatPath $verboseBat -Schedule "daily" -ExtraArgs @("/st", $DailyVerboseAt)
Register-Task -Name "MT5_WeeklyReports" -BatPath $weeklyBat -Schedule "weekly" -ExtraArgs @("/d", $WeeklyReportsDay, "/st", $WeeklyReportsAt)

Write-Host ""
Write-Host "=== Registered tasks ==="
schtasks /query /tn "MT5_PaperRun" /fo list | Select-String "TaskName|Status|Next Run"
schtasks /query /tn "MT5_PaperVerbose" /fo list | Select-String "TaskName|Status|Next Run"
schtasks /query /tn "MT5_WeeklyReports" /fo list | Select-String "TaskName|Status|Next Run"

Write-Host ""
Write-Host "Ensure config/settings.yaml has ops.heartbeat_timeout_seconds: 3900"
Write-Host "Or copy ops section from config/settings.local.yaml.example to settings.local.yaml"
