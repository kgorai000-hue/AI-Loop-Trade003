@echo off
cd /d "%~dp0\..\.."
if not exist logs mkdir logs
set PYTHONIOENCODING=utf-8
call .venv\Scripts\activate.bat
set LOG=logs\weekly_reports.log
echo ===== %date% %time% trade_log_report =====>> "%LOG%"
python scripts\trade_log_report.py >> "%LOG%" 2>&1
echo ===== %date% %time% death_modes_report =====>> "%LOG%"
python scripts\death_modes_report.py --compact >> "%LOG%" 2>&1
