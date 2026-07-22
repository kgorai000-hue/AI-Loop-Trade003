@echo off
cd /d "%~dp0\..\.."
if not exist logs mkdir logs
set PYTHONIOENCODING=utf-8
call .venv\Scripts\activate.bat
python scripts\dry_run_pipeline.py >> logs\dry_run_verbose.log 2>&1
