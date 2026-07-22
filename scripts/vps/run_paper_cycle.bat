@echo off
cd /d "%~dp0\..\.."
if not exist logs mkdir logs
call .venv\Scripts\activate.bat
python main.py run --sync >> logs\scheduler.log 2>&1
