@echo off
chcp 65001 >nul
cd /d "%~dp0"
python run_all.py
if errorlevel 1 (
  echo.
  echo PIPELINE FAILED
  pause
  exit /b 1
)
echo.
echo PIPELINE COMPLETED
pause
