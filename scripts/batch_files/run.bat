@echo off
REM Crypto AI Predictor Launcher
REM Unified application with symbol tracking, charts, and data pipeline

echo ========================================
echo  Crypto AI Predictor
echo  Version 3.0.0
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo Starting application...
echo.

REM Run the application
python app.py

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo Application exited with error
    pause
)
