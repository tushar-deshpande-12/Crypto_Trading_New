@echo off
REM Cryptocurrency Data Fetching Application Launcher
REM Launches the data pipeline GUI for fetching historical OHLCV data

echo ========================================
echo Crypto Data Pipeline Launcher
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo Starting Data Fetch Application...
echo.

REM Run the application
python fetch_data.py

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo Application exited with error
    pause
)
