@echo off
REM Crypto AI Predictor - Main Launcher
REM Launches the unified GUI application

echo ================================================
echo   Crypto AI Predictor v3.0
echo   Starting Application...
echo ================================================
echo.

python main.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application failed to start
    echo Press any key to exit...
    pause >nul
)
