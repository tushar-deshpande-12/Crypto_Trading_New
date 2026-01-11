@echo off
REM Crypto Market Tracker Launcher v2.0
REM Starts the Binance cryptocurrency monitoring GUI with chart viewer

echo ========================================
echo  Crypto Market Tracker v2.0
echo  Loading Binance market data...
echo ========================================
echo.

REM Run the GUI application
python main.py

REM Pause if there's an error so user can see it
if errorlevel 1 (
    echo.
    echo ========================================
    echo  Error occurred!
    echo ========================================
    pause
)
