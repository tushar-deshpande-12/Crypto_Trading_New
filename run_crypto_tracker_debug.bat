@echo off
REM Crypto Market Tracker Launcher v2.0 (Debug Mode)
REM Keeps console window open to view logs and errors

echo ========================================
echo  Crypto Market Tracker v2.0 (Debug)
echo  Loading Binance market data...
echo ========================================
echo.

REM Run the GUI application
python main.py

REM Always pause to see output/errors
echo.
echo ========================================
echo  Application closed
echo ========================================
pause
