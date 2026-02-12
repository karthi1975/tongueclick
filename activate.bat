@echo off
REM Helper script to activate the virtual environment on Windows

echo ================================================
echo Tongue Click Detector - Virtual Environment
echo ================================================
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Virtual environment activated!
echo.
echo Available commands:
echo   python demo.py                     # Interactive demo
echo   python demo.py --mode realtime     # Real-time detection
echo   python demo.py --mode devices      # List audio devices
echo   python test_basic.py               # Run tests
echo.
echo To deactivate: type 'deactivate'
echo ================================================
echo.

cmd /k
