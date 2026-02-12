@echo off
REM ============================================
REM Tongue Click Detector - Complete Setup (Windows)
REM From fresh clone to running detector
REM ============================================

echo ============================================
echo  Tongue Click Detector - Full Setup
echo ============================================
echo.

REM ------------------------------------------
REM Step 1: Find Python
REM ------------------------------------------
echo [1/5] Finding Python...

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Install Python 3.11+ from python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo   Using: Python %PYTHON_VERSION%
echo.

REM ------------------------------------------
REM Step 2: Create virtual environment
REM ------------------------------------------
echo [2/5] Setting up virtual environment...

if exist venv (
    echo   Existing venv found. Removing...
    rmdir /s /q venv
)

python -m venv venv
echo   venv created
echo.

REM ------------------------------------------
REM Step 3: Activate and upgrade pip
REM ------------------------------------------
echo [3/5] Activating venv and upgrading pip...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
echo   pip upgraded
echo.

REM ------------------------------------------
REM Step 4: Install dependencies
REM ------------------------------------------
echo [4/5] Installing dependencies...
echo   This may take a few minutes...
echo.
pip install -r requirements.txt
echo.

REM ------------------------------------------
REM Step 5: Verify installation
REM ------------------------------------------
echo [5/5] Verifying installation...
echo.
python verify_installation.py
echo.

echo ============================================
echo  Setup Complete!
echo ============================================
echo.
echo USAGE:
echo.
echo   # Activate the venv (each new terminal)
echo   venv\Scripts\activate.bat
echo.
echo   # Run the detector
echo   python tongue_click_detector.py
echo.
echo   # Run the demo
echo   python demo.py
echo.
echo   # Deactivate when done
echo   deactivate
echo.
pause
