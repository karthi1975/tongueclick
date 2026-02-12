@echo off
REM Quick script to run the demo with virtual environment (Windows)

call venv\Scripts\activate.bat
python demo.py %*
call deactivate
