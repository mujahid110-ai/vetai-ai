@echo off
setlocal
echo ========================================
echo   VetAI — Flask ML API (backend/)
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH.
    pause
    exit /b 1
)

set ROOT=%~dp0
cd /d "%ROOT%backend"

if not exist "%ROOT%venv" (
    echo Creating venv...
    python -m venv "%ROOT%venv"
)

call "%ROOT%venv\Scripts\activate.bat"
echo Installing backend dependencies...
pip install -r requirements.txt -q

if "%~1"=="train" (
    echo Training models...
    python train_models.py
)

echo.
echo Starting API on http://0.0.0.0:5000
echo Press Ctrl+C to stop
python ml_server.py
pause
