@echo off
rem Double-click to open the Cutline web app in your browser.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo First-time setup: installing Cutline. This takes a few minutes...
    python -m venv .venv || goto :nopython
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :failed
)
".venv\Scripts\python.exe" app.py
pause
exit /b

:nopython
echo Python was not found. Install Python 3.10+ from https://www.python.org/downloads/ and try again.
pause
exit /b 1

:failed
echo Installing the requirements failed. See the messages above.
pause
exit /b 1
