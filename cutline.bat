@echo off
rem Command-line launcher: cutline.bat "C:\path\to\video.mp4" -n 5
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
"%~dp0.venv\Scripts\python.exe" -m cutline %*
