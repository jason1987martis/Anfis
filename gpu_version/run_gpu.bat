@echo off
set "SCRIPT_DIR=%~dp0"
set "VENV_PYTHON=%SCRIPT_DIR%..\.venv\Scripts\python.exe"

if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" "%SCRIPT_DIR%run_gpu.py" %*
) else (
    python "%SCRIPT_DIR%run_gpu.py" %*
)
