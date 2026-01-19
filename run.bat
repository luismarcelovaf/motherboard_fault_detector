@echo off
setlocal enabledelayedexpansion

title Motherboard Fault Detection AI

echo ============================================================
echo   MOTHERBOARD FAULT DETECTION AI
echo   Startup Script
echo ============================================================
echo.

cd /d "%~dp0"

set VENV_DIR=venv
set PYTHON_EXE=%VENV_DIR%\Scripts\python.exe
set REQUIREMENTS=requirements.txt

:: Check if venv exists and torch is installed
if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" -c "import torch" >nul 2>&1
    if !errorlevel! equ 0 (
        echo [OK] Virtual environment found.
        goto :check_cuda
    ) else (
        echo [!] Virtual environment found but PyTorch missing.
        goto :install_packages
    )
)

echo [!] Virtual environment not found. Setting up...
echo.

:: Try to find Python 3.12 or 3.11
set PYTHON_CMD=
for %%V in (3.12 3.11 3.10) do (
    if not defined PYTHON_CMD (
        py -%%V --version >nul 2>&1
        if !errorlevel! equ 0 (
            set PYTHON_CMD=py -%%V
            echo [OK] Found Python %%V
        )
    )
)

if not defined PYTHON_CMD (
    echo [ERROR] Python 3.10, 3.11, or 3.12 not found!
    echo.
    echo Please install Python 3.12 from:
    echo   https://www.python.org/downloads/
    echo.
    echo Or via winget:
    echo   winget install Python.Python.3.12
    echo.
    pause
    exit /b 1
)

:: Create virtual environment
echo.
echo Creating virtual environment...
%PYTHON_CMD% -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment!
    pause
    exit /b 1
)
echo [OK] Virtual environment created.

:install_packages
:: Upgrade pip
echo.
echo Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip >nul 2>&1

:: Detect NVIDIA GPU
echo.
echo Detecting GPU...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] NVIDIA GPU detected - installing PyTorch with CUDA support
    echo This may take several minutes (downloading ~2.5GB)...
    "%PYTHON_EXE%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    if errorlevel 1 (
        echo [WARN] CUDA version failed, trying CPU version...
        "%PYTHON_EXE%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    )
) else (
    echo [INFO] No NVIDIA GPU detected - installing PyTorch CPU version
    echo This may take a few minutes (downloading ~200MB)...
    "%PYTHON_EXE%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

:: Install other requirements
echo.
echo Installing dependencies...
"%PYTHON_EXE%" -m pip install -r "%REQUIREMENTS%"
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies!
    pause
    exit /b 1
)

echo.
echo [OK] Setup complete!
echo.

:check_cuda
:: Quick CUDA check (only if torch is installed)
echo Checking GPU status...
"%PYTHON_EXE%" -c "import torch; cuda=torch.cuda.is_available(); print(f'[{\"OK\" if cuda else \"INFO\"}] CUDA: {\"Available - \" + torch.cuda.get_device_name(0) if cuda else \"Not available (using CPU)\"}')" 2>nul
if errorlevel 1 (
    echo [WARN] PyTorch not installed properly. Try deleting venv folder and run again.
)
echo.

:run_main
:: Run main application
echo Starting application...
echo ============================================================
echo.
"%PYTHON_EXE%" main.py
set EXIT_CODE=%errorlevel%

echo.
if %EXIT_CODE% neq 0 (
    echo Application exited with error code %EXIT_CODE%
    pause
)

exit /b %EXIT_CODE%
