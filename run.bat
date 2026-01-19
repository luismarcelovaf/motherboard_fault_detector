@echo off
setlocal enabledelayedexpansion

:: Check for API mode BEFORE inner call wrapper
if /i "%1"=="api" goto :api_direct
if /i "%1"=="--api" goto :api_direct

:: Catch any crashes
if "%1"=="__INNER__" goto :main
cmd /c "%~f0" __INNER__ %*
if errorlevel 1 (
    echo.
    echo [!] Script ended unexpectedly. Press any key to close.
    pause >nul
)
exit /b %errorlevel%

:api_direct
:: API mode called directly, run inner with API flag
cmd /c "%~f0" __INNER__ __API__
exit /b %errorlevel%

:main
:: Check if API mode was requested
set API_MODE=0
if "%2"=="__API__" set API_MODE=1

if %API_MODE%==1 (
    title Motherboard Fault Detection API
) else (
    title Motherboard Fault Detection AI
)

echo ============================================================
if %API_MODE%==1 (
    echo   MOTHERBOARD FAULT DETECTION API
    echo   API Server Mode
) else (
    echo   MOTHERBOARD FAULT DETECTION AI
    echo   Startup Script
)
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

:: Ask user which PyTorch version to install
echo.
echo Which PyTorch version do you want to install?
echo.
echo   [1] GPU (NVIDIA CUDA) - Recommended if you have an NVIDIA GPU
echo       Faster training and inference (~2.5GB download)
echo.
echo   [2] CPU only - Works on any computer
echo       Slower but compatible everywhere (~200MB download)
echo.
set /p TORCH_CHOICE="Enter choice [1/2]: "

if "%TORCH_CHOICE%"=="1" (
    echo.
    echo Installing PyTorch with CUDA support...
    echo This may take several minutes...
    "%PYTHON_EXE%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    if errorlevel 1 (
        echo [WARN] CUDA version failed, trying CPU version...
        goto :install_cpu_torch
    )
) else (
    goto :install_cpu_torch
)
goto :after_torch_install

:install_cpu_torch
echo.
echo Installing PyTorch CPU version...
echo This may take a few minutes...
"%PYTHON_EXE%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

:after_torch_install
if errorlevel 1 (
    echo [ERROR] Failed to install PyTorch!
    pause
    exit /b 1
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
:: Run main application or API server
if %API_MODE%==1 (
    echo Starting API server...
    echo ============================================================
    echo.
    echo Usage: run.bat api
    echo.
    echo To stop the server, press Ctrl+C
    echo.
    "%PYTHON_EXE%" api_server.py
) else (
    echo Starting application...
    echo ============================================================
    echo.
    "%PYTHON_EXE%" main.py
)
set EXIT_CODE=%errorlevel%

echo.
if %EXIT_CODE% neq 0 (
    echo Application exited with error code %EXIT_CODE%
    pause
)

exit /b %EXIT_CODE%
