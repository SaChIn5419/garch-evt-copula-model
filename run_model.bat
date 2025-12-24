@echo off
title GARCH-Copula Portfolio Optimizer
color 0A
echo.
echo ==================================================
echo      GARCH-Copula-CVaR Optimization Model
echo ==================================================
echo.

:: Ensure we are in the script's directory
cd /d "%~dp0"

:: --- CONDA DETECTION START ---
:: Try to find Conda activation script in common locations
set "CONDA_PATH="
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" set "CONDA_PATH=%USERPROFILE%\anaconda3\Scripts\activate.bat"
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" set "CONDA_PATH=%USERPROFILE%\miniconda3\Scripts\activate.bat"
if exist "%ProgramData%\anaconda3\Scripts\activate.bat" set "CONDA_PATH=%ProgramData%\anaconda3\Scripts\activate.bat"
if exist "%ProgramData%\miniconda3\Scripts\activate.bat" set "CONDA_PATH=%ProgramData%\miniconda3\Scripts\activate.bat"

if defined CONDA_PATH (
    echo [INFO] Found Conda at: %CONDA_PATH%
    echo Activating Conda base environment...
    call "%CONDA_PATH%"
) else (
    echo [INFO] Could not auto-detect Conda. Assuming Python is in PATH...
)
:: --- CONDA DETECTION END ---

:: Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is still not found.
    echo If you installed Conda in a custom location, please edit this .bat file
    echo and set the path to 'activate.bat' manually.
    echo.
    pause
    exit /b
)

echo Running Model...
echo.
python quant_model.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] The script encountered an error.
    echo.
    color 0C
) else (
    echo.
    echo [SUCCESS] Optimization completed successfully.
)

pause
