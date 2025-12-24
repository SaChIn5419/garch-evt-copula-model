@echo off
title Vine Copula Stress Test
color 0E
echo.
echo ==================================================
echo      ADVANCED VINE COPULA STRESS TEST
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

echo Running Vine Stress Test...
echo.
python vine_stress_test.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] The script encountered an error.
    echo.
    color 0C
) else (
    echo.
    echo [SUCCESS] Stress Test Completed.
)

pause
