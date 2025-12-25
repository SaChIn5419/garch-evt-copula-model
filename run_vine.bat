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
REM Attempt to find Conda activation script automatically
set "CONDA_ACTIVATE="
for %%p in (
    "C:\Users\%USERNAME%\anaconda3\Scripts\activate.bat"
    "C:\ProgramData\Anaconda3\Scripts\activate.bat"
    "%USERPROFILE%\anaconda3\Scripts\activate.bat"
    "%LOCALAPPDATA%\Continuum\anaconda3\Scripts\activate.bat"
) do (
    if exist %%p (
        REM Use ~ to strip quotes from the loop variable then set variable explicitly
        set "CONDA_ACTIVATE=%%~p"
        goto FoundConda
    )
)

:FoundConda
if defined CONDA_ACTIVATE (
    echo [INFO] Found Conda at: "%CONDA_ACTIVATE%"
    echo Activating Conda base environment...
    call "%CONDA_ACTIVATE%"
) else (
    echo [WARNING] Conda not found. Trying to use system Python...
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
