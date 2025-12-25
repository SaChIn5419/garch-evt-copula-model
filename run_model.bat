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
:: Attempt to find Conda activation script automatically
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
