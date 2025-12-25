@echo off
setlocal

echo =========================================
echo    GARCH-EVT-COPULA MODEL PIPELINE
echo =========================================

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

echo Running Full Pipeline...
python pipeline.py

pause
