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

echo.
echo =========================================================
echo    Select Crisis Cluster to Diagnose
echo =========================================================
echo    [1] Yen Carry Unwind (JPY, Nifty, SPX)
echo    [2] China Industrial (Copper, Tata Steel, HSI)
echo    [3] Real Estate Rates (US 10Y, DLF, REITs)
echo    [4] AI Arms Race (Nvidia, TSMC, Infosys)
echo    [5] Defensive Rotation (Nifty, Unilever, ITC)
echo    [all] RUN ALL
echo =========================================================
set /p CLUSTER_ID="Enter Selection (1-5 or all): "

echo.
echo Running Analysis...
python pipeline.py %CLUSTER_ID%

pause
