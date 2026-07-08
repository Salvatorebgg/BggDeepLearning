# ============================================================
# BggDeepLearning simulated data generation script
# File: scripts/windows/generate_simulated_data.ps1
#
# Usage:
# .\scripts\windows\generate_simulated_data.ps1
# ============================================================

$ProjectRoot = "D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning"

Write-Host "============================================================"
Write-Host "BggDeepLearning simulated data generation started"
Write-Host "============================================================"

Set-Location $ProjectRoot

Write-Host "Project root:"
Write-Host (Get-Location)

$VenvPython = ".\.venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    Write-Host "Using virtual environment Python:"
    Write-Host $VenvPython

    & $VenvPython "scripts\python\generate_simulated_clinical_data.py" --n 500 --seed 42 --missing-rate 0.03
} else {
    Write-Host "Virtual environment Python was not found."
    Write-Host "Using system Python instead."

    python "scripts\python\generate_simulated_clinical_data.py" --n 500 --seed 42 --missing-rate 0.03
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "------------------------------------------------------------"
    Write-Host "Simulated data generation finished successfully." -ForegroundColor Green
    Write-Host "Data file: data\simulated\clinical_simulated_data.csv"
} else {
    Write-Host "------------------------------------------------------------"
    Write-Host "Simulated data generation failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
