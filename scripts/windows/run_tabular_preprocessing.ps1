# ============================================================
# BggDeepLearning tabular preprocessing script
# File: scripts/windows/run_tabular_preprocessing.ps1
#
# Usage:
# .\scripts\windows\run_tabular_preprocessing.ps1
# ============================================================

$ProjectRoot = "D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning"

Write-Host "============================================================"
Write-Host "BggDeepLearning tabular preprocessing started"
Write-Host "============================================================"

Set-Location $ProjectRoot

Write-Host "Project root:"
Write-Host (Get-Location)

$VenvPython = ".\.venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    Write-Host "Using virtual environment Python:"
    Write-Host $VenvPython

    & $VenvPython "scripts\python\run_tabular_preprocessing.py"
} else {
    Write-Host "Virtual environment Python was not found."
    Write-Host "Using system Python instead."

    python "scripts\python\run_tabular_preprocessing.py"
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "------------------------------------------------------------"
    Write-Host "Tabular preprocessing finished successfully." -ForegroundColor Green
    Write-Host "Processed data folder: data\processed"
    Write-Host "Report file: outputs\reports\preprocessing_report.txt"
} else {
    Write-Host "------------------------------------------------------------"
    Write-Host "Tabular preprocessing failed." -ForegroundColor Red
    exit $LASTEXITCODE
}