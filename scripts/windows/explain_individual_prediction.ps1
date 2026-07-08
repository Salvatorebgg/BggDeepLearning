# ============================================================
# BggDeepLearning individual prediction explanation script
# File: scripts/windows/explain_individual_prediction.ps1
#
# Usage:
# .\scripts\windows\explain_individual_prediction.ps1
# ============================================================

$ProjectRoot = "D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning"

Write-Host "============================================================"
Write-Host "BggDeepLearning individual prediction explanation started"
Write-Host "============================================================"

Set-Location $ProjectRoot

Write-Host "Project root:"
Write-Host (Get-Location)

$VenvPython = ".\.venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    Write-Host "Using virtual environment Python:"
    Write-Host $VenvPython

    & $VenvPython "scripts\python\explain_individual_prediction.py" --model gradient_boosting --split test --sample-index 0
} else {
    Write-Host "Virtual environment Python was not found."
    Write-Host "Using system Python instead."

    python "scripts\python\explain_individual_prediction.py" --model gradient_boosting --split test --sample-index 0
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "------------------------------------------------------------"
    Write-Host "Individual prediction explanation finished successfully." -ForegroundColor Green
    Write-Host "Output folders:"
    Write-Host "outputs\tables"
    Write-Host "outputs\figures"
    Write-Host "outputs\reports"
} else {
    Write-Host "------------------------------------------------------------"
    Write-Host "Individual prediction explanation failed." -ForegroundColor Red
    exit $LASTEXITCODE
}