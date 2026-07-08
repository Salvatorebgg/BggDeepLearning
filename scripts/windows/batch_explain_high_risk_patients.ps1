# ============================================================
# BggDeepLearning batch high-risk patient explanation script
# File: scripts/windows/batch_explain_high_risk_patients.ps1
#
# Usage:
# .\scripts\windows\batch_explain_high_risk_patients.ps1
# ============================================================

$ProjectRoot = "D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning"

Write-Host "============================================================"
Write-Host "BggDeepLearning batch high-risk patient explanation started"
Write-Host "============================================================"

Set-Location $ProjectRoot

Write-Host "Project root:"
Write-Host (Get-Location)

$VenvPython = ".\.venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    Write-Host "Using virtual environment Python:"
    Write-Host $VenvPython

    & $VenvPython "scripts\python\batch_explain_high_risk_patients.py" --model gradient_boosting --split test --top-n-patients 5
} else {
    Write-Host "Virtual environment Python was not found."
    Write-Host "Using system Python instead."

    python "scripts\python\batch_explain_high_risk_patients.py" --model gradient_boosting --split test --top-n-patients 5
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "------------------------------------------------------------"
    Write-Host "Batch high-risk patient explanation finished successfully." -ForegroundColor Green
    Write-Host "Output folders:"
    Write-Host "outputs\tables"
    Write-Host "outputs\figures"
    Write-Host "outputs\reports"
} else {
    Write-Host "------------------------------------------------------------"
    Write-Host "Batch high-risk patient explanation failed." -ForegroundColor Red
    exit $LASTEXITCODE
}