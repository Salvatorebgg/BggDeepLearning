# ============================================================
# BggDeepLearning model calibration evaluation script
# File: scripts/windows/evaluate_model_calibration.ps1
#
# Usage:
# .\scripts\windows\evaluate_model_calibration.ps1
# ============================================================

$ProjectRoot = "D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning"

Write-Host "============================================================"
Write-Host "BggDeepLearning model calibration evaluation started"
Write-Host "============================================================"

Set-Location $ProjectRoot

Write-Host "Project root:"
Write-Host (Get-Location)

$VenvPython = ".\.venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    Write-Host "Using virtual environment Python:"
    Write-Host $VenvPython

    & $VenvPython "scripts\python\evaluate_model_calibration.py"
} else {
    Write-Host "Virtual environment Python was not found."
    Write-Host "Using system Python instead."

    python "scripts\python\evaluate_model_calibration.py"
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "------------------------------------------------------------"
    Write-Host "Model calibration evaluation finished successfully." -ForegroundColor Green
    Write-Host "Metrics file: outputs\tables\model_calibration_metrics.csv"
    Write-Host "Report file: outputs\reports\model_calibration_report.txt"
} else {
    Write-Host "------------------------------------------------------------"
    Write-Host "Model calibration evaluation failed." -ForegroundColor Red
    exit $LASTEXITCODE
}