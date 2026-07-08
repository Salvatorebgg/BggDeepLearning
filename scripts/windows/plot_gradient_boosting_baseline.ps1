# ============================================================
# BggDeepLearning Gradient Boosting plotting script
# File: scripts/windows/plot_gradient_boosting_baseline.ps1
#
# Usage:
# .\scripts\windows\plot_gradient_boosting_baseline.ps1
# ============================================================

$ProjectRoot = "D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning"

Write-Host "============================================================"
Write-Host "BggDeepLearning Gradient Boosting plotting started"
Write-Host "============================================================"

Set-Location $ProjectRoot

Write-Host "Project root:"
Write-Host (Get-Location)

$VenvPython = ".\.venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    Write-Host "Using virtual environment Python:"
    Write-Host $VenvPython

    & $VenvPython "scripts\python\plot_gradient_boosting_baseline.py"
} else {
    Write-Host "Virtual environment Python was not found."
    Write-Host "Using system Python instead."

    python "scripts\python\plot_gradient_boosting_baseline.py"
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "------------------------------------------------------------"
    Write-Host "Gradient Boosting plotting finished successfully." -ForegroundColor Green
    Write-Host "Figure folder: outputs\figures"
    Write-Host "Report file: outputs\reports\gradient_boosting_plot_report.txt"
} else {
    Write-Host "------------------------------------------------------------"
    Write-Host "Gradient Boosting plotting failed." -ForegroundColor Red
    exit $LASTEXITCODE
}