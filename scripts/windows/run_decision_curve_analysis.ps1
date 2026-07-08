# ============================================================
# BggDeepLearning Decision Curve Analysis script
# File: scripts/windows/run_decision_curve_analysis.ps1
#
# Usage:
# .\scripts\windows\run_decision_curve_analysis.ps1
# ============================================================

$ProjectRoot = "D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning"

Write-Host "============================================================"
Write-Host "BggDeepLearning Decision Curve Analysis started"
Write-Host "============================================================"

Set-Location $ProjectRoot

Write-Host "Project root:"
Write-Host (Get-Location)

$VenvPython = ".\.venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    Write-Host "Using virtual environment Python:"
    Write-Host $VenvPython

    & $VenvPython "scripts\python\run_decision_curve_analysis.py"
} else {
    Write-Host "Virtual environment Python was not found."
    Write-Host "Using system Python instead."

    python "scripts\python\run_decision_curve_analysis.py"
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "------------------------------------------------------------"
    Write-Host "Decision Curve Analysis finished successfully." -ForegroundColor Green
    Write-Host "DCA table: outputs\tables\dca_net_benefit_points.csv"
    Write-Host "DCA report: outputs\reports\dca_report.txt"
} else {
    Write-Host "------------------------------------------------------------"
    Write-Host "Decision Curve Analysis failed." -ForegroundColor Red
    exit $LASTEXITCODE
}