# ============================================================
# BggDeepLearning comprehensive model report script
# File: scripts/windows/generate_comprehensive_model_report.ps1
#
# Usage:
# .\scripts\windows\generate_comprehensive_model_report.ps1
# ============================================================

$ProjectRoot = "D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning"

Write-Host "============================================================"
Write-Host "BggDeepLearning comprehensive model report generation started"
Write-Host "============================================================"

Set-Location $ProjectRoot

Write-Host "Project root:"
Write-Host (Get-Location)

$VenvPython = ".\.venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    Write-Host "Using virtual environment Python:"
    Write-Host $VenvPython

    & $VenvPython "scripts\python\generate_comprehensive_model_report.py"
} else {
    Write-Host "Virtual environment Python was not found."
    Write-Host "Using system Python instead."

    python "scripts\python\generate_comprehensive_model_report.py"
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "------------------------------------------------------------"
    Write-Host "Comprehensive model report generation finished successfully." -ForegroundColor Green
    Write-Host "Markdown report: outputs\reports\comprehensive_model_evaluation_report.md"
    Write-Host "Text report: outputs\reports\comprehensive_model_evaluation_report.txt"
} else {
    Write-Host "------------------------------------------------------------"
    Write-Host "Comprehensive model report generation failed." -ForegroundColor Red
    exit $LASTEXITCODE
}