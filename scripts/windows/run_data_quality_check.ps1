# ============================================================
# BggDeepLearning data quality check script
# File: scripts/windows/run_data_quality_check.ps1
#
# Usage:
# .\scripts\windows\run_data_quality_check.ps1
# ============================================================

$ProjectRoot = "D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning"

Write-Host "============================================================"
Write-Host "BggDeepLearning data quality check started"
Write-Host "============================================================"

Set-Location $ProjectRoot

Write-Host "Project root:"
Write-Host (Get-Location)

$VenvPython = ".\.venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    Write-Host "Using virtual environment Python:"
    Write-Host $VenvPython

    & $VenvPython "scripts\python\run_data_quality_check.py"
} else {
    Write-Host "Virtual environment Python was not found."
    Write-Host "Using system Python instead."

    python "scripts\python\run_data_quality_check.py"
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "------------------------------------------------------------"
    Write-Host "Data quality check finished successfully." -ForegroundColor Green
    Write-Host "Report file: outputs\reports\data_quality_report.txt"
} else {
    Write-Host "------------------------------------------------------------"
    Write-Host "Data quality check failed." -ForegroundColor Red
    exit $LASTEXITCODE
}