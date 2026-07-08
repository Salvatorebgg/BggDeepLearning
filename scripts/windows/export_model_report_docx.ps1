# ============================================================
# BggDeepLearning Word report export script
# File: scripts/windows/export_model_report_docx.ps1
#
# Usage:
# .\scripts\windows\export_model_report_docx.ps1
# ============================================================

$ProjectRoot = "D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning"

Write-Host "============================================================"
Write-Host "BggDeepLearning Word report export started"
Write-Host "============================================================"

Set-Location $ProjectRoot

Write-Host "Project root:"
Write-Host (Get-Location)

$VenvPython = ".\.venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    Write-Host "Using virtual environment Python:"
    Write-Host $VenvPython

    & $VenvPython "scripts\python\export_model_report_docx.py"
} else {
    Write-Host "Virtual environment Python was not found."
    Write-Host "Using system Python instead."

    python "scripts\python\export_model_report_docx.py"
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "------------------------------------------------------------"
    Write-Host "Word report export finished successfully." -ForegroundColor Green
    Write-Host "Comprehensive DOCX: outputs\reports\comprehensive_model_evaluation_report.docx"
    Write-Host "Clinical results DOCX: outputs\reports\clinical_model_results_report.docx"
} else {
    Write-Host "------------------------------------------------------------"
    Write-Host "Word report export failed." -ForegroundColor Red
    exit $LASTEXITCODE
}