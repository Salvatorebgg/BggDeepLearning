# ============================================================
# BggDeepLearning SCI-style Word report with SHAP export script
# File: scripts/windows/export_sci_word_report_with_shap.ps1
#
# Usage:
# .\scripts\windows\export_sci_word_report_with_shap.ps1
# ============================================================

$ProjectRoot = "D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning"

Write-Host "============================================================"
Write-Host "BggDeepLearning SCI-style Word report with SHAP export started"
Write-Host "============================================================"

Set-Location $ProjectRoot

Write-Host "Project root:"
Write-Host (Get-Location)

$VenvPython = ".\.venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    Write-Host "Using virtual environment Python:"
    Write-Host $VenvPython

    & $VenvPython "scripts\python\export_sci_word_report_with_shap.py"
} else {
    Write-Host "Virtual environment Python was not found."
    Write-Host "Using system Python instead."

    python "scripts\python\export_sci_word_report_with_shap.py"
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "------------------------------------------------------------"
    Write-Host "SCI-style Word report with SHAP export finished successfully." -ForegroundColor Green
    Write-Host "SHAP DOCX: outputs\reports\clinical_model_results_report_sci_shap.docx"
    Write-Host "SHAP text: outputs\reports\shap_interpretation_text.txt"
} else {
    Write-Host "------------------------------------------------------------"
    Write-Host "SCI-style Word report with SHAP export failed." -ForegroundColor Red
    exit $LASTEXITCODE
}