# ============================================================
# BggDeepLearning SHAP explainability script
# File: scripts/windows/run_shap_explainability.ps1
#
# Usage:
# .\scripts\windows\run_shap_explainability.ps1
# ============================================================

$ProjectRoot = "D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning"

Write-Host "============================================================"
Write-Host "BggDeepLearning SHAP explainability started"
Write-Host "============================================================"

Set-Location $ProjectRoot

Write-Host "Project root:"
Write-Host (Get-Location)

$VenvPython = ".\.venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    Write-Host "Using virtual environment Python:"
    Write-Host $VenvPython

    & $VenvPython "scripts\python\run_shap_explainability.py"
} else {
    Write-Host "Virtual environment Python was not found."
    Write-Host "Using system Python instead."

    python "scripts\python\run_shap_explainability.py"
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "------------------------------------------------------------"
    Write-Host "SHAP explainability finished successfully." -ForegroundColor Green
    Write-Host "Combined table: outputs\tables\shap_model_global_importance_all.csv"
    Write-Host "Report file: outputs\reports\shap_explainability_report.txt"
} else {
    Write-Host "------------------------------------------------------------"
    Write-Host "SHAP explainability failed." -ForegroundColor Red
    exit $LASTEXITCODE
}