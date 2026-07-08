# ============================================================
# BggDeepLearning Random Forest baseline training script
# File: scripts/windows/train_random_forest_baseline.ps1
#
# Usage:
# .\scripts\windows\train_random_forest_baseline.ps1
# ============================================================

$ProjectRoot = "D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning"

Write-Host "============================================================"
Write-Host "BggDeepLearning Random Forest baseline training started"
Write-Host "============================================================"

Set-Location $ProjectRoot

Write-Host "Project root:"
Write-Host (Get-Location)

$VenvPython = ".\.venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    Write-Host "Using virtual environment Python:"
    Write-Host $VenvPython

    & $VenvPython "scripts\python\train_random_forest_baseline.py"
} else {
    Write-Host "Virtual environment Python was not found."
    Write-Host "Using system Python instead."

    python "scripts\python\train_random_forest_baseline.py"
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "------------------------------------------------------------"
    Write-Host "Random Forest baseline training finished successfully." -ForegroundColor Green
    Write-Host "Metrics file: outputs\tables\random_forest_baseline_metrics.csv"
    Write-Host "Model file: outputs\models\random_forest_baseline.joblib"
    Write-Host "Comparison file: outputs\tables\model_comparison_metrics.csv"
} else {
    Write-Host "------------------------------------------------------------"
    Write-Host "Random Forest baseline training failed." -ForegroundColor Red
    exit $LASTEXITCODE
}