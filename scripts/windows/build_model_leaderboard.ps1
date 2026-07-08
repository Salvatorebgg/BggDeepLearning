# ============================================================
# BggDeepLearning model leaderboard script
# File: scripts/windows/build_model_leaderboard.ps1
#
# Usage:
# .\scripts\windows\build_model_leaderboard.ps1
# ============================================================

$ProjectRoot = "D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning"

Write-Host "============================================================"
Write-Host "BggDeepLearning model leaderboard building started"
Write-Host "============================================================"

Set-Location $ProjectRoot

Write-Host "Project root:"
Write-Host (Get-Location)

$VenvPython = ".\.venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    Write-Host "Using virtual environment Python:"
    Write-Host $VenvPython

    & $VenvPython "scripts\python\build_model_leaderboard.py"
} else {
    Write-Host "Virtual environment Python was not found."
    Write-Host "Using system Python instead."

    python "scripts\python\build_model_leaderboard.py"
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "------------------------------------------------------------"
    Write-Host "Model leaderboard building finished successfully." -ForegroundColor Green
    Write-Host "Leaderboard file: outputs\tables\model_leaderboard_all.csv"
    Write-Host "Report file: outputs\reports\model_leaderboard_report.txt"
} else {
    Write-Host "------------------------------------------------------------"
    Write-Host "Model leaderboard building failed." -ForegroundColor Red
    exit $LASTEXITCODE
}