# ============================================================
# BggDeepLearning database initialization script
# File: scripts/windows/init_database.ps1
#
# Usage:
# Run this command in the project root:
# .\scripts\windows\init_database.ps1
# ============================================================

$ProjectRoot = "D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning"

Write-Host "============================================================"
Write-Host "BggDeepLearning database initialization started"
Write-Host "============================================================"

Set-Location $ProjectRoot

Write-Host "Project root:"
Write-Host (Get-Location)

$VenvPython = ".\.venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    Write-Host "Using virtual environment Python:"
    Write-Host $VenvPython
    & $VenvPython "scripts\python\init_sqlite_database.py"
} else {
    Write-Host "Virtual environment Python was not found."
    Write-Host "Using system Python instead."
    python "scripts\python\init_sqlite_database.py"
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "------------------------------------------------------------"
    Write-Host "Database initialization finished successfully." -ForegroundColor Green
    Write-Host "Database file: database\sqlite\bgg_clinical.db"
} else {
    Write-Host "------------------------------------------------------------"
    Write-Host "Database initialization failed." -ForegroundColor Red
    exit $LASTEXITCODE
}