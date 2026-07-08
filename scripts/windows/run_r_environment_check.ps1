# ============================================================
# BggDeepLearning R environment check launcher
# File: scripts/windows/run_r_environment_check.ps1
#
# Usage:
# Run this command in the project root:
# .\scripts\windows\run_r_environment_check.ps1
# ============================================================

$ProjectRoot = "D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning"

Write-Host "============================================================"
Write-Host "BggDeepLearning R environment check started"
Write-Host "============================================================"

Set-Location $ProjectRoot

Write-Host "Project root:"
Write-Host (Get-Location)

Write-Host "------------------------------------------------------------"
Write-Host "Checking Rscript command..."

$RscriptCommand = Get-Command Rscript -ErrorAction SilentlyContinue

if ($null -eq $RscriptCommand) {
    Write-Host "Rscript was not found." -ForegroundColor Red
    Write-Host "Please install R and add the R bin folder to the system Path."
    Write-Host "Typical path example: C:\Program Files\R\R-4.x.x\bin"
    exit 1
}

Write-Host "Rscript found:"
Write-Host $RscriptCommand.Source

Write-Host "------------------------------------------------------------"
Write-Host "Running R environment check script..."

Rscript "r\scripts\environment_check.R"

if ($LASTEXITCODE -eq 0) {
    Write-Host "------------------------------------------------------------"
    Write-Host "R environment check finished successfully." -ForegroundColor Green
    Write-Host "Log file: outputs\logs\r_environment_check.txt"
} else {
    Write-Host "------------------------------------------------------------"
    Write-Host "R environment check failed. Please check the error message above." -ForegroundColor Red
    exit $LASTEXITCODE
}