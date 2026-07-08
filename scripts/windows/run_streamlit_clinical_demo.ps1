# ============================================================
# BggDeepLearning Streamlit clinical risk demo launcher
# File: scripts/windows/run_streamlit_clinical_demo.ps1
#
# Usage:
# .\scripts\windows\run_streamlit_clinical_demo.ps1
# ============================================================

$ProjectRoot = "D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning"

Write-Host "============================================================"
Write-Host "BggDeepLearning Streamlit clinical risk demo launcher"
Write-Host "============================================================"

Set-Location $ProjectRoot

Write-Host "Project root:"
Write-Host (Get-Location)

$VenvPython = ".\.venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    Write-Host "Using virtual environment Python:"
    Write-Host $VenvPython

    & $VenvPython -m streamlit run "apps\streamlit\clinical_risk_demo.py"
} else {
    Write-Host "Virtual environment Python was not found."
    Write-Host "Using system Python instead."

    python -m streamlit run "apps\streamlit\clinical_risk_demo.py"
}