# ============================================================
# BggDeepLearning C++ build script
# File: scripts/windows/build_cpp_engine.ps1
#
# Usage:
# Run this command in the project root:
# .\scripts\windows\build_cpp_engine.ps1
# ============================================================

$ProjectRoot = "D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning"

Write-Host "============================================================"
Write-Host "BggDeepLearning C++ build script started"
Write-Host "============================================================"

Set-Location $ProjectRoot

Write-Host "Project root:"
Write-Host (Get-Location)

$CppRoot = "cpp"
$BuildDir = "cpp\build"

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

Write-Host "------------------------------------------------------------"
Write-Host "Checking CMake command..."

$CMakeCommand = Get-Command cmake -ErrorAction SilentlyContinue

if ($null -eq $CMakeCommand) {
    Write-Host "CMake was not found." -ForegroundColor Yellow
    Write-Host "The C++ source files have been created successfully."
    Write-Host "Please install CMake later if you want to build the C++ module."
    exit 1
}

Write-Host "CMake found:"
Write-Host $CMakeCommand.Source

Write-Host "------------------------------------------------------------"
Write-Host "Configuring C++ project..."

cmake -S $CppRoot -B $BuildDir

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configure step failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "------------------------------------------------------------"
Write-Host "Building C++ project..."

cmake --build $BuildDir --config Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "C++ build step failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "------------------------------------------------------------"
Write-Host "Searching for test executable..."

$PossibleExecutables = @(
    "cpp\build\Release\test_risk_engine.exe",
    "cpp\build\Debug\test_risk_engine.exe",
    "cpp\build\test_risk_engine.exe"
)

$TestExe = $null

foreach ($ExePath in $PossibleExecutables) {
    if (Test-Path $ExePath) {
        $TestExe = $ExePath
        break
    }
}

if ($null -eq $TestExe) {
    Write-Host "The test executable was not found." -ForegroundColor Red
    Write-Host "Please check the build output folder."
    exit 1
}

Write-Host "Test executable found:"
Write-Host $TestExe

Write-Host "------------------------------------------------------------"
Write-Host "Running C++ risk engine test..."

& ".\$TestExe"

if ($LASTEXITCODE -eq 0) {
    Write-Host "------------------------------------------------------------"
    Write-Host "C++ risk engine test finished successfully." -ForegroundColor Green
} else {
    Write-Host "------------------------------------------------------------"
    Write-Host "C++ risk engine test failed." -ForegroundColor Red
    exit $LASTEXITCODE
}