# ============================================================
# BggDeepLearning C build script
# File: scripts/windows/build_c_tools.ps1
#
# Usage:
# Run this command in the project root:
# .\scripts\windows\build_c_tools.ps1
# ============================================================

$ProjectRoot = "D:\BaiduSyncdisk\document\workdocument\task\BggDeepLearning"

Write-Host "============================================================"
Write-Host "BggDeepLearning C build script started"
Write-Host "============================================================"

Set-Location $ProjectRoot

Write-Host "Project root:"
Write-Host (Get-Location)

$IncludeDir = "c\include"
$SourceFile = "c\src\bgg_math.c"
$TestFile = "c\tests\test_bgg_math.c"
$BuildDir = "c\build"
$OutputExe = "c\build\test_bgg_math.exe"

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

Write-Host "------------------------------------------------------------"
Write-Host "Checking available C compiler..."

$GccCommand = Get-Command gcc -ErrorAction SilentlyContinue
$ClangCommand = Get-Command clang -ErrorAction SilentlyContinue
$ClCommand = Get-Command cl -ErrorAction SilentlyContinue

if ($null -ne $GccCommand) {
    Write-Host "Using GCC:"
    Write-Host $GccCommand.Source

    gcc -I $IncludeDir $SourceFile $TestFile -o $OutputExe

    if ($LASTEXITCODE -ne 0) {
        Write-Host "GCC build failed." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}
elseif ($null -ne $ClangCommand) {
    Write-Host "Using Clang:"
    Write-Host $ClangCommand.Source

    clang -I $IncludeDir $SourceFile $TestFile -o $OutputExe

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Clang build failed." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}
elseif ($null -ne $ClCommand) {
    Write-Host "Using Microsoft C compiler:"
    Write-Host $ClCommand.Source

    cl /I $IncludeDir $SourceFile $TestFile /Fe:$OutputExe

    if ($LASTEXITCODE -ne 0) {
        Write-Host "MSVC build failed." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}
else {
    Write-Host "No C compiler was found." -ForegroundColor Yellow
    Write-Host "Please install one of the following compilers later:"
    Write-Host "1. GCC via MinGW or MSYS2"
    Write-Host "2. Clang"
    Write-Host "3. Microsoft C/C++ Build Tools"
    Write-Host "The C source files have been created successfully."
    exit 1
}

Write-Host "------------------------------------------------------------"
Write-Host "C build finished successfully."
Write-Host "Output executable:"
Write-Host $OutputExe

Write-Host "------------------------------------------------------------"
Write-Host "Running C test executable..."

& ".\c\build\test_bgg_math.exe"

if ($LASTEXITCODE -eq 0) {
    Write-Host "------------------------------------------------------------"
    Write-Host "C module test finished successfully." -ForegroundColor Green
} else {
    Write-Host "------------------------------------------------------------"
    Write-Host "C module test failed." -ForegroundColor Red
    exit $LASTEXITCODE
}