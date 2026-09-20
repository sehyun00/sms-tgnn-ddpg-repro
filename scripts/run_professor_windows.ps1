[CmdletBinding()]
param(
    [switch]$CoreOnly,
    [switch]$SkipInstall,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvRoot = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"

Set-Location -LiteralPath $repoRoot

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git was not found. Install Git for Windows and rerun this script."
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        & $pyLauncher.Source -3 -m venv $venvRoot
    }
    elseif ($pythonCommand) {
        & $pythonCommand.Source -m venv $venvRoot
    }
    else {
        throw "Python 3 was not found. Install Python 3 and rerun this script."
    }
    if (-not (Test-Path -LiteralPath $venvPython)) {
        throw "Python was found, but .venv creation failed. Install a supported 64-bit Python 3 release and rerun."
    }
}

if (-not $SkipInstall) {
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
    & $venvPython -m pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw "CUDA-enabled PyTorch installation failed" }
    & $venvPython -m pip install -r (Join-Path $repoRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
}

$runMode = if ($CoreOnly) { "core" } else { "full" }
$runnerArgs = @("scripts/run_professor_local.py", "--mode", $runMode)
if ($SkipTests) { $runnerArgs += "--skip-tests" }

& $venvPython @runnerArgs
if ($LASTEXITCODE -ne 0) { throw "Professor experiment runner failed with exit code $LASTEXITCODE" }
