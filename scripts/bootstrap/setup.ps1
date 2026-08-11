param(
    [string]$EnvironmentPath = ".venv"
)

$ErrorActionPreference = "Stop"

python -m venv $EnvironmentPath
$pythonPath = Join-Path $EnvironmentPath "Scripts\python.exe"
& $pythonPath -m pip install --upgrade pip
& $pythonPath -m pip install -e ".[dev]"
& $pythonPath -m quadpilot.cli.main --help
