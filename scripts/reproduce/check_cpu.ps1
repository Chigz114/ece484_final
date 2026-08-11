$ErrorActionPreference = "Stop"

$env:PYTHONPATH = Join-Path $PSScriptRoot "..\..\src"
python -m unittest discover -s (Join-Path $PSScriptRoot "..\..\tests") -p "test_*.py"
python -m quadpilot.cli.main simulate oracle --track all --max-steps 1200
