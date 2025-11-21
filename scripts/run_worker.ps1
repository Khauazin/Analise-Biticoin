$ErrorActionPreference = 'Stop'

$venvPython = Join-Path $PSScriptRoot '..\venv\Scripts\python.exe'
$venvPip    = Join-Path $PSScriptRoot '..\venv\Scripts\pip.exe'
$repoRoot   = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $repoRoot

if (-not (Test-Path $venvPython)) {
  Write-Host 'Criando venv em .\venv...' -ForegroundColor Cyan
  python -m venv .\venv
}

Write-Host 'Instalando dependências para worker...' -ForegroundColor Cyan
try { & $venvPython -m pip install --upgrade pip | Out-Null } catch {}
& $venvPip install -r requirements.txt

if (-not $env:BACKGROUND_INTERVAL) { $env:BACKGROUND_INTERVAL = '30' }

Write-Host "Iniciando worker de análise (intervalo=$env:BACKGROUND_INTERVAL s)..." -ForegroundColor Green
& $venvPython worker_service.py

