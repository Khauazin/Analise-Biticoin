$ErrorActionPreference = 'Stop'

param(
  [switch]$Dev
)

$venvPython = Join-Path $PSScriptRoot '..\venv\Scripts\python.exe'
$venvPip    = Join-Path $PSScriptRoot '..\venv\Scripts\pip.exe'
$repoRoot   = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $repoRoot

if (-not (Test-Path $venvPython)) {
  Write-Host 'Criando venv em .\venv...' -ForegroundColor Cyan
  python -m venv .\venv
}

Write-Host 'Instalando dependências do backend...' -ForegroundColor Cyan
try { & $venvPython -m pip install --upgrade pip | Out-Null } catch {}
& $venvPip install -r requirements.txt

if ($Dev) {
  $env:ENABLE_BACKGROUND_SERVICE = '1'
  Write-Host 'Iniciando API (modo DEV com reload) em http://localhost:8000 ...' -ForegroundColor Green
  & $venvPython -m uvicorn api.app:app --reload --port 8000
} else {
  $env:ENABLE_BACKGROUND_SERVICE = '0'
  $workers = $env:UVICORN_WORKERS
  if (-not $workers) { $workers = 1 }
  Write-Host "Iniciando API (PROD, workers=$workers) em http://0.0.0.0:8000 ..." -ForegroundColor Green
  & $venvPython -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers $workers
}
