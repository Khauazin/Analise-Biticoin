$ErrorActionPreference = 'Stop'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$webDir   = Join-Path $repoRoot 'web'
Set-Location $webDir

if (-not (Test-Path '.env.local')) {
  "NEXT_PUBLIC_API_BASE=http://localhost:8000" | Out-File -Encoding UTF8 -FilePath .env.local -Force
}

Write-Host 'Instalando dependências do front (web)...' -ForegroundColor Cyan
if (Test-Path 'package-lock.json') {
  npm ci
} else {
  npm install
}

Write-Host 'Iniciando Next.js em http://localhost:3000 ...' -ForegroundColor Green
npm run dev

