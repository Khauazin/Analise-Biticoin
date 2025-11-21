# Analise Financeira Bitcoin

Backend e utilitários para análise técnica e execução de decisões de
trading em Bitcoin, com FastAPI, Binance e persistência local em SQLite.

## Requisitos

- Python 3.11+  
- Git  
- (Opcional) Docker/Docker Compose

## Configuração rápida

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Variáveis de ambiente principais

- `DB_PATH`: caminho do banco SQLite (padrão: btc_trader.db)
- `ENCRYPTION_KEY_FILE`: arquivo com chave Fernet (padrão:
  encryption_key.key)
- `ALLOW_ORIGINS`: origens permitidas no CORS (padrão: *)
- `ENABLE_BACKGROUND_SERVICE`: `1` para habilitar o loop de serviço
  automático
- Chaves Binance:
  - `ENCRYPTION_KEY` (opcional, base64) ou via endpoint `/keys`

## Executando a API

```bash
uvicorn api.app:app --reload --port 8000
```

Endpoints úteis:
- `GET /status` – saúde e chaves configuradas
- `GET /summary` – resumo de indicadores/risco
- `GET /analysis?tf=1h` – análise detalhada por timeframe
- `GET /direction?tf=1h` – decisão simplificada

## Outras tarefas

- Limpeza de cache: `cache_manager.start_cache_cleanup()`
- Sincronizar conta spot: `POST /sync`
- Configurações automáticas: `POST /config`

## Docker (opcional)

```bash
docker compose up --build
```
