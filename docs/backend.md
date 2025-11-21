# CryptoVision Backend

## Visão Geral
- API FastAPI com análise de mercado (Binance), indicadores técnicos, previsões de ML, gerenciador de risco e persistência leve em SQLite.
- Endpoints para status, sumário, análise por timeframe, ordens abertas, configuração e armazenamento de chaves da Binance.

## Como Rodar (Local)
- Requisitos: Python 3.11+, pip, venv.
- Windows PowerShell:
  - `python -m venv .venv`
  - `.\.venv\Scripts\Activate.ps1`
  - `pip install -r requirements.txt`
  - `uvicorn api.app:app --reload --port 8000`
- Linux/Mac:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
  - `uvicorn api.app:app --reload --port 8000`
- Documentação interativa: `GET /docs` (Swagger) e `GET /redoc`.

### Docker / Docker Compose
- Subir API e Worker: `docker compose up -d --build api worker`
- Observação: o serviço `web` do compose requer uma pasta `./web` (frontend) que pode não estar no repositório. Se não existir, suba apenas `api` e `worker`.
- Variáveis úteis: `DB_PATH`, `ALLOW_ORIGINS`, `BACKGROUND_INTERVAL` e chaves da Binance.

## Variáveis de Ambiente
- Binance (opcional, recomendado): `BINANCE_API_KEY`, `BINANCE_API_SECRET`
- CORS: `ALLOW_ORIGINS` (padrão `*`, ex.: `http://localhost:3000,http://meuapp.com`)
- Serviço em background (loop interno): `ENABLE_BACKGROUND_SERVICE=1` para ativar; `BACKGROUND_INTERVAL` (segundos)
- Banco/Criptografia: `DB_PATH` (padrão `btc_trader.db`), `ENCRYPTION_KEY_FILE` (padrão `encryption_key.key`), `ENCRYPTION_KEY` (base64 urlsafe)

## Arquitetura (arquivos-chave)
- `api/app.py`: aplicação FastAPI e rotas
- `Market_Analyzer.py`: coleta de dados, indicadores técnicos, previsões de ML e decisão
- `RiskManager.py`: sizing/controle de risco e snapshots
- `BinanceConnector.py`: integração com Binance (Spot), cache e proteção (circuit breaker)
- `database.py`: SQLite, chaves criptografadas, configs e caches
- `ml_model.py`: treino/carregamento do modelo
- `News_Worker.py` (opcional): sentimento de notícias
- Utilitários: `cache_manager.py`, `alert_system.py`, `Logger.py`

## Endpoints (atuais)
Todas as respostas são JSON. Erros retornam HTTP 4xx/5xx com `detail`.

- GET `/status`
  - Status básico e presença de chaves
  - Ex.: `{ "ok": true, "has_results": bool, "has_keys": bool }`

- GET `/summary`
  - Sumário da última análise (executa uma se necessário)
  - Inclui: `price`, `indicators`, `ml_predictions`, `risk_summary`

- GET `/analysis?tf=1h`
  - Análise detalhada para o timeframe (`15m`, `1h`, `4h`, `1d`)
  - Campos: `timeframe`, `price`, `indicators`, `ml_predictions`, `decision`, `confidence`, `take_profit`, `stop_loss`, `entry_hour`, `volatility_pct`, `sentiment_score`, `series` (time, close, sma20, sma50, sma200, bb_*), `df_tail`

- GET `/orders/open?symbol=BTCUSDT`
  - Ordens abertas do símbolo
  - Ex.: `{ "symbol": "BTCUSDT", "orders": [ ... ] }`

- GET `/config`
  - Lê parâmetros de configuração (`auto_trade_min_confidence`, `auto_trade_min_ml_score`, `min_notional`)

- POST `/config`
  - Atualiza configurações; body ex.: `{ "auto_trade_min_confidence": 0.7, "auto_trade_min_ml_score": 0.6, "min_notional": 10 }`

- POST `/keys`
  - Persiste chaves da Binance (criptografadas no SQLite); body: `{ "api_key": "...", "api_secret": "..." }`

- POST `/analyze`
  - Executa análise imediata e atualiza o cache interno; resp.: `{ "ok": true, "result": { ... } }`

- POST `/sync`
  - Sincroniza/resumo de conta no gerenciador de risco; resp.: `{ "ok": true }`

## Serviço em Background
- `TradingService` pode manter um loop periódico; ligue com `ENABLE_BACKGROUND_SERVICE=1` e ajuste `BACKGROUND_INTERVAL`
- Em produção, prefira rodar `worker_service.py` como processo/serviço separado

## Banco de Dados
- SQLite em `DB_PATH` (padrão `btc_trader.db`)
- Chaves da Binance criptografadas com Fernet; configure `ENCRYPTION_KEY`/`ENCRYPTION_KEY_FILE`
- Tabelas: `btc_prices`, `trading_decisions`, `user_config`, `news_cache`, `price_cache`, `ml_metrics`, `account_snapshot`

## Logs e Observabilidade
- Logs em `logs/` (quando configurado) via `Logger.py`
- Acompanhe falhas do conector (circuit breaker), latência e erros 5xx

## Exemplos de cURL
- `curl "http://localhost:8000/status"`
- `curl "http://localhost:8000/summary"`
- `curl "http://localhost:8000/analysis?tf=1h"`
- `curl -X POST "http://localhost:8000/keys" -H "Content-Type: application/json" -d "{\"api_key\":\"...\",\"api_secret\":\"...\"}"`
- `curl -X POST "http://localhost:8000/analyze"`

