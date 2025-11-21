# Documentação do Projeto CryptoVision

## Sobre o Projeto
- Backend em FastAPI que analisa o mercado de BTCUSDT usando indicadores técnicos, previsões de ML e (opcional) sentimento de notícias.
- Persistência leve em SQLite com chaves da Binance criptografadas.
- Pode rodar com loop em background ou com um worker dedicado.

## O que o Projeto Entrega
- Endpoints REST:
  - `GET /status`: status básico e presença de chaves
  - `GET /summary`: sumário da última análise (preço, indicadores, ML, risco)
  - `GET /analysis?tf=1h`: análise detalhada por timeframe (séries p/ gráficos)
  - `GET /orders/open?symbol=BTCUSDT`: ordens abertas
  - `GET /config` e `POST /config`: leitura/atualização de parâmetros (limiar de confiança etc.)
  - `POST /keys`: salva chaves Binance (criptografadas no SQLite)
  - `POST /analyze`: executa análise imediata
  - `POST /sync`: sincroniza/resumo de conta no gerenciador de risco
- Recursos internos:
  - Indicadores (RSI, MACD, SMAs, Bandas de Bollinger, ATR)
  - Previsões de ML (módulo `ml_model.py`)
  - Gerenciamento de risco (`RiskManager.py`)
  - Sentimento de notícias com VADER (opcional) via `News_Worker.py`
  - Logs em `logs/` quando configurado

## Como Rodar
- Pré‑requisitos: Python 3.11+, pip, venv.
- Passos (Windows PowerShell):
  - `python -m venv .venv`
  - `.\.venv\Scripts\Activate.ps1`
  - `pip install -r requirements.txt`
  - `uvicorn api.app:app --reload --port 8000`
- Passos (Linux/Mac):
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
  - `uvicorn api.app:app --reload --port 8000`
- Documentação interativa: `GET /docs` (Swagger) e `GET /redoc`.

### Rodando o Worker (opcional)
- Em produção, recomenda-se separar o loop de análise:
  - `python worker_service.py`
- Ou ative o loop interno no app definindo `ENABLE_BACKGROUND_SERVICE=1` e ajustando `BACKGROUND_INTERVAL`.

### Docker / Docker Compose
- Subir serviços API e Worker:
  - `docker compose up -d --build api worker`
- Observação: o serviço `web` no compose espera uma pasta `./web` (frontend). Se não estiver presente, suba apenas `api` e `worker`.
- Variáveis úteis: `DB_PATH`, `ALLOW_ORIGINS`, `BACKGROUND_INTERVAL`, `BINANCE_API_KEY`, `BINANCE_API_SECRET`.

## Dependências
- Instale via `pip install -r requirements.txt`.
- Para reproduzir exatamente o ambiente atual: `pip install -r requirements.lock.txt`.
- NLTK (VADER) baixa o corpus `vader_lexicon` na primeira execução. Se precisar baixar manualmente:
  - `python -c "import nltk; nltk.download('vader_lexicon')"`
- Veja `dependencias.txt` para passo a passo e snapshot completo do ambiente.

## Variáveis de Ambiente Importantes
- `BINANCE_API_KEY`, `BINANCE_API_SECRET` — chaves da Binance (opcional, recomendado)
- `ALLOW_ORIGINS` — CORS, padrão `*`
- `DB_PATH` — caminho do SQLite (padrão `btc_trader.db`)
- `ENCRYPTION_KEY_FILE` ou `ENCRYPTION_KEY` — criptografia das chaves no banco
- `ENABLE_BACKGROUND_SERVICE`, `BACKGROUND_INTERVAL` — loop interno de análise

## Observações e Próximos Passos
- Proteja endpoints administrativos (`/keys`, `/config`, `/sync`, `/analyze`) com rede/ACL ou autenticação no gateway.
- Para frontend, o `docker-compose.yml` espera uma pasta `web` (não inclusa aqui). Se desejar, integre um cliente web apontando para `http://localhost:8000`.
