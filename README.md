# 🪙 Bitcoin Trader — Sistema Profissional de Análise e Trading

Sistema completo de análise de mercado para Bitcoin com Machine Learning, análise técnica multi-timeframe, gestão de risco avançada, análise de sentimento de notícias e interface gráfica em PyQt5.

---

## 🧠 Visão Geral da Arquitetura

O projeto é dividido em camadas bem definidas que se comunicam entre si:

```
┌─────────────────────────────────────────────┐
│              Interface (PyQt5)              │
│         Dashboard  •  OrdersWidget         │
│         APIKeyDialog  •  Configurações      │
└───────────────────┬─────────────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│           Pipeline de Análise               │
│             MarketAnalyzer                  │
│  16 etapas: Dados → Regime → Setup → MTF   │
│       → Risco → ML → Macro → Decisão       │
└────┬──────────┬──────────┬──────────────────┘
     │          │          │
┌────▼───┐ ┌───▼────┐ ┌───▼──────────────────┐
│Binance │ │MLModel │ │     RiskManager       │
│Connect.│ │MLP NN  │ │ Position sizing       │
│Circuit │ │Regime  │ │ Kill switch / Cooldown│
│Breaker │ │OHLCV   │ │ Backtest engine       │
└────┬───┘ └───┬────┘ └──────────────────────┘
     │         │
┌────▼─────────▼──────────────────────────────┐
│            Infraestrutura                   │
│   DatabaseManager (SQLite + criptografia)  │
│   CacheManager (TTL in-memory)             │
│   AlertSystem (SMTP e-mail)                │
│   Logger (diário + console + retenção)     │
│   NewsWorker (Alpha Vantage/CryptoPanic)   │
└─────────────────────────────────────────────┘
```

---

## ✨ Funcionalidades

**Análise de Mercado (pipeline de 16 etapas):**
- Verificação de saúde dos dados (Data Health Score)
- Classificação de regime de mercado: TREND UP/DOWN, RANGE, HIGH_VOL, LOW_LIQUIDITY
- Detecção de setups: Pullback, VWAP Rejection, Break + Retest, Liquidity Sweep, padrões gráficos
- Análise Multi-Timeframe (5m, 15m, 1h, 4h, 1d) com detecção de confluência
- Cálculo de Risk/Reward com stops dinâmicos baseados em ATR, S/R e Volume Profile

**Machine Learning:**
- Modelo MLP (sklearn) treinado por regime (TREND, RANGE, HIGH_VOL) e por timeframe
- Features: RSI, MACD, ATR, ADX, VWAP, Volume Profile (POC), retornos defasados
- Modelo de qualidade de trade (outcome model) treinado em resultados reais
- Retreinamento automático com detecção de mudança de regime por volatilidade
- Walk-forward validation integrada

**Gestão de Risco (Prop Firm style):**
- Position sizing dinâmico por risco percentual da conta
- Kill switch diário por drawdown máximo (configurável)
- Cooldown anti-revenge trading após stop
- Tracking de exposição e correlação entre ativos
- Backtest engine walk-forward integrado
- Sincronização com conta Spot da Binance via FIFO cost basis

**Análise de Notícias e Sentimento:**
- Suporte a 3 fontes: Alpha Vantage, CryptoPanic e NewsAPI (com fallback automático)
- Análise de sentimento via VADER (NLP cripto-otimizado) com fallback para TextBlob
- Integração do score de sentimento como sinal macro no pipeline de decisão
- Calendário econômico de alto impacto via Alpha Vantage

**Infraestrutura:**
- Banco de dados SQLite com WAL mode e índices otimizados
- Chaves de API criptografadas em repouso com Fernet (cryptography)
- Cache in-memory com TTL configurável por tipo de dado
- Sistema de alertas por e-mail (SMTP) para falhas críticas
- Logs diários rotativos com retenção configurável (padrão 15 dias)
- Circuit breaker na API da Binance (5 falhas → pausa 5 min)
- Worker background para análise automática periódica

---

## 🗂️ Estrutura do Projeto

```
bitcoin_trader/
├── Market_Analyzer.py      # Pipeline principal de análise (16 etapas)
├── BinanceConnector.py     # Conector Binance com circuit breaker e cache
├── ml_model.py             # Modelo ML (MLP) por regime e timeframe
├── RiskManager.py          # Gestão de risco avançada e backtest
├── News_Worker.py          # Notícias e análise de sentimento
├── database.py             # SQLite com criptografia das chaves API
├── TradingDecision.py      # Modelo de dados da decisão de trading
├── BitcoinPrice.py         # Modelo de dados de preço
├── alert_system.py         # Alertas por e-mail
├── cache_manager.py        # Cache in-memory com TTL
├── Logger.py               # Logger diário + retenção automática
├── worker_service.py       # Worker background de análise periódica
├── dashboard.py            # Interface principal PyQt5 (frontend)
├── OrdersWidget.py         # Widget de ordens abertas na Binance
├── APIKeyDialog.py         # Diálogo de configuração de chaves API
├── apply_api_keys.py       # Script para aplicar chaves via env vars
├── check_api_status.py     # Diagnóstico de chaves configuradas
├── inspect_db.py           # Inspeção do banco de dados
├── logs/                   # Logs gerados automaticamente
└── models/                 # Modelo ML serializado (ml_model.pkl)
```

---

## ⚙️ Configuração

### 1. Variáveis de Ambiente

Crie um arquivo `.env` ou exporte as variáveis no seu ambiente:

```env
# Binance
BINANCE_API_KEY=sua_api_key
BINANCE_API_SECRET=seu_api_secret
BINANCE_ENV=live           # ou "testnet" para modo de teste

# APIs de notícias (ao menos uma recomendada)
ALPHAVANTAGE_API_KEY=sua_chave
CRYPTOPANIC_API_KEY=sua_chave
NEWSAPI_KEY=sua_chave

# Configurações de trading
TRADING_ENABLED=0          # 1 para habilitar envio real de ordens
TRADING_PROFILE=conservative  # conservative | balanced | active

# ML
ENABLE_ML_TRAIN=1
ML_RETRAIN_INTERVAL_SEC=900
ML_OUTCOME_MIN_PROB=0.52

# Infraestrutura
DB_PATH=btc_trader.db
LOG_RETENTION_DAYS=15
BACKGROUND_INTERVAL=30

# Alertas por e-mail (opcional)
ALERT_EMAIL=seu@email.com
ALERT_PASSWORD=sua_senha_app
```

### 2. Aplicar chaves no banco de dados

```bash
python apply_api_keys.py
```

---

## 🚀 Como Executar

### Pré-requisitos

- Python 3.10+
- Dependências:

```bash
pip install pyqt5 python-binance pandas numpy scikit-learn joblib \
            cryptography requests textblob nltk python-dateutil
python -m nltk.downloader vader_lexicon
```

### Inicializar o banco de dados

```bash
python -c "from database import DatabaseManager; DatabaseManager.initialize_database()"
```

### Rodar a interface gráfica

```bash
python dashboard.py
```

### Rodar apenas o worker em background (sem interface)

```bash
python worker_service.py
```

### Verificar status das chaves configuradas

```bash
python check_api_status.py
```

---

## 🔬 Como Funciona o Pipeline de Decisão

Toda decisão de trading passa por 12 etapas sequenciais. Um bloqueio `HARD` em qualquer etapa resulta em `HOLD`:

```
Dados ✓ → Regime ✓ → Setup ✓ → MTF ✓ → Risco ✓ → ML ✓ → Decisão
```

**Bloqueadores por severidade:**

| Severidade | Efeito |
|------------|--------|
| `HARD`     | HOLD imediato, trade cancelado |
| `SOFT`     | Trade permitido com tamanho reduzido |
| `INFO`     | Registrado, sem impacto |

**Perfis de trading (`TRADING_PROFILE`):**

| Perfil | Comportamento |
|--------|--------------|
| `conservative` | Thresholds mais altos, bloqueia conflitos MTF |
| `balanced` | Thresholds moderados |
| `active` | Mais permissivo, aceita setups com conflitos leves |

---

## 🤖 Machine Learning

O sistema usa dois modelos independentes:

**Modelo de Direção** — Prevê UP / DOWN / NEUTRAL para o próximo horizonte de velas. Features incluem retornos defasados, RSI, MACD, ATR, ADX, VWAP distance, Volume Profile POC, regime de mercado. Treinado separadamente para cada timeframe e regime.

**Outcome Model** — Prevê probabilidade de WIN dado um setup específico. Treinado em resultados reais de trades encerrados (TP/SL hit ou timeout). Bloqueia trades com `prob_win < 0.52` (configurável).

O retreinamento é automático a cada 15 minutos (padrão) ou quando uma mudança de regime é detectada via variação de volatilidade > 50%.

---

## 🛡️ Segurança

- Chaves de API **nunca em texto claro** no banco — armazenadas com Fernet (AES-128-CBC)
- Chave de criptografia gerada localmente e salva em `encryption_key.key`, ou via variável de ambiente `ENCRYPTION_KEY`
- Trading desabilitado por padrão (`TRADING_ENABLED=0`) — envio real de ordens só ocorre quando explicitamente habilitado
- Suporte a Binance Testnet para desenvolvimento e testes

---

## 📊 Banco de Dados

O SQLite (`btc_trader.db`) contém as seguintes tabelas principais:

| Tabela | Conteúdo |
|--------|----------|
| `btc_prices` | Histórico de preços para fallback |
| `trading_decisions` | Todas as decisões geradas |
| `analysis_logs` | Snapshot completo de cada análise (auditoria) |
| `signal_outcomes` | Trades abertos/fechados com MFE, MAE e resultado |
| `user_config` | Chaves API criptografadas e configurações |
| `ml_metrics` | Métricas de cada treino do modelo |
| `safety_events` | Registros de kill switch e eventos críticos |

---

## 📄 Licença

MIT License — sinta-se à vontade para usar, modificar e distribuir.
