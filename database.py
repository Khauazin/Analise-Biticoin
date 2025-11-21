import os
import json
import time
import sqlite3
import traceback
from datetime import datetime

import pandas as pd
from cryptography.fernet import Fernet

from Logger import iniciar_logger

# Permite configurar via variáveis de ambiente em produção
DATABASE_PATH = os.getenv("DB_PATH", "btc_trader.db")
ENCRYPTION_KEY_FILE = os.getenv("ENCRYPTION_KEY_FILE", "encryption_key.key")


def get_or_create_key():
    """Obtém a chave de criptografia.
    Prioriza variável de ambiente ENCRYPTION_KEY (base64 urlsafe). Se ausente,
    lê/escreve do arquivo ENCRYPTION_KEY_FILE (idempotente).
    """
    env_key = os.getenv("ENCRYPTION_KEY")
    if env_key:
        try:
            return env_key.encode()
        except Exception:
            pass
    if os.path.exists(ENCRYPTION_KEY_FILE):
        with open(ENCRYPTION_KEY_FILE, "rb") as f:
            return f.read()
    key = Fernet.generate_key()
    try:
        with open(ENCRYPTION_KEY_FILE, "wb") as f:
            f.write(key)
    except Exception:
        pass
    return key


cipher = Fernet(get_or_create_key())
logger = iniciar_logger("backend-db")


class DatabaseManager:
    @staticmethod
    def initialize_database():
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                conn.row_factory = sqlite3.Row
                c = conn.cursor()

                c.execute('''
                    CREATE TABLE IF NOT EXISTS btc_prices (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        price REAL NOT NULL,
                        volume REAL,
                        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                c.execute('''
                    CREATE TABLE IF NOT EXISTS trading_decisions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        decision TEXT NOT NULL,
                        price REAL NOT NULL,
                        confidence REAL,
                        indicators TEXT,
                        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                c.execute('''
                    CREATE TABLE IF NOT EXISTS user_config (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                ''')

                c.execute('''
                    CREATE TABLE IF NOT EXISTS news_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cache_key TEXT UNIQUE,
                        data TEXT,
                        timestamp REAL,
                        ttl REAL
                    )
                ''')

                c.execute('''
                    CREATE TABLE IF NOT EXISTS price_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT,
                        price REAL,
                        volume REAL,
                        timestamp REAL,
                        ttl REAL
                    )
                ''')

                c.execute('''
                    CREATE TABLE IF NOT EXISTS ml_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timeframe TEXT,
                        accuracy REAL,
                        mean_return REAL,
                        risk_adjusted REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                c.execute('''
                    CREATE TABLE IF NOT EXISTS account_snapshot (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        exposure_notional REAL,
                        open_positions_count INTEGER,
                        pnl_unrealized REAL,
                        positions_json TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                c.execute(
                    'CREATE INDEX IF NOT EXISTS idx_btc_prices_timestamp '
                    'ON btc_prices (timestamp)'
                )
                c.execute(
                    'CREATE INDEX IF NOT EXISTS idx_decisions_timestamp '
                    'ON trading_decisions (timestamp)'
                )
                c.execute(
                    'CREATE INDEX IF NOT EXISTS idx_ml_metrics_time '
                    'ON ml_metrics (created_at)'
                )
                c.execute(
                    'CREATE INDEX IF NOT EXISTS idx_account_snapshot_time '
                    'ON account_snapshot (created_at)'
                )
                conn.commit()
                logger.info("Banco de dados inicializado com sucesso")
                return True
        except Exception as e:
            logger.error(f"Erro ao inicializar DB: {e}")
            traceback.print_exc()
            return False

    # -------------------------
    # API keys (encrypted)
    # -------------------------
    @staticmethod
    def save_api_keys(api_key: str, api_secret: str) -> bool:
        try:
            enc_k = cipher.encrypt(api_key.encode()).decode()
            enc_s = cipher.encrypt(api_secret.encode()).decode()
            with sqlite3.connect(DATABASE_PATH) as conn:
                c = conn.cursor()
                c.execute(
                    "INSERT OR REPLACE INTO user_config "
                    "(key, value) VALUES (?, ?)",
                    ("binance_api_key", enc_k),
                )
                c.execute(
                    "INSERT OR REPLACE INTO user_config "
                    "(key, value) VALUES (?, ?)",
                    ("binance_api_secret", enc_s),
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Erro salvando chaves API: {e}")
            return False

    @staticmethod
    def get_api_keys():
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                c.execute(
                    "SELECT value FROM user_config WHERE key=?",
                    ("binance_api_key",),
                )
                r1 = c.fetchone()
                c.execute(
                    "SELECT value FROM user_config WHERE key=?",
                    ("binance_api_secret",),
                )
                r2 = c.fetchone()
                api_k = (
                    cipher.decrypt(r1['value'].encode()).decode()
                    if r1 and r1['value']
                    else ""
                )
                api_s = (
                    cipher.decrypt(r2['value'].encode()).decode()
                    if r2 and r2['value']
                    else ""
                )
                return api_k, api_s
        except Exception as e:
            logger.error(f"Erro lendo chaves API: {e}")
            return "", ""

    # -------------------------
    # Simple config
    # -------------------------
    @staticmethod
    def set_config(key: str, value: str) -> bool:
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                c = conn.cursor()
                c.execute(
                    "INSERT OR REPLACE INTO user_config "
                    "(key, value) VALUES (?, ?)",
                    (key, value),
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"set_config error {key}: {e}")
            return False

    @staticmethod
    def get_config_float(key: str, default: float) -> float:
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                c.execute("SELECT value FROM user_config WHERE key=?", (key,))
                r = c.fetchone()
                if not r:
                    return float(default)
                return float(r['value'])
        except Exception:
            return float(default)

    # -------------------------
    # BTC price storage
    # -------------------------
    @staticmethod
    def insert_btc_price(price: float, volume: float | None = None) -> bool:
        """Persiste o preço atual no banco para uso de fallback."""
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                c = conn.cursor()
                c.execute(
                    "INSERT INTO btc_prices (price, volume, timestamp) "
                    "VALUES (?, ?, ?)",
                    (
                        float(price),
                        volume,
                        datetime.utcnow().isoformat(timespec="seconds"),
                    ),
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"insert_btc_price error: {e}")
            return False

    @staticmethod
    def get_recent_btc_prices(limit: int = 10) -> list[float]:
        """Retorna os preços mais recentes.
        Útil para fallback de cotação.
        """
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                c.execute(
                    "SELECT price FROM btc_prices "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (int(limit),),
                )
                rows = c.fetchall()
                return [float(r["price"]) for r in rows]
        except Exception as e:
            logger.error(f"get_recent_btc_prices error: {e}")
            return []

    @staticmethod
    def get_historical_btc_prices(days: int = 30, interval: str = "1h"):
        """Retorna candles agregados a partir dos preços salvos localmente."""
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                df = pd.read_sql_query(
                    "SELECT price, volume, timestamp "
                    "FROM btc_prices "
                    "WHERE timestamp >= datetime('now', ?)",
                    conn,
                    params=(f"-{int(days)} days",),
                )
            if df.empty:
                return None

            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
            if df.empty:
                return None

            # Converter frequências da Binance para pandas
            freq_map = {
                "1m": "1min",
                "3m": "3min",
                "5m": "5min",
                "15m": "15min",
                "30m": "30min",
                "1h": "1h",
                "2h": "2h",
                "4h": "4h",
                "6h": "6h",
                "8h": "8h",
                "12h": "12h",
                "1d": "1d",
            }
            freq = freq_map.get(interval.lower(), interval)

            df = df.set_index("timestamp")
            resampled = df.resample(freq).agg(
                {
                    "price": ["first", "max", "min", "last"],
                    "volume": "sum",
                }
            )
            resampled = resampled.dropna(
                subset=[("price", "first"), ("price", "last")]
            )
            if resampled.empty:
                return None

            resampled.columns = ["open", "high", "low", "close", "volume"]
            resampled = resampled.reset_index()
            return resampled
        except Exception as e:
            logger.error(f"get_historical_btc_prices error: {e}")
            return None

    # -------------------------
    # Caches
    # -------------------------
    @staticmethod
    def save_news_cache(cache_key, data, ttl=600):
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                c = conn.cursor()
                c.execute(
                    "INSERT OR REPLACE INTO news_cache "
                    "(cache_key, data, timestamp, ttl) VALUES (?, ?, ?, ?)",
                    (cache_key, json.dumps(data), time.time(), float(ttl)),
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"save_news_cache: {e}")
            return False

    @staticmethod
    def get_news_cache(cache_key):
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                now = time.time()
                c.execute(
                    "SELECT data, timestamp, ttl "
                    "FROM news_cache WHERE cache_key=?",
                    (cache_key,),
                )
                r = c.fetchone()
                if r and (now - r['timestamp']) < r['ttl']:
                    return json.loads(r['data'])
                if r:
                    c.execute(
                        "DELETE FROM news_cache WHERE cache_key=?",
                        (cache_key,),
                    )
                    conn.commit()
                return None
        except Exception as e:
            logger.error(f"get_news_cache: {e}")
            return None

    @staticmethod
    def save_price_cache(symbol, price, volume=None, ttl=60):
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                c = conn.cursor()
                c.execute(
                    "INSERT OR REPLACE INTO price_cache "
                    "(symbol, price, volume, timestamp, ttl) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (symbol, float(price), volume, time.time(), float(ttl)),
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"save_price_cache: {e}")
            return False

    @staticmethod
    def get_price_cache(symbol):
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                now = time.time()
                c.execute(
                    "SELECT price, volume, timestamp, ttl FROM price_cache "
                    "WHERE symbol=?",
                    (symbol,),
                )
                r = c.fetchone()
                if r and (now - r['timestamp']) < r['ttl']:
                    return {'price': r['price'], 'volume': r['volume']}
                if r:
                    c.execute(
                        "DELETE FROM price_cache WHERE symbol=?",
                        (symbol,),
                    )
                    conn.commit()
                return None
        except Exception as e:
            logger.error(f"get_price_cache: {e}")
            return None

    # -------------------------
    # ML metrics & account snapshots & decisions
    # -------------------------
    @staticmethod
    def insert_decision(
        decision: str,
        price: float,
        confidence: float,
        indicators: dict,
    ) -> bool:
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                c = conn.cursor()
                c.execute(
                    "INSERT INTO trading_decisions "
                    "(decision, price, confidence, indicators) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        str(decision),
                        float(price),
                        float(confidence or 0.0),
                        json.dumps(indicators or {}),
                    ),
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"insert_decision error: {e}")
            return False

    @staticmethod
    def get_recent_decisions(limit: int = 10):
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                c.execute(
                    "SELECT decision, price, confidence, "
                    "indicators, timestamp "
                    "FROM trading_decisions "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (int(limit),),
                )
                rows = c.fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"get_recent_decisions error: {e}")
            return []

    # -------------------------
    # ML metrics & account snapshots
    # -------------------------
    @staticmethod
    def save_ml_metrics(timeframe, metrics: dict):
        try:
            if not isinstance(metrics, dict):
                return False
            acc = float(metrics.get('accuracy', 0.0) or 0.0)
            mret = float(metrics.get('mean_return', 0.0) or 0.0)
            rsk = float(metrics.get('risk_adjusted', 0.0) or 0.0)
            with sqlite3.connect(DATABASE_PATH) as conn:
                c = conn.cursor()
                c.execute(
                    "INSERT INTO ml_metrics "
                    "(timeframe, accuracy, mean_return, risk_adjusted) "
                    "VALUES (?, ?, ?, ?)",
                    (str(timeframe or ''), acc, mret, rsk),
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"save_ml_metrics: {e}")
            return False

    @staticmethod
    def save_account_snapshot(
        exposure_notional: float,
        open_positions_count: int,
        pnl_unrealized: float,
        positions_json: str,
    ) -> bool:
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                c = conn.cursor()
                c.execute(
                    "INSERT INTO account_snapshot "
                    "(exposure_notional, open_positions_count, "
                    "pnl_unrealized, positions_json) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        float(exposure_notional or 0.0),
                        int(open_positions_count or 0),
                        float(pnl_unrealized or 0.0),
                        positions_json,
                    ),
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"save_account_snapshot: {e}")
            return False
