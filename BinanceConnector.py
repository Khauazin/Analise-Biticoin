from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import time
import os
from database import DatabaseManager
from Logger import iniciar_logger
from cache_manager import cache_manager
from alert_system import alert_system

# Inicializar logger para backend
logger = iniciar_logger("backend")


class BinanceConnector:
    """Conector para a API da Binance
    com cache e tratamento robusto de erros"""

    def __init__(self, api_key: str = "", api_secret: str = "") -> None:
        """Inicializa o conector da Binance,
        tentando buscar chaves do banco se necessário."""
        self.api_key = api_key
        self.api_secret = api_secret
        self._client = None
        self._testnet = False  # AMBIENTE REAL POR PADRÃO
        self._circuit_breaker_failures = 0
        self._circuit_breaker_timeout = 300  # 5 minutos
        self._last_failure_time = 0
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Inicializa o cliente da Binance,
        carregando as credenciais se necessário."""
        if not self.api_key or not self.api_secret:
            self.api_key, self.api_secret = DatabaseManager.get_api_keys()
            if not self.api_key or not self.api_secret:
                # Fallback para variáveis de ambiente em produção
                self.api_key = os.getenv("BINANCE_API_KEY", self.api_key or "")
                self.api_secret = os.getenv("BINANCE_API_SECRET",
                                            self.api_secret or "")

        try:
            self._client = Client(self.api_key, self.api_secret,
                                  requests_params={"timeout": 10})
            # Ajustar endpoint conforme ambiente
            try:
                if self._testnet:
                    self._client.API_URL = 'https://testnet.binance.vision/api'
                else:
                    self._client.API_URL = 'https://api.binance.com/api'
            except Exception:
                pass
            print("Binance client initialized successfully")
            logger.info("Binance client initialized successfully")
        except Exception as e:
            print(f"Error initializing Binance client: {e}")
            logger.error(f"Error initializing Binance client: {e}")
            self._client = None

    @property
    def client(self) -> Client:
        """Garante que o cliente da Binance está inicializado antes do uso."""
        if self._client is None:
            self._initialize_client()
        return self._client

    def get_current_price(self, symbol: str = "BTCUSDT") -> float | None:
        """Obtém o preço atual do ativo especificado."""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            if ticker and 'price' in ticker:
                price = float(ticker['price'])
            else:
                logger.warning(f"No price data retrieved for {symbol}")
                return None
            if price:
                logger.info(f"Retrieved current price for {symbol}: {price}")
            return price
        except BinanceAPIException as e:
            print(f"Binance API Error: {e}")
            logger.error(
                f"Binance API Error getting current price for {symbol}: {e}")
        except Exception as e:
            print(f"Error getting current price: {e}")
            logger.error(f"Error getting current price for {symbol}: {e}")
        return None

    def get_historical_klines(self, symbol: str = "BTCUSDT",
                              interval: str = "1h",
                              limit: int = 500) -> pd.DataFrame | None:
        """Obtém dados históricos de preços
        da Binance e retorna um DataFrame."""
        try:
            klines = self.client.get_historical_klines(
                symbol,
                interval,
                limit=limit,
            )
            if not klines:
                logger.warning(
                    f"No historical klines data retrieved for, {symbol}")
                return None

            # Converter dados para DataFrame
            columns = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                'ignore'
            ]
            df = pd.DataFrame(klines, columns=columns)

            # Ajustar tipos de dados
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = df[[
                'open', 'high', 'low', 'close', 'volume']].astype(float)

            logger.info(f"Retrieved {len(df)} historical klines for {symbol}")
            return df
        except BinanceAPIException as e:
            print(f"Binance API Error: {e}")
            logger.error(
                "Binance API Error getting historical klines "
                f"for {symbol}: {e}")
        except Exception as e:
            print(f"Error getting historical klines: {e}")
            logger.error(f"Error getting historical klines for {symbol}: {e}")
        return None

    def get_24h_stats(self, symbol: str = "BTCUSDT") -> dict | None:
        """Obtém estatísticas das últimas 24 horas do ativo especificado."""
        try:
            stats = self.client.get_ticker(symbol=symbol)
            logger.info(f"Retrieved 24h stats for {symbol}")
            return stats
        except BinanceAPIException as e:
            print(f"Binance API Error: {e}")
            logger.error(
                f"Binance API Error getting 24h stats for {symbol}: {e}")
        except Exception as e:
            print(f"Error getting 24h stats: {e}")
            logger.error(f"Error getting 24h stats for {symbol}: {e}")
        return None

    def _check_circuit_breaker(self) -> bool:
        """Verifica se o circuit breaker está ativo."""
        if self._circuit_breaker_failures >= 5:
            if time.time() - self._last_failure_time < \
                    self._circuit_breaker_timeout:
                logger.warning("Circuit breaker active - skipping API call")
                return False
            else:
                # Reset circuit breaker
                self._circuit_breaker_failures = 0
                logger.info("Circuit breaker reset")
        return True

    def _handle_api_failure(self, error: Exception, operation: str) -> None:
        """Trata falhas de API com circuit breaker e alertas."""
        self._circuit_breaker_failures += 1
        self._last_failure_time = time.time()
        logger.error(f"API failure in {operation}: {error}")
        alert_system.alert_api_failure("Binance", f"{operation}: {str(error)}")

    def get_current_price_cached(
            self,
            symbol: str = "BTCUSDT",
            ttl: int = 60) -> float | None:
        """Obtém preço atual com cache."""
        cache_key = f"binance_price_{symbol}"
        cached_price = cache_manager.get(cache_key)
        if cached_price is not None:
            return cached_price

        if not self._check_circuit_breaker():
            return None

        try:
            price = self.get_current_price(symbol)
            if price:
                cache_manager.set(cache_key, price, ttl)
            return price
        except Exception as e:
            self._handle_api_failure(e, f"get_current_price({symbol})")
            return None

    def get_historical_klines_cached(
            self,
            symbol: str = "BTCUSDT",
            interval: str = "1h",
            limit: int = 500,
            ttl: int = 300) -> pd.DataFrame | None:
        """Obtém dados históricos com cache."""
        cache_key = f"binance_klines_{symbol}_{interval}_{limit}"
        cached_data = cache_manager.get(cache_key)
        if cached_data is not None:
            return cached_data

        if not self._check_circuit_breaker():
            return None

        try:
            df = self.get_historical_klines(symbol, interval, limit)
            if df is not None and not df.empty:
                cache_manager.set(cache_key, df, ttl)
            return df
        except Exception as e:
            self._handle_api_failure(
                e, f"get_historical_klines({symbol}, {interval}, {limit})")
            return None

    def get_24h_stats_cached(
            self,
            symbol: str = "BTCUSDT",
            ttl: int = 300) -> dict | None:
        """Obtém estatísticas 24h com cache."""
        cache_key = f"binance_stats_{symbol}"
        cached_stats = cache_manager.get(cache_key)
        if cached_stats is not None:
            return cached_stats

        if not self._check_circuit_breaker():
            return None

        try:
            stats = self.get_24h_stats(symbol)
            if stats:
                cache_manager.set(cache_key, stats, ttl)
            return stats
        except Exception as e:
            self._handle_api_failure(e, f"get_24h_stats({symbol})")
            return None

    # ============================
    # Trading helpers (Spot)
    # ============================
    def use_testnet(self, enable: bool = True) -> bool:
        """Ativa/desativa o endpoint Spot Testnet.
        Seguro para desenvolvimento.
        """
        try:
            self._testnet = bool(enable)
            if self._client is None:
                self._initialize_client()
            if self._client is not None:
                if self._testnet:
                    self._client.API_URL = 'https://testnet.binance.vision/api'
                    logger.info(
                        "Switched Binance client to SPOT testnet endpoint")
                else:
                    self._client.API_URL = 'https://api.binance.com/api'
                    logger.info(
                        "Switched Binance client to SPOT live endpoint")
                return True
            return False
        except Exception as e:
            self._handle_api_failure(e, "use_testnet")
            return False

    # ============================
    # Exchange filters & rounding
    # ============================
    def _get_symbol_filters(self, symbol: str) -> dict:
        try:
            info = self.client.get_symbol_info(symbol.upper())
            flt = {}
            for f in info.get('filters', []):
                flt[f['filterType']] = f
            return flt
        except Exception as e:
            logger.error(f"Error fetching symbol filters for {symbol}: {e}")
            return {}

    @staticmethod
    def _round_step(value: float, step: float) -> float:
        import math
        if step <= 0:
            return float(value)
        return math.floor(float(value) / step) * step

    def _sanitize_order(
            self,
            symbol: str,
            side: str,
            quantity: float,
            price: float | None = None) -> tuple[float, float | None]:
        flt = self._get_symbol_filters(symbol)
        qty = float(quantity)
        pr = float(price) if price is not None else None
        # LOT_SIZE
        lot = flt.get('LOT_SIZE')
        if lot:
            step = float(lot.get('stepSize', 0))
            min_qty = float(lot.get('minQty', 0))
            if step:
                qty = self._round_step(qty, step)
            if qty < min_qty:
                qty = 0.0
        # PRICE_FILTER
        pf = flt.get('PRICE_FILTER')
        if pr is not None and pf:
            tick = float(pf.get('tickSize', 0))
            min_price = float(pf.get('minPrice', 0))
            if tick:
                pr = self._round_step(pr, tick)
            if pr < min_price:
                pr = min_price
        # MIN_NOTIONAL
        mn = flt.get('MIN_NOTIONAL')
        if mn and pr is not None:
            min_notional = float(mn.get('minNotional', 0))
            if (qty * pr) < min_notional:
                qty = 0.0
        return qty, pr

    def place_order(self,
                    symbol: str,
                    side: str,
                    quantity: float,
                    order_type: str = 'MARKET',
                    price: float | None = None,
                    time_in_force: str = 'GTC') -> dict | None:
        """
        Coloca uma ordem Spot básica (MARKET ou LIMIT).
        Para LIMIT informe price.
        """
        try:
            if not self._check_circuit_breaker():
                return None
            s = symbol.upper()
            sd = side.upper()
            ot = order_type.upper()
            qty = float(quantity)
            if sd not in ('BUY', 'SELL'):
                raise ValueError('side must be BUY or SELL')
            if ot == 'MARKET':
                qty, _ = self._sanitize_order(s, sd, qty, None)
                if qty <= 0:
                    raise ValueError(
                        'quantity below LOT/MIN_NOTIONAL after rounding')
                resp = self.client.create_order(
                    symbol=s, side=sd, type='MARKET', quantity=qty)
                logger.info(f"Placed MARKET order {sd} {qty} {s}")
                return resp
            elif ot == 'LIMIT':
                if price is None:
                    raise ValueError('price required for LIMIT order')
                qty, pr = self._sanitize_order(s, sd, qty, float(price))
                if qty <= 0:
                    raise ValueError(
                        'quantity below LOT/MIN_NOTIONAL after rounding')
                resp = self.client.create_order(
                    symbol=s,
                    side=sd,
                    type='LIMIT',
                    timeInForce=time_in_force,
                    quantity=qty,
                    price=f"{pr:.8f}")
                logger.info(f"Placed LIMIT order {sd} {qty} {s} @ {pr}")
                return resp
            else:
                raise ValueError(
                    'Unsupported order_type (use MARKET or LIMIT)')
        except BinanceAPIException as e:
            self._handle_api_failure(
                e, f"place_order({symbol},{side},{order_type})")
            return None
        except Exception as e:
            self._handle_api_failure(
                e, f"place_order({symbol},{side},{order_type})")
            return None

    def place_oco_sell(
            self,
            symbol: str,
            quantity: float,
            take_profit: float,
            stop_loss: float,
            stop_limit_offset: float = 0.0005) -> dict | None:
        """
        Cria uma ordem OCO de venda (só SELL no Spot).
        stop_limit_price = stop_loss * (1 - offset).
        """
        try:
            if not self._check_circuit_breaker():
                return None
            s = symbol.upper()
            qty, _ = self._sanitize_order(s, 'SELL', float(quantity), None)
            tp = float(take_profit)
            sl = float(stop_loss)
            sl_limit = max(sl * (1 - float(stop_limit_offset)), 0.0)
            resp = self.client.create_oco_order(
                symbol=s,
                side='SELL',
                quantity=qty,
                price=f"{tp:.8f}",
                stopPrice=f"{sl:.8f}",
                stopLimitPrice=f"{sl_limit:.8f}",
                stopLimitTimeInForce='GTC'
            )
            logger.info(f"Placed OCO SELL {qty} {s} tp={tp} sl={sl}")
            return resp
        except BinanceAPIException as e:
            self._handle_api_failure(e, f"place_oco_sell({symbol})")
            return None
        except Exception as e:
            self._handle_api_failure(e, f"place_oco_sell({symbol})")
            return None

    def get_order_status(self, symbol: str, order_id: int) -> dict | None:
        try:
            s = symbol.upper()
            resp = self.client.get_order(symbol=s, orderId=int(order_id))
            return resp
        except Exception as e:
            self._handle_api_failure(
                e, f"get_order_status({symbol},{order_id})")
            return None

    def cancel_order(self, symbol: str, order_id: int) -> dict | None:
        try:
            s = symbol.upper()
            resp = self.client.cancel_order(symbol=s, orderId=int(order_id))
            logger.info(f"Canceled order {order_id} on {s}")
            return resp
        except Exception as e:
            self._handle_api_failure(e, f"cancel_order({symbol},{order_id})")
            return None

    def get_open_orders(self, symbol: str = "BTCUSDT") -> list[dict] | None:
        try:
            s = symbol.upper()
            resp = self.client.get_open_orders(symbol=s)
            return resp or []
        except Exception as e:
            self._handle_api_failure(e, f"get_open_orders({symbol})")
            return None
