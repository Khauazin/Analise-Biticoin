import os
import json
import logging
import numpy as np
import pandas as pd
import time
import threading
import functools
from datetime import datetime
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeout,
)

# Binance e database
from database import DatabaseManager, DATABASE_PATH as DB_DEFAULT
from BinanceConnector import BinanceConnector
from TradingDecision import TradingDecision
from ml_model import MLModel
from Logger import iniciar_logger

# Inicializar logger para backend
backend_logger = iniciar_logger("backend")

# Constantes
# Em produção, use o caminho de DB configurado em database.DATABASE_PATH
DATABASE_PATH = DB_DEFAULT

# Executor global (usado pelo run_with_timeout)
_executor = ThreadPoolExecutor(max_workers=4)


def run_with_timeout(fn, *args, timeout=10, **kwargs):
    """Executa fn(*args, **kwargs) com timeout (segundos).
    Retorna (ok, result_or_error).
    """
    f = _executor.submit(functools.partial(fn, *args, **kwargs))
    try:
        res = f.result(timeout=timeout)
        return True, res
    except FuturesTimeout:
        try:
            f.cancel()
        except Exception:
            pass
        return False, f"timeout after {timeout}s"
    except Exception as e:
        return False, str(e)


class MarketAnalyzer:
    """Analisador de mercado Bitcoin com ML e indicadores tecnicos."""

    def __init__(self, database_path=DATABASE_PATH):
        self.db = DatabaseManager()
        self.binance = BinanceConnector()
        self._setup_logging()
        self.ml_model = MLModel()
        self.ml_trained = False

        # Cache e proteção para concorrencia
        self._cached_results = {}  # cache de resultados recentes
        self._lock = threading.Lock()

        self._last_api_call_time = 0
        self._api_call_interval = 60  # segundos

        if not os.path.exists(database_path):
            try:
                self.db.initialize_database()
                self.logger.info("Database initialized successfully.")
            except Exception as e:
                self.logger.exception("Error initializing database: %s", e)

        # Tentar carregar modelo salvo na inicialização
        if self.ml_model.load_model():
            self.ml_trained = True
            self.logger.info(
                "Modelo ML carregado com sucesso da inicialização."
            )
        else:
            self.logger.info(
                "Nenhum modelo salvo encontrado; treinamento necessario."
            )

    def _setup_logging(self):
        self.logger = logging.getLogger("MarketAnalyzer")
        # evita múltiplos handlers se instanciar variaveis vezes
        if not self.logger.handlers:
            self.logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(handler)
        self.logger.info("Logger configurado com sucesso!")

    # =========================
    # Obtenção de preços
    # =========================
    def get_current_price(self):
        """Obtém o Preço atual do Bitcoin da Binance com fallback."""
        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    current_price = self.binance.get_current_price()
                except Exception as e:
                    self.logger.debug(
                        f"Erro na chamada binance.get_current_price(): {e}"
                    )
                    current_price = None

                if current_price and current_price > 0:
                    try:
                        self.db.insert_btc_price(current_price)
                    except Exception as e:
                        self.logger.debug(
                            f"Não foi possivel inserir Preço no DB: {e}"
                        )
                    self.logger.info(
                        f"Preço atual obtido da Binance: "
                        f"{current_price:.2f} USDT"
                    )
                    return current_price

                self.logger.warning(
                    f"Tentativa {attempt+1}/{max_retries} "
                    f"falhou para obter Preço valido."
                )
                time.sleep(1)

            # fallback local
            self.logger.warning("Falha na API Binance, usando fallback local.")
            try:
                recent_prices = self.db.get_recent_btc_prices(limit=10) or []
            except Exception as e:
                self.logger.debug(f"Erro ao obter recent_prices do DB: {e}")
                recent_prices = []

            if recent_prices:
                valid_prices = [p for p in recent_prices if 20000 < p < 100000]
                if valid_prices:
                    fallback_price = float(np.mean(valid_prices))
                    self.logger.info(
                        "Preço de fallback (média local): "
                        f"{fallback_price:.2f} USDT"
                    )
                    return fallback_price

            base_price = 44000.0
            variation = np.clip(np.random.normal(0, 500), -1500, 1500)
            fallback_price = base_price + variation
            try:
                self.db.insert_btc_price(fallback_price)
            except Exception as e:
                # Registrar falha no salvamento do fallback, mas seguir com o
                # valor calculado
                self.logger.debug(f"Erro ao registrar preço de fallback: {e}")
            self.logger.info(
                "Retornando preço de fallback simulado: "
                f"{fallback_price:.2f} USDT"
            )
            return float(fallback_price)
        except Exception as e:
            self.logger.error(f"Erro ao obter Preço atual: {e}", exc_info=True)
            return None

    def get_historical_prices(self, days=30, interval="1h"):
        """Obtém Preços historicos com fallback."""
        try:
            max_retries = 3
            limit = 1000
            for attempt in range(max_retries):
                try:
                    df = self.binance.get_historical_klines(
                        interval=interval, limit=limit
                    )
                except Exception as e:
                    self.logger.debug(
                        f"Erro na chamada binance.get_historical_klines(): {e}"
                    )
                    df = None

                if df is not None and not df.empty:
                    self.logger.info("Dados historicos obtidos da Binance.")
                    return df
                self.logger.warning(
                    f"Tentativa {attempt+1}/{max_retries} "
                    "falhou para dados historicos."
                )
                time.sleep(1)

            df_local = self.db.get_historical_btc_prices(
                days=days, interval=interval
            )
            if df_local is not None and not df_local.empty:
                self.logger.info("Dados historicos obtidos do DB local.")
                return df_local

            self.logger.warning("Usando dados simulados.")
            return self._generate_simulated_data(days)
        except Exception as e:
            self.logger.error(
                f"Error getting historical prices: {e}", exc_info=True
            )
            return self._generate_simulated_data(days)

    def _generate_simulated_data(self, days):
        """Gera dados simulados."""
        try:
            base_price = 44000.0
            dates = pd.date_range(
                end=datetime.now(), periods=days * 24, freq="h"
            )
            np.random.seed(42)
            random_walk = np.random.normal(0, 100, size=len(dates)).cumsum()
            price_series = (
                base_price + random_walk + np.linspace(0, 2000, len(dates))
            )

            df = pd.DataFrame(
                {
                    "timestamp": dates,
                    "open": price_series,
                    "high": price_series
                    * (1 + np.random.uniform(0, 0.01, len(dates))),
                    "low": price_series
                    * (1 - np.random.uniform(0, 0.01, len(dates))),
                    "close": price_series,
                    "volume": np.random.uniform(100, 1000, len(dates)),
                }
            )
            return df
        except Exception as e:
            self.logger.error(
                f"Error generating simulated data: {e}", exc_info=True
            )
            return None

    # =========================
    # Indicadores tecnicos
    # =========================
    def calculate_technical_indicators(self, df):
        """Calcula indicadores tecnicos e retorna dicionario."""
        result = {}
        try:
            if df is None or len(df) < 200:
                return {"error": "Not enough data"}

            close = df["close"]

            # médias moveis
            result["sma_20"] = close.rolling(20).mean().iloc[-1]
            result["sma_50"] = close.rolling(50).mean().iloc[-1]
            result["sma_200"] = close.rolling(200).mean().iloc[-1]

            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            result["rsi"] = 100 - (100 / (1 + rs.iloc[-1]))

            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            result["macd"] = macd_line.iloc[-1]
            result["macd_signal"] = signal_line.iloc[-1]
            result["macd_histogram"] = (
                macd_line.iloc[-1] - signal_line.iloc[-1]
            )

            # Bollinger
            middle_band = close.rolling(20).mean()
            std_dev = close.rolling(20).std()
            result["bb_upper"] = (middle_band + 2 * std_dev).iloc[-1]
            result["bb_middle"] = middle_band.iloc[-1]
            result["bb_lower"] = (middle_band - 2 * std_dev).iloc[-1]

            # ATR
            high_low = df["high"] - df["low"]
            high_close_prev = (df["high"] - df["close"].shift()).abs()
            low_close_prev = (df["low"] - df["close"].shift()).abs()
            true_range = pd.concat(
                [high_low, high_close_prev, low_close_prev], axis=1
            ).max(axis=1)
            atr = true_range.rolling(14).mean()
            result["atr"] = atr.iloc[-1]

            # Stochastic
            low_14 = df["low"].rolling(14).min()
            high_14 = df["high"].rolling(14).max()
            stoch_k = 100 * (close - low_14) / (high_14 - low_14)
            stoch_d = stoch_k.rolling(3).mean()
            result["stoch_k"] = stoch_k.iloc[-1]
            result["stoch_d"] = stoch_d.iloc[-1]

            # Ichimoku Cloud
            tenkan_sen = (
                df["high"].rolling(9).max() + df["low"].rolling(9).min()
            ) / 2
            kijun_sen = (
                df["high"].rolling(26).max() + df["low"].rolling(26).min()
            ) / 2
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            senkou_span_b = (
                (df["high"].rolling(52).max() + df["low"].rolling(52).min())
                / 2
            ).shift(26)
            chikou_span = close.shift(-26)
            result["ichimoku_tenkan"] = tenkan_sen.iloc[-1]
            result["ichimoku_kijun"] = kijun_sen.iloc[-1]
            result["ichimoku_senkou_a"] = senkou_span_a.iloc[-1]
            result["ichimoku_senkou_b"] = senkou_span_b.iloc[-1]
            result["ichimoku_chikou"] = (
                chikou_span.iloc[-1]
                if not pd.isna(chikou_span.iloc[-1])
                else close.iloc[-1]
            )

            # Trend
            if result["sma_20"] > result["sma_50"] > result["sma_200"]:
                result["trend"] = "BULLISH"
            elif result["sma_20"] < result["sma_50"] < result["sma_200"]:
                result["trend"] = "BEARISH"
            else:
                result["trend"] = "SIDEWAYS"

            # Volatility
            result["volatility"] = (
                std_dev.iloc[-1] / middle_band.iloc[-1]
            ) * 100

            # Support / Resistance
            sorted_prices = sorted(df["close"].tail(100))
            result["support"] = float(np.mean(sorted_prices[:10]))
            result["resistance"] = float(np.mean(sorted_prices[-10:]))

            return result
        except Exception as e:
            self.logger.error(
                f"Error calculating indicators: {e}", exc_info=True
            )
            return {"error": str(e)}

    # =========================
    # Decisão de trading
    # =========================
    def make_trading_decision(
        self,
        indicators,
        best_timeframe=None,
        optimal_hour=None,
        ml_predictions=None,
        sentiment_score=None,
    ):
        try:
            if not indicators or "error" in indicators:
                self.logger.debug(
                    "Indicators missing or error present; skipping decision."
                )
                return None

            current_price = indicators.get("current_price", 0)
            trend = indicators.get("trend", "SIDEWAYS")
            volatility = indicators.get("volatility", 0)

            # Pesos dinamicos baseados em condições de mercado
            base_weights = {
                "sma": 1,
                "rsi": 1.5,
                "macd": 1,
                "bollinger": 1,
                "ml": 2.0,
                "sentiment": 1.0,
            }
            weights = base_weights.copy()

            # Ajustar pesos dinamicamente
            if trend == "SIDEWAYS":
                # Aumentar peso do RSI em mercados laterais
                weights["rsi"] *= 1.5
                weights["stoch"] = 1.2  # Adicionar peso ao Stochastic
                self.logger.debug(
                    "Mercado lateral detectado; aumentando peso do RSI "
                    "e Stochastic."
                )
            if volatility > 5:
                # Aumentar peso do ML em alta volatilidade
                weights["ml"] *= 1.2
                weights["ichimoku"] = 1.0  # Adicionar peso ao Ichimoku
                self.logger.debug(
                    "Alta volatilidade detectada; aumentando peso do ML "
                    "e Ichimoku."
                )
            if trend in ["BULLISH", "BEARISH"]:
                # Aumentar peso do Ichimoku em Tendencias fortes
                weights["ichimoku"] = 1.5
                self.logger.debug(
                    "Tendencia forte detectada; aumentando peso do Ichimoku."
                )

            buy_conf = sell_conf = 0

            # SMA
            if indicators["sma_20"] > indicators["sma_50"]:
                buy_conf += weights["sma"]
            else:
                sell_conf += weights["sma"]
            self.logger.debug(f"SMA signal: buy={buy_conf}, sell={sell_conf}")

            # RSI
            rsi = indicators.get("rsi", 50)
            if rsi < 30:
                buy_conf += weights["rsi"]
                self.logger.debug("RSI < 30 -> BUY")
            elif rsi > 70:
                sell_conf += weights["rsi"]
                self.logger.debug("RSI > 70 -> SELL")

            # MACD
            if indicators.get("macd", 0) > indicators.get("macd_signal", 0):
                buy_conf += weights["macd"]
                self.logger.debug("MACD bullish -> BUY")
            else:
                sell_conf += weights["macd"]
                self.logger.debug("MACD bearish -> SELL")

            # Bollinger
            if current_price < indicators.get("bb_lower", -1):
                buy_conf += weights["bollinger"]
            elif current_price > indicators.get("bb_upper", 10**10):
                sell_conf += weights["bollinger"]

            # Stochastic
            stoch_k = indicators.get("stoch_k", 50)
            stoch_d = indicators.get("stoch_d", 50)
            if stoch_k < 20 and stoch_d < 20:
                buy_conf += weights.get("stoch", 1.0)
                self.logger.debug("Stochastic oversold -> BUY")
            elif stoch_k > 80 and stoch_d > 80:
                sell_conf += weights.get("stoch", 1.0)
                self.logger.debug("Stochastic overbought -> SELL")

            # Ichimoku
            tenkan = indicators.get("ichimoku_tenkan", current_price)
            kijun = indicators.get("ichimoku_kijun", current_price)
            senkou_a = indicators.get("ichimoku_senkou_a", current_price)
            senkou_b = indicators.get("ichimoku_senkou_b", current_price)
            chikou = indicators.get("ichimoku_chikou", current_price)
            if (
                current_price > senkou_a
                and current_price > senkou_b
                and tenkan > kijun
                and chikou > current_price
            ):
                buy_conf += weights.get("ichimoku", 1.0)
                self.logger.debug("Ichimoku bullish -> BUY")
            elif (
                current_price < senkou_a
                and current_price < senkou_b
                and tenkan < kijun
                and chikou < current_price
            ):
                sell_conf += weights.get("ichimoku", 1.0)
                self.logger.debug("Ichimoku bearish -> SELL")

            # ML Predictions Integration (agora previsão Preço absoluto)
            if (
                ml_predictions
                and ml_predictions.get("best_model_prediction") is not None
            ):
                predicted_price = ml_predictions["best_model_prediction"]
                if predicted_price > current_price:
                    buy_conf += weights["ml"]
                    self.logger.debug("ML prediction bullish -> BUY")
                elif predicted_price < current_price:
                    sell_conf += weights["ml"]
                    self.logger.debug("ML prediction bearish -> SELL")
                else:
                    self.logger.debug("ML prediction neutral")
            else:
                self.logger.debug("No ML predictions available for decision")

            # Sentiment Integration
            if sentiment_score is not None:
                if sentiment_score > 0.5:
                    buy_conf += weights["sentiment"]
                    self.logger.debug(
                        f"Sentiment positivo ({sentiment_score:.2f}) -> BUY"
                    )
                elif sentiment_score < -0.5:
                    sell_conf += weights["sentiment"]
                    self.logger.debug(
                        f"Sentiment negativo ({sentiment_score:.2f}) -> SELL"
                    )
                else:
                    self.logger.debug(
                        f"Sentiment neutro ({sentiment_score:.2f})"
                    )
            else:
                self.logger.debug("No sentiment score available for decision")

            total_weight = sum(weights.values())
            confidence = (
                max(buy_conf, sell_conf) / total_weight
                if total_weight > 0
                else 0.5
            )
            if buy_conf > sell_conf:
                decision = "BUY"
            elif sell_conf > buy_conf:
                decision = "SELL"
            else:
                decision = "HOLD"
                confidence = 0.5

            # Entry / TP / SL
            entry_price = current_price
            atr = indicators.get("atr", 0)
            support = indicators.get("support", 0)
            resistance = indicators.get("resistance", 0)

            # aceitar EN/PT
            if decision in ("BUY", "COMPRAR"):
                take_profit = (
                    (resistance * 0.995)
                    if resistance > 0
                    else entry_price * 1.02
                )
                stop_loss = (
                    (support * 0.995) if support > 0 else entry_price - atr
                )
                if stop_loss >= entry_price:
                    stop_loss = entry_price * 0.98
            elif decision in ("SELL", "VENDER"):
                take_profit = (
                    (support * 1.005) if support > 0 else entry_price * 0.98
                )
                stop_loss = (
                    (resistance * 1.005)
                    if resistance > 0
                    else entry_price + atr
                )
                if stop_loss <= entry_price:
                    stop_loss = entry_price * 1.02
            else:
                take_profit = stop_loss = None

            trading_decision = TradingDecision(
                decision=decision,
                price=current_price,
                confidence=confidence,
                indicators=indicators,
                entry_price=entry_price,
                take_profit=take_profit,
                stop_loss=stop_loss,
                best_timeframe=best_timeframe,
                optimal_entry_hour=optimal_hour,
            )

            # Try salvar no DB (Não bloquear)
            try:
                self.db.insert_decision(
                    decision=trading_decision.decision,
                    price=trading_decision.price,
                    confidence=trading_decision.confidence,
                    indicators=trading_decision.indicators,
                )
            except Exception as e:
                self.logger.debug(f"Erro ao salvar Decisão no DB: {e}")

            # Atualiza cache protegido por lock
            with self._lock:
                self._cached_results.update(
                    {
                        "entry_price": entry_price,
                        "take_profit": take_profit,
                        "stop_loss": stop_loss,
                        "trading_decision": trading_decision,
                    }
                )

            return trading_decision
        except Exception as e:
            self.logger.error(
                f"Erro em make_trading_decision: {e}", exc_info=True
            )
            return None

    # =========================
    # ML Predictions revisadas
    # =========================
    def _get_ml_predictions(self, df, current_price):
        try:
            if not self.ml_trained:
                self.logger.info("Treinamento ML em background...")
                _executor.submit(lambda: self.ml_model.train(df))
                return {
                    "linear_regression_prediction": None,
                    "neural_network_prediction": None,
                    "best_model_prediction": None,
                }, []

            # faz lags em cópia pra Não modificar df original
            tmp = df[["close"]].copy()
            n_lags = 5
            for lag in range(1, n_lags + 1):
                tmp[f"lag_{lag}"] = tmp["close"].shift(lag)
            latest_data = tmp.tail(1).dropna()
            if latest_data.empty:
                return {
                    "linear_regression_prediction": None,
                    "neural_network_prediction": None,
                    "best_model_prediction": None,
                }, []

            ok, pred = run_with_timeout(
                self.ml_model.predict, latest_data.iloc[-1], timeout=5
            )
            if ok and isinstance(pred, dict):
                # Predictions are absolute prices, use directly
                lr_pred = pred.get(
                    "linear_regression_prediction", current_price
                )
                nn_pred = pred.get("neural_network_prediction", current_price)
                best_pred = pred.get("best_model_prediction", current_price)
                return {
                    "linear_regression_prediction": lr_pred,
                    "neural_network_prediction": nn_pred,
                    "best_model_prediction": best_pred,
                    "ml_score": pred.get("ml_score", 0.5),
                }, []
            else:
                return {
                    "linear_regression_prediction": None,
                    "neural_network_prediction": None,
                    "best_model_prediction": None,
                }, []
        except Exception as e:
            self.logger.error(f"Erro em ML prediction: {e}", exc_info=True)
            return {
                "linear_regression_prediction": current_price,
                "neural_network_prediction": current_price,
                "best_model_prediction": current_price,
            }, []

    # =========================
    # Analise completa
    # =========================
    def get_timeframe_signal(self, df):
        """Calcula sinal buy/sell/neutral para um dataframe baseado em
        indicadores."""
        try:
            if df is None or len(df) < 200:
                return "neutral"

            indicators = self.calculate_technical_indicators(df)
            if "error" in indicators:
                return "neutral"

            current_price = df["close"].iloc[-1]
            sma_20 = indicators.get("sma_20", current_price)
            sma_50 = indicators.get("sma_50", current_price)
            rsi = indicators.get("rsi", 50)
            macd = indicators.get("macd", 0)
            macd_signal = indicators.get("macd_signal", 0)

            # LÃƒÂ³gica de sinal simples baseada em indicadores
            buy_signals = 0
            sell_signals = 0

            # SMA crossover
            if sma_20 > sma_50:
                buy_signals += 1
            else:
                sell_signals += 1

            # RSI
            if rsi < 30:
                buy_signals += 1
            elif rsi > 70:
                sell_signals += 1

            # MACD
            if macd > macd_signal:
                buy_signals += 1
            else:
                sell_signals += 1

            # Decisão
            if buy_signals > sell_signals:
                return "comprar"
            elif sell_signals > buy_signals:
                return "vender"
            else:
                return "neutro"

        except Exception as e:
            self.logger.error(f"Erro em get_timeframe_signal: {e}")
            return "neutral"

    def analyze_timeframes(self):
        """Analisa diferentes timeframes para sinais multi-timeframe e
        Decisão consolidada."""
        try:
            self.logger.info("Analisando sinais multi-timeframe...")
            timeframes = ["1d", "4h", "1h", "15m"]
            signals = {}
            timeframe_scores = {}

            for tf in timeframes:
                try:
                    # Obter dados do timeframe
                    ok, df_or_err = run_with_timeout(
                        self.binance.get_historical_klines,
                        interval=tf,
                        limit=500,
                        timeout=10,
                    )
                    if (
                        not ok
                        or df_or_err is None
                        or isinstance(df_or_err, str)
                    ):
                        self.logger.warning(
                            f"Falha ao obter dados para timeframe {tf}"
                        )
                        signals[tf] = "neutral"
                        continue

                    df = df_or_err.copy()

                    # Calcular sinal
                    signal = self.get_timeframe_signal(df)
                    signals[tf] = signal

                    # Calcular indicadores para score (mantem compatibilidade)
                    indicators = self.calculate_technical_indicators(df)
                    if "error" not in indicators:
                        volatility = indicators.get("volatility", 0)
                        trend_strength = 0
                        sma_20 = indicators.get("sma_20", 0)
                        sma_50 = indicators.get("sma_50", 0)
                        sma_200 = indicators.get("sma_200", 0)

                        if sma_20 > sma_50 > sma_200:
                            trend_strength = 1.0
                        elif sma_20 < sma_50 < sma_200:
                            trend_strength = -1.0
                        else:
                            trend_strength = 0.0

                        rsi = indicators.get("rsi", 50)
                        momentum_score = 1 - abs(50 - rsi) / 50
                        score = (
                            (abs(trend_strength) * 0.4)
                            + (momentum_score * 0.3)
                            + ((1 - volatility / 10) * 0.3)
                        )
                        timeframe_scores[tf] = score

                    self.logger.debug(
                        f"Timeframe {tf}: signal={signal}, "
                        f"score={timeframe_scores.get(tf, 'N/A')}"
                    )

                except Exception as e:
                    self.logger.error(f"Erro ao analisar timeframe {tf}: {e}")
                    signals[tf] = "neutral"
                    continue

            # Regras de Decisão consolidada
            macro = signals.get("1d", "neutral")
            mid = signals.get("4h", "neutral")
            short = signals.get("1h", "neutral")
            micro = signals.get("15m", "neutral")

            if (
                macro == "comprar"
                and mid == "comprar"
                and short == "comprar"
                and micro == "comprar"
            ):
                final_decision = "COMPRA FORTE"
            elif (
                macro == "vender"
                and mid == "vender"
                and short == "vender"
                and micro == "vender"
            ):
                final_decision = "VENDA FORTE"
            elif (
                macro == "comprar" and mid == "comprar" and micro == "comprar"
            ):
                final_decision = "COMPRAR"
            elif macro == "vender" and mid == "vender" and micro == "vender":
                final_decision = "VENDER"
            elif macro == mid == "comprar" and short in ["comprar", "neutro"]:
                final_decision = "COMPRAR (CONFIRMAR ESPERA)"
            elif macro == mid == "vender" and short in ["vender", "neutro"]:
                final_decision = "VENDER (CONFIRMAR ESPERA)"
            else:
                final_decision = "AGUARDAR / CONFLITO"

            # Melhor timeframe (compatibilidade)
            if timeframe_scores:
                best_tf = max(timeframe_scores, key=timeframe_scores.get)
                best_score = timeframe_scores[best_tf]
                self.logger.info(
                    f"Melhor timeframe: {best_tf} (score: {best_score:.3f})"
                )
            else:
                best_tf = "1h"

            self.logger.info(f"Decisão multi-timeframe: {final_decision}")
            return best_tf, timeframe_scores, signals, final_decision

        except Exception as e:
            self.logger.error(f"Erro em analyze_timeframes: {e}")
            return (
                "1h",
                {},
                {
                    "1d": "neutral",
                    "4h": "neutral",
                    "1h": "neutral",
                    "15m": "neutral",
                },
                "WAIT / CONFLICT",
            )

    def analyze_optimal_entry_time(self, best_timeframe):
        """Analisa o melhor historico de entrada baseado em dados
        historicos."""
        try:
            self.logger.info(
                f"Analisando historico ótimo de entrada para timeframe "
                f"{best_timeframe}..."
            )

            # Obter dados historicos mais longos
            ok, df_or_err = run_with_timeout(
                self.binance.get_historical_klines,
                interval=best_timeframe,
                limit=1000,
                timeout=15,
            )
            if not ok or df_or_err is None or isinstance(df_or_err, str):
                self.logger.warning(
                    "Falha ao obter dados para Analise de historico"
                )
                return None, {}

            df = df_or_err.copy()

            # Extrair hora do dia
            df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

            # Calcular retornos por hora
            df["returns"] = df["close"].pct_change()

            # Agrupar por hora e calcular metricas
            hourly_stats = (
                df.groupby("hour")
                .agg(
                    {
                        "returns": ["mean", "std", "count"],
                        "close": ["mean", "std"],
                    }
                )
                .round(6)
            )

            # Calcular score para cada hora
            hourly_scores = {}
            for hour in range(24):
                if hour in hourly_stats.index:
                    mean_return = hourly_stats.loc[hour, ("returns", "mean")]
                    std_return = hourly_stats.loc[hour, ("returns", "std")]
                    count = hourly_stats.loc[hour, ("returns", "count")]

                    if count > 10:
                        # Score baseado em retorno médio ajustado pelo risco
                        if std_return > 0:
                            sharpe_ratio = mean_return / std_return
                            # Penalizar volatilidade muito alta
                            risk_adjusted_score = sharpe_ratio * (
                                1 - min(std_return * 10, 0.5)
                            )
                            hourly_scores[hour] = risk_adjusted_score
                        else:
                            hourly_scores[hour] = mean_return

            # Encontrar melhor historico
                    if hourly_scores:
                        best_hour = max(hourly_scores, key=hourly_scores.get)
                        best_score = hourly_scores[best_hour]
                        self.logger.info(
                            f"Melhor historico de entrada: {best_hour:02d}:00 "
                            f"(score: {best_score:.4f})"
                        )
                        return best_hour, hourly_scores
            else:
                self.logger.warning("Nenhum historico pôde ser analisado")
                return None, {}

        except Exception as e:
            self.logger.error(f"Erro em analyze_optimal_entry_time: {e}")
            return None, {}

    def analyze_market(self):
        """Realiza Analise completa com ML, indicadores, Analise de
        timeframe e historico ótimo."""
        try:
            start_time = time.time()
            self.logger.info("===== HISTORICO DA ANALISE DO MERCADO =====")
            # Sincroniza Spot (Não bloqueante)
            try:
                if hasattr(self, "risk_manager"):
                    self.risk_manager.sync_spot_account(self.binance)
            except Exception:
                pass
            # Controle de cache
            current_time = time.time()
            if (
                current_time - self._last_api_call_time
                < self._api_call_interval
            ):
                self.logger.debug("Usando cache de resultados recentes.")
                with self._lock:
                    if self._cached_results:
                        self.logger.info("Retornando dados do cache.")
                        return dict(self._cached_results)

            #  Analise de timeframes
            best_timeframe, timeframe_scores, signals, final_decision = (
                self.analyze_timeframes()
            )

            #  Analise de historico ótimo
            optimal_hour, hourly_scores = self.analyze_optimal_entry_time(
                best_timeframe
            )

            #  Obter Preço atual (com timeout e fallback)
            ok, res = run_with_timeout(
                self.binance.get_current_price, timeout=8
            )
            current_price = res if ok else self.get_current_price()
            if current_price is None:
                self.logger.error("Falha ao obter Preço atual da Binance.")
                backend_logger.error(
                    "Failed to get current price from Binance"
                )
                return {"error": "Failed to get current price"}
            self.logger.info(f"Preço atual obtido: {current_price:.2f} USDT")
            backend_logger.info(
                f"Current price retrieved: {current_price:.2f} USDT"
            )

            # 3. Obter Estatisticas 24h
            ok_stats, stats_24h = run_with_timeout(
                self.binance.get_24h_stats, timeout=8
            )
            if not ok_stats or stats_24h is None:
                self.logger.warning(
                    "Falha ao obter Estatisticas 24h da Binance"
                )
                stats_24h = {}
            else:
                self.logger.info("Estatisticas 24h obtidas da Binance")

            # 4Ã¯Â¸ÂÃ¢Æ’Â£ Dados historicos usando melhor timeframe para
            # indicadores
            ok, df_or_err = run_with_timeout(
                self.binance.get_historical_klines,
                interval=best_timeframe,
                limit=1000,
                timeout=12,
            )
            if ok and df_or_err is not None and not isinstance(df_or_err, str):
                df = df_or_err
                self.logger.info(
                    f"Dados historicos obtidos da Binance ({best_timeframe})."
                )
            else:
                self.logger.warning(
                    "Falha na Binance, usando historico local ou simulado."
                )
                df = self.get_historical_prices()

            if df is None or df.empty:
                self.logger.error("Nenhum dado historico disoinivel.")
                return {"error": "Failed to get historical data"}

            df = df.copy()

            # Dados historicos para ML sempre em 1h para previsão de 1H para
            # frente
            ok_ml, df_ml_or_err = run_with_timeout(
                self.binance.get_historical_klines,
                interval="1h",
                limit=1000,
                timeout=12,
            )
            if (
                ok_ml
                and df_ml_or_err is not None
                and not isinstance(df_ml_or_err, str)
            ):
                df_ml = df_ml_or_err
                self.logger.info(
                    "Dados historicos para ML obtidos da Binance (1h)."
                )
            else:
                self.logger.warning(
                    "Falha na Binance para ML, usando historico local "
                    "ou simulado."
                )
                df_ml = self.get_historical_prices(days=30, interval="1h")

            if df_ml is None or df_ml.empty:
                self.logger.warning(
                    "Nenhum dado historico para ML, usando df principal."
                )
                df_ml = df

            #  Calcular indicadores tecnicos
            indicators = self.calculate_technical_indicators(df)
            if "error" in indicators:
                self.logger.error(
                    f"Erro ao calcular indicadores: {indicators.get('error')}"
                )
                return indicators
            indicators["current_price"] = current_price
            self.logger.info("Indicadores graficos calculados com sucesso.")

            # Treinamento / predição ML
            ml_predictions = {"best_model_prediction": None}
            ml_multi_step_predictions = []

            if not self.ml_trained:
                self.logger.info("Treinando modelo ML em background...")

                def _train_and_flag():
                    try:
                        # Priorizar dados reais da Binance para treinamento
                        if (
                            df_ml is not None
                            and not df_ml.empty
                            and "close" in df_ml.columns
                        ):
                            # Verificar se os dados serão reais (não simulados)
                            if (
                                df_ml["close"].max() > 50000
                            ):  # Threshold para dados reais
                                self.ml_model.train(df_ml)
                                self.ml_model.save_model()
                                self.ml_trained = True
                                self.logger.info(
                                    "Treinamento ML concluido com sucesso "
                                    "e modelo salvo."
                                )
                            else:
                                self.logger.warning(
                                    "Dados simulados detectados; pulando "
                                    "treinamento."
                                )
                        else:
                            self.logger.warning(
                                "Dados insuficientes para treinamento ML."
                            )
                    except Exception as e:
                        self.logger.exception(
                            "Erro durante treinamento ML: %s", e
                        )

                _executor.submit(_train_and_flag)
                # Fallback imediato: previsão por regressão linear de
                # tendencias recente
                try:
                    closes = (
                        df_ml["close"].dropna().values
                        if "close" in df_ml.columns
                        else None
                    )
                    if closes is not None and len(closes) >= 20:
                        import numpy as np

                        n = 60 if len(closes) >= 60 else len(closes)
                        y = closes[-n:]
                        x = np.arange(n)
                        a, b = np.polyfit(x, y, 1)  # y = a*x + b
                        pred_baseline = float(a * (n) + b)
                        ml_predictions = {
                            "linear_regression_prediction": pred_baseline,
                            "neural_network_prediction": pred_baseline,
                            "best_model_prediction": pred_baseline,
                            "ml_score": 0.5,
                        }
                    else:
                        ml_predictions = {
                            "linear_regression_prediction": None,
                            "neural_network_prediction": None,
                            "best_model_prediction": None,
                        }
                except Exception:
                    ml_predictions = {
                        "linear_regression_prediction": None,
                        "neural_network_prediction": None,
                        "best_model_prediction": None,
                    }
                ml_multi_step_predictions = []
            else:
                # Verificar se precisa retreinar baseado no tempo ou mudança de
                # regime
                if self.ml_model.should_retrain(df_ml):
                    self.logger.info(
                        "Retreinamento ML necessario, "
                        "treinando em background..."
                    )

                    def _retrain_and_flag():
                        try:
                            if (
                                df_ml is not None
                                and not df_ml.empty
                                and "close" in df_ml.columns
                            ):
                                if df_ml["close"].max() > 50000:
                                    self.ml_model.train(df_ml)
                                    self.ml_model.save_model()
                                    self.logger.info(
                                        "Retreinamento ML concluido "
                                        "com sucesso."
                                    )
                                else:
                                    self.logger.warning(
                                        "Dados simulados detectados; pulando "
                                        "retreinamento."
                                    )
                        except Exception as e:
                            self.logger.exception(
                                "Erro durante retreinamento ML: %s", e
                            )

                    _executor.submit(_retrain_and_flag)
                    ml_predictions = {
                        "linear_regression_prediction": None,
                        "neural_network_prediction": None,
                        "best_model_prediction": None,
                    }
                    ml_multi_step_predictions = []
                else:
                    ml_predictions, ml_multi_step_predictions = (
                        self._get_ml_predictions(df_ml, current_price)
                    )
                    # Validar e persistir METRICAS para múltiplos timeframes
                    try:
                        from database import DatabaseManager

                        for tf in ["15m", "1h", "4h", "1d"]:
                            ok_tf, df_tf = run_with_timeout(
                                self.binance.get_historical_klines,
                                interval=tf,
                                limit=1000,
                                timeout=10,
                            )
                            if (
                                ok_tf
                                and df_tf is not None
                                and not isinstance(df_tf, str)
                            ):
                                met = self.ml_model.walk_forward_validate(
                                    df_tf
                                )
                                DatabaseManager.save_ml_metrics(tf, met)
                    except Exception:
                        pass
                    self.logger.info(
                        f"Predição ML concluída: {ml_predictions}"
                    )
                    try:
                        metrics = self.ml_model.walk_forward_validate(df_ml)
                    except Exception:
                        metrics = {}
                    if isinstance(ml_predictions, dict):
                        ml_predictions["metrics"] = metrics
                    # Persistir metricas por timeframe para ajuste de
                    # thresholds
                    try:
                        from database import DatabaseManager

                        DatabaseManager.save_ml_metrics("1h", metrics)
                    except Exception:
                        pass

            # 7Ã¯Â¸ÂÃ¢Æ’Â£ Calcular sentimento do mercado
            from News_Worker import NewsWorker

            try:
                sentiment_score = NewsWorker.get_average_sentiment_score()
                self.logger.info(
                    f"Market sentiment score calculated: {sentiment_score}"
                )
            except Exception as e:
                self.logger.error(
                    f"Error calculating sentiment score: {e}, "
                    "using 0.0 as fallback"
                )
                sentiment_score = 0.0

            # 8Ã¯Â¸ÂÃ¢Æ’Â£ Decisão de trading
            trading_decision = self.make_trading_decision(
                indicators,
                best_timeframe,
                optimal_hour,
                ml_predictions,
                sentiment_score,
            )
            decision = (
                trading_decision.decision if trading_decision else "HOLD"
            )
            confidence = getattr(trading_decision, "confidence", 0.5)
            self.logger.info(
                f"Decisão: {decision} | Confiança: {confidence:.3f} | "
                f"Tendencia: {indicators.get('trend')} | "
                f"RSI: {indicators.get('rsi'):.2f}"
            )
            backend_logger.info(
                f"Trading decision made: {decision} "
                f"with confidence {confidence:.3f}"
            )

            #  Resultado consolidado com informaçoes de timeframe e historico
            result = {
                "price": float(current_price),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "decision": decision,
                "confidence": float(confidence),
                "best_timeframe": best_timeframe,
                "optimal_entry_hour": optimal_hour,
                "timeframe_scores": timeframe_scores,
                "signals": signals,
                "final_decision": final_decision,
                "hourly_scores": hourly_scores,
                "support": indicators.get("support"),
                "resistance": indicators.get("resistance"),
                "trend": indicators.get("trend"),
                "volatility": indicators.get("volatility"),
                "rsi": indicators.get("rsi"),
                "macd": indicators.get("macd"),
                "sma_20": indicators.get("sma_20"),
                "sma_50": indicators.get("sma_50"),
                "sma_200": indicators.get("sma_200"),
                "bb_upper": indicators.get("bb_upper"),
                "bb_middle": indicators.get("bb_middle"),
                "bb_lower": indicators.get("bb_lower"),
                "stats_24h": stats_24h,
                "sentiment_score": sentiment_score,
                "ml_predictions": ml_predictions,
                "ml_multi_step_predictions": ml_multi_step_predictions,
                "df_tail": df[["timestamp", "open", "high", "low", "close"]]
                .tail(5)
                .to_dict(orient="records"),
            }

            # Atualiza cache
            with self._lock:
                self._cached_results.update(result)

            self._last_api_call_time = time.time()

            #  Log final com informaçoes de timeframe e historico
            total_time = time.time() - start_time
            self.logger.info(f"Tempo total de Analise: {total_time:.2f}s")
            self.logger.info(f"Melhor timeframe: {best_timeframe}")
            if optimal_hour is not None:
                self.logger.info(
                    f"historico ótimo de entrada: {optimal_hour:02d}:00"
                )
                # Snapshot de conta após Analise
            try:
                from database import DatabaseManager

                rs = (
                    getattr(self, "risk_manager", None).summary()
                    if hasattr(self, "risk_manager")
                    else {}
                )
                exposure = rs.get("portfolio_exposure_notional", 0.0)
                open_count = rs.get("open_positions_count", 0)
                pnl_unreal = 0.0
                for pos in rs.get("open_positions", []):
                    entry = pos.get("entry_price", None)
                    qty = float(pos.get("position_units", 0) or 0)
                    sym = pos.get("symbol", "BTCUSDT")
                    if entry is None or qty <= 0:
                        continue
                    if sym == "BTCUSDT":
                        px = float(result.get("price", entry))
                    else:
                        try:
                            t = self.binance.get_current_price(sym)
                            px = float(t) if t else entry
                        except Exception:
                            px = entry
                    pnl_unreal += (px - entry) * qty
                DatabaseManager.save_account_snapshot(
                    exposure,
                    open_count,
                    pnl_unreal,
                    json.dumps(rs.get("open_positions", [])),
                )
            except Exception:
                pass
            self.logger.info("===== FIM DA ANALISE =====`n")
            return dict(self._cached_results)

        except Exception as e:
            self.logger.exception("Erro inesperado em analyze_market: %s", e)
            return {"error": str(e)}

    def shutdown(self, wait=True):
        """Chama ao fechar a aplicação para encerrar executors e liberar
        recursos."""
        try:
            self.logger.info("Shutting down MarketAnalyzer...")
            # fecha executor global se desejar (atenção: pode afetar outras
            # partes que usam _executor)
            try:
                _executor.shutdown(wait=wait)
                self.logger.info("Global executor shutdown completed.")
            except Exception as e:
                self.logger.debug(f"Error shutting down executor: {e}")
        except Exception:
            pass
