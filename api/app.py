# -*- coding: utf-8 -*-
import os
import threading
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from Market_Analyzer import MarketAnalyzer
from RiskManager import RiskManager
from database import DatabaseManager


class TradingService:
    def __init__(self, db_path: str = "btc_trader.db"):
        self.analyzer = MarketAnalyzer(db_path)
        self.risk = RiskManager()
        self.last_results: Dict[str, Any] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self, interval_sec: int = 30):
        if self._thread and self._thread.is_alive():
            return

        def _loop():
            # Loop com espera baseada no Event para shutdown rápido
            while not self._stop.is_set():
                try:
                    try:
                        self.risk.sync_spot_account(self.analyzer.binance)
                    except Exception:
                        pass
                    res = self.analyzer.analyze_market()
                    if isinstance(res, dict):
                        self.last_results = res
                except Exception:
                    pass
                # Sai cedo se _stop for setado durante a espera
                if self._stop.wait(interval_sec):
                    break

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self):
        # Sinaliza parada e aguarda a thread encerrar sem travar shutdown
        self._stop.set()
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=5)
            except Exception:
                pass

    def summary(self) -> Dict[str, Any]:
        rs = self.risk.summary() or {}
        res = dict(self.last_results) if isinstance(
            self.last_results, dict) else {}
        res["risk_summary"] = rs
        return res

    def analysis_for_timeframe(self, tf: str) -> Dict[str, Any]:
        df = self.analyzer.binance.get_historical_klines(
            symbol='BTCUSDT', interval=tf, limit=1000)
        if df is None or getattr(df, 'empty', True):
            raise HTTPException(
                status_code=400, detail=f"Sem dados para timeframe {tf}")
        indicators = self.analyzer.calculate_technical_indicators(df)
        if 'error' in indicators:
            raise HTTPException(status_code=400, detail=indicators['error'])
        current_price = float(df['close'].iloc[-1])
        indicators['current_price'] = current_price
        preds, _ = self.analyzer._get_ml_predictions(df, current_price)
        # Melhor horário de entrada para o timeframe
        try:
            entry_hour, _scores = self.analyzer.analyze_optimal_entry_time(tf)
        except Exception:
            entry_hour = None
        td = self.analyzer.make_trading_decision(
            indicators,
            best_timeframe=tf,
            optimal_hour=entry_hour,
            ml_predictions=preds,
            sentiment_score=None)
        # séries para gráfico (MAs e BB)
        try:
            df2 = df.copy()
            df2['sma20'] = df2['close'].rolling(20).mean()
            df2['sma50'] = df2['close'].rolling(50).mean()
            df2['sma200'] = df2['close'].rolling(200).mean()
            mid = df2['close'].rolling(20).mean()
            std = df2['close'].rolling(20).std()
            df2['bb_up'] = mid + 2*std
            df2['bb_mid'] = mid
            df2['bb_low'] = mid - 2*std
            times = [str(t) for t in df2['timestamp'].astype(str).tolist()]
            series = {
                'time': times,
                'close': df2['close'].round(2).tolist(),
                'sma20': df2['sma20'].round(2).bfill().tolist(),
                'sma50': df2['sma50'].round(2).bfill().tolist(),
                'sma200': df2['sma200'].round(2).bfill().tolist(),
                'bb_up': df2['bb_up'].round(2).bfill().tolist(),
                'bb_mid': df2['bb_mid'].round(2).bfill().tolist(),
                'bb_low': df2['bb_low'].round(2).bfill().tolist(),
            }
        except Exception:
            series = {}
        # sentiment
        try:
            from News_Worker import NewsWorker
            sentiment_score = NewsWorker.get_average_sentiment_score()
        except Exception:
            sentiment_score = None

        return {
            'timeframe': tf,
            'price': current_price,
            'indicators': indicators,
            'ml_predictions': preds,
            'decision': getattr(td, 'decision', 'HOLD') if td else 'HOLD',
            'confidence': getattr(td, 'confidence', 0.5) if td else 0.5,
            'take_profit': getattr(td, 'take_profit', None) if td else None,
            'stop_loss': getattr(td, 'stop_loss', None) if td else None,
            'entry_hour': entry_hour,
            'volatility_pct': (
                float(indicators.get('atr', 0)) / current_price * 100.0
            ) if current_price else None,
            'sentiment_score': sentiment_score,
            'series': series,
            'df_tail': df[['timestamp', 'open', 'high', 'low', 'close']]
            .tail(5).to_dict(orient='records'),
        }


service = TradingService()

app = FastAPI(title="CryptoVision API", version="1.0.0")
allow_origins_env = os.getenv("ALLOW_ORIGINS", "*")
allow_origins = [o.strip() for o in allow_origins_env.split(",")
                 ] if allow_origins_env != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    DatabaseManager.initialize_database()
    # Em produção, é possível desabilitar o loop e rodar um worker separado
    enable_bg = os.getenv("ENABLE_BACKGROUND_SERVICE", "0").strip() in {
        "1", "true", "True"}
    if enable_bg:
        service.start(interval_sec=int(os.getenv("BACKGROUND_INTERVAL", "30")))


@app.on_event("shutdown")
def _shutdown():
    service.stop()


@app.get("/status")
def status():
    try:
        k, s = DatabaseManager.get_api_keys()
        has_keys = bool(k and s)
    except Exception:
        has_keys = False
    return {
        "ok": True,
        "has_results": isinstance(service.last_results, dict)
        and bool(service.last_results),
        "has_keys": has_keys,
    }


@app.get("/summary")
def get_summary():
    if not isinstance(service.last_results, dict) or not service.last_results:
        try:
            res = service.analyzer.analyze_market()
            if isinstance(res, dict):
                service.last_results = res
        except Exception:
            pass
    return service.summary()


@app.get("/analysis")
def get_analysis(tf: str = "1h"):
    return service.analysis_for_timeframe(tf)


@app.get("/orders/open")
def get_open_orders(symbol: str = "BTCUSDT"):
    try:
        orders = service.analyzer.binance.get_open_orders(symbol)
        return {"symbol": symbol, "orders": orders or []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class Keys(BaseModel):
    api_key: str
    api_secret: str


@app.post("/keys")
def post_keys(keys: Keys):
    try:
        ok = DatabaseManager.save_api_keys(keys.api_key, keys.api_secret)
        if not ok:
            raise HTTPException(
                status_code=400, detail="Falha ao salvar chaves")
        try:
            from BinanceConnector import BinanceConnector
            service.analyzer.binance = BinanceConnector(
                keys.api_key, keys.api_secret)
        except Exception:
            pass
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
def post_analyze():
    try:
        res = service.analyzer.analyze_market()
        if isinstance(res, dict):
            service.last_results = res
        return {"ok": True, "result": service.last_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sync")
def post_sync():
    try:
        service.risk.sync_spot_account(service.analyzer.binance)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
def get_config():
    return {
        "auto_trade_min_confidence": DatabaseManager.get_config_float(
            'auto_trade_min_confidence', 0.70),
        "auto_trade_min_ml_score": DatabaseManager.get_config_float(
            'auto_trade_min_ml_score', 0.60),
        "min_notional": DatabaseManager.get_config_float('min_notional', 10.0),
    }


@app.post("/config")
def post_config(payload: Dict[str, Any]):
    try:
        for k in ['auto_trade_min_confidence', 'auto_trade_min_ml_score',
                  'min_notional']:
            if k in payload:
                DatabaseManager.set_config(k, str(payload[k]))
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/decisions")
def get_decisions(limit: int = 10):
    try:
        rows = DatabaseManager.get_recent_decisions(limit) or []
        return {"items": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Simple direction endpoint
# -----------------------------
def _format_details(ind: dict, preds: dict, td) -> str:
    try:
        parts = []
        parts.append("Resumo da análise:")
        if ind:
            rsi = ind.get('rsi')
            macd = ind.get('macd')
            macd_sig = ind.get('macd_signal')
            trend = 'Alta' if (
                ind.get('sma_20', 0) > ind.get('sma_50', 0)
                > ind.get('sma_200', 0)
            ) else (
                'Baixa' if (
                    ind.get('sma_20', 0) < ind.get('sma_50', 0)
                    < ind.get('sma_200', 0)
                ) else 'Lateral'
            )
            parts.append(f"- Tendência: {trend}")
            if rsi is not None:
                parts.append(
                    f"- RSI(14): {rsi:.2f} ("
                    f"{'sobrecomprado' if rsi > 70 else 'sobrevendido'}"
                    f"if rsi < 30 else 'neutro')"
                )
            if macd is not None and macd_sig is not None:
                parts.append(f"- MACD: {macd:.4f} vs sinal {macd_sig:.4f}")
            bb_up, bb_mid, bb_low = ind.get('bb_upper'), ind.get(
                'bb_middle'), ind.get('bb_lower')
            if bb_up and bb_mid and bb_low:
                parts.append(
                    f"- Bandas de Bollinger: topo {bb_up:.2f} | "
                    f"meio {bb_mid:.2f} | fundo {bb_low:.2f}")
        if preds:
            best = preds.get('best_model_prediction')
            lr = preds.get('linear_regression_prediction')
            nn = preds.get('neural_network_prediction')
            ml_score = preds.get('ml_score')
            if best:
                parts.append(f"- Previsão (melhor modelo): {best:.2f}")
            if lr is not None and nn is not None:
                parts.append(f"- Modelos: LR {lr:.2f} | NN {nn:.2f}")
            if ml_score is not None:
                parts.append(f"- Confiança ML: {ml_score:.2f}")
        if td:
            parts.append(
                f"- Decisão: {getattr(td,'decision','HOLD')} | "
                f"Confiança: {getattr(td,'confidence',0.5):.2f}"
            )
            if getattr(td, 'take_profit', None) or getattr(
                    td, 'stop_loss', None):
                parts.append(
                    f"- TP: {getattr(td,'take_profit',None)} | "
                    f"SL: {getattr(td,'stop_loss',None)}")
        return "\n".join(parts)
    except Exception:
        return "Análise detalhada não disponível no momento."


@app.get("/direction")
def get_direction(tf: str = "1h"):
    try:
        # dados
        df = service.analyzer.binance.get_historical_klines(
            symbol='BTCUSDT', interval=tf, limit=1000)
        if df is None or getattr(df, 'empty', True):
            raise HTTPException(
                status_code=400, detail=f"Sem dados para timeframe {tf}")
        indicators = service.analyzer.calculate_technical_indicators(df)
        if 'error' in indicators:
            raise HTTPException(status_code=400, detail=indicators['error'])
        price = float(df['close'].iloc[-1])
        preds, ml_score = service.analyzer._get_ml_predictions(df, price)
        # Melhor hora de entrada para o timeframe
        try:
            entry_hour, _scores = service.analyzer.analyze_optimal_entry_time(
                tf)
        except Exception:
            entry_hour = None
        td = service.analyzer.make_trading_decision(
            indicators,
            best_timeframe=tf,
            optimal_hour=entry_hour,
            ml_predictions=preds,
            sentiment_score=None)

        # direction simplificada
        decision = getattr(td, 'decision', 'HOLD') if td else 'HOLD'
        d_up = {
            'BUY',
            'COMPRAR',
        }
        d_down = {
            'SELL',
            'VENDER',
        }
        direction = 'UP' if str(decision).upper() in d_up else (
            'DOWN' if str(decision).upper() in d_down else 'NEUTRAL'
        )
        confidence = float(getattr(td, 'confidence', 0.5) or 0.5)

        # séries para gráfico
        df2 = df.copy()
        df2['sma20'] = df2['close'].rolling(20).mean()
        df2['sma50'] = df2['close'].rolling(50).mean()
        df2['sma200'] = df2['close'].rolling(200).mean()
        mid = df2['close'].rolling(20).mean()
        std = df2['close'].rolling(20).std()
        df2['bb_up'] = mid + 2*std
        df2['bb_mid'] = mid
        df2['bb_low'] = mid - 2*std
        times = [str(t) for t in df2['timestamp'].astype(str).tolist()]
        series = {
            'time': times,
            'close': df2['close'].round(2).tolist(),
            'sma20': df2['sma20'].round(2).bfill().tolist(),
            'sma50': df2['sma50'].round(2).bfill().tolist(),
            'sma200': df2['sma200'].round(2).bfill().tolist(),
            'bb_up': df2['bb_up'].round(2).bfill().tolist(),
            'bb_mid': df2['bb_mid'].round(2).bfill().tolist(),
            'bb_low': df2['bb_low'].round(2).bfill().tolist(),
        }

        # 24h stats
        try:
            stats24 = service.analyzer.binance.get_24h_stats(
                symbol='BTCUSDT'
            ) or {}
        except Exception:
            stats24 = {}
        # risk summary
        try:
            risk_sum = service.risk.summary() or {}
        except Exception:
            risk_sum = {}

        # sentiment score (News)
        try:
            from News_Worker import NewsWorker
            sentiment_score = NewsWorker.get_average_sentiment_score()
        except Exception:
            sentiment_score = None

        details = _format_details(indicators, preds or {}, td)
        return {
            'timeframe': tf,
            'price': price,
            'decision': decision,
            'direction': direction,
            'confidence': confidence,
            'ml_predictions': preds,
            'indicators': indicators,
            'entry_hour': entry_hour,
            'volatility_pct': (float(indicators.get('atr', 0)) / price * 100.0)
            if price else None,
            'stats_24h': stats24,
            'risk_summary': risk_sum,
            'sentiment_score': sentiment_score,
            'details': details,
            'series': series,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
