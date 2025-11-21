# risk_manager.py
import os
import json
from datetime import datetime
from collections import defaultdict, deque
from typing import List, Dict, Optional, Any

import pandas as pd

from Logger import iniciar_logger

# ------------------------------------------------------------
# RiskManager: versão avançada (Prop Firm style)
# ------------------------------------------------------------


class RiskManager:
    """
    RiskManager avançado:
    - Gestão global de posições
    - Position sizing dinâmico
    - Stop / TP / trailing stop
    - Combinação multi-timeframe
    - Controle de exposição e correlação
    - Histórico de trades e métricas (daily/monthly)
    - Backtesting simples
    - Persistência simples (JSON)
    """

    def __init__(
        self,
        account_balance: float = 10000.0,
        max_risk_per_trade: float = 0.02,
        max_total_risk: float = 0.10,
        max_exposure_pct: float = 0.25,
        max_correlation_threshold: float = 0.75,
        history_path: str = None
    ):
        self.logger = iniciar_logger("RiskManager")
        self.account_balance = float(account_balance)
        self.max_risk_per_trade = float(max_risk_per_trade)
        self.max_total_risk = float(max_total_risk)
        self.max_exposure_pct = float(max_exposure_pct)
        self.max_correlation_threshold = float(max_correlation_threshold)

        # Posições abertas: lista de dicts
        self.open_positions: List[Dict[str, Any]] = []
        # Histórico de trades realizados
        self.trade_history: List[Dict[str, Any]] = []
        # Performance tracking por setup
        self.setup_performance = defaultdict(
            lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0.0, 'count': 0})
        self.daily_pnl = defaultdict(float)
        self.monthly_pnl = defaultdict(float)

        # Cálculos de correlação (cache)
        self.correlation_matrix = {}  # symbol -> {other: corr}
        # Pequeno cache para exposures
        self._exposure_cache = None
        self._history_path = history_path or os.path.join(
            os.path.dirname(__file__), "risk_history.json")
        self._max_history_len = 10000

        # Backtest results and optimal params
        self.backtest_results = {}
        self.optimal_params = {}

        # Keep last N closed trades in memory for quick stats
        self._recent_closed = deque(maxlen=500)

        # Ensure history file exists
        if not os.path.exists(self._history_path):
            try:
                with open(self._history_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "trade_history": [],
                            "setup_performance": {},
                            "meta": {}
                        },
                        f,
                    )
            except Exception:
                pass

        self.logger.info(
            "RiskManager initialized (account_balance=%.2f)",
            self.account_balance)

    # -------------------------
    # Persistence
    # -------------------------
    def save_state(self):
        try:
            data = {
                "account_balance": self.account_balance,
                "max_risk_per_trade": self.max_risk_per_trade,
                "max_total_risk": self.max_total_risk,
                "open_positions": self.open_positions,
                "trade_history": self.trade_history[-self._max_history_len:],
                "setup_performance": dict(self.setup_performance),
                "daily_pnl": dict(self.daily_pnl),
                "monthly_pnl": dict(self.monthly_pnl),
                "optimal_params": self.optimal_params
            }
            with open(self._history_path, "w", encoding="utf-8") as f:
                json.dump(data, f, default=str, ensure_ascii=False, indent=2)
            self.logger.info("RiskManager state saved to %s",
                             self._history_path)
            return True
        except Exception as e:
            self.logger.error("Error saving RiskManager state: %s", e)
            return False

    def load_state(self):
        try:
            if not os.path.exists(self._history_path):
                return False
            with open(self._history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.account_balance = float(
                data.get("account_balance", self.account_balance))
            self.max_risk_per_trade = float(
                data.get("max_risk_per_trade", self.max_risk_per_trade))
            self.max_total_risk = float(
                data.get("max_total_risk", self.max_total_risk))
            self.open_positions = data.get("open_positions", [])
            self.trade_history = data.get("trade_history", [])
            self.setup_performance.update(data.get("setup_performance", {}))
            self.daily_pnl.update(data.get("daily_pnl", {}))
            self.monthly_pnl.update(data.get("monthly_pnl", {}))
            self.optimal_params.update(data.get("optimal_params", {}))
            self.logger.info(
                "RiskManager state loaded from %s", self._history_path)
            return True
        except Exception as e:
            self.logger.error("Error loading RiskManager state: %s", e)
            return False

    # -------------------------
    # Helper utils
    # -------------------------
    @staticmethod
    def _safe_float(x, default=0.0):
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    # -------------------------
    # Position Sizing & Risk
    # -------------------------
    def calculate_position_size(
            self,
            entry_price: float,
            stop_loss: float,
            volatility: float,
            trend_strength: float = 1.0,
            use_account_balance: Optional[float] = None) -> float:
        """
        Calculate position size in asset units.
        - Uses account_balance * max_risk_per_trade as risk amount
          (in quote currency).
        - Uses stop distance (absolute price) to compute units.
        - Adjusts by volatility and trend_strength.
        """
        try:
            account = float(use_account_balance or self.account_balance)
            risk_amount = account * self.max_risk_per_trade

            entry = float(entry_price)
            stop = float(stop_loss)
            if stop == entry:
                self.logger.warning("Stop equals entry; returning 0 size.")
                return 0.0

            stop_distance = abs(entry - stop)
            if stop_distance <= 0:
                return 0.0

            # Basic position size in quote currency -> units = risk_amount /
            # stop_distance
            units = (risk_amount / stop_distance)

            # Adjustments
            # more conservative in high vol
            volatility_adj = max(0.4, 1 - (volatility / 10.0))
            # increases with trend strength
            trend_adj = min(1.6, 1 + (trend_strength * 0.25))

            size = units * volatility_adj * trend_adj

            # Cap: don't open position > max_exposure_pct of account
            max_position_value = account * self.max_exposure_pct
            max_units = max_position_value / entry if entry > 0 else size
            size = min(size, max_units)

            self.logger.debug(
                "Position size calc: entry=%.2f stop=%.2f stop_dist=%.2f "
                "units=%.6f adj(vol)=%.2f adj(trend)=%.2f final=%.6f",
                entry,
                stop,
                stop_distance,
                units,
                volatility_adj,
                trend_adj,
                size)
            return float(size)
        except Exception as e:
            self.logger.error("Error in calculate_position_size: %s", e)
            return 0.0

    def calculate_dynamic_stop_loss(
            self,
            entry_price: float,
            atr: float,
            support: float,
            resistance: float,
            volatility: float,
            trend: str = "NEUTRAL") -> float:
        """
        Calculate dynamic stop loss price based on ATR, S/R and volatility.
        Returns stop price (for long: below entry; for short: above entry).
        """
        try:
            entry = float(entry_price)
            base = float(atr) * \
                1.5 if atr and atr > 0 else max(entry * 0.01, 1.0)

            if volatility > 6:
                base *= 1.25
            elif volatility < 2:
                base *= 0.9

            if trend in ["BULLISH", "BEARISH"]:
                distance = base
            else:
                if support and resistance and resistance > support:
                    distance = (resistance - support) * 0.25
                else:
                    distance = base

            if trend == "BULLISH":
                stop = entry - distance
            elif trend == "BEARISH":
                stop = entry + distance
            else:
                stop = entry - distance  # default long-oriented

            min_dist = entry * 0.005
            if abs(entry - stop) < min_dist:
                stop = (
                    entry - min_dist
                    if trend != "BEARISH"
                    else entry + min_dist
                )

            return float(stop)
        except Exception as e:
            self.logger.error("Error calculate_dynamic_stop_loss: %s", e)
            return float(entry_price * 0.98)

    def calculate_take_profit(
            self,
            entry_price: float,
            stop_loss: float,
            risk_reward: float = 2.0,
            resistance: Optional[float] = None,
            trend: Optional[str] = None) -> float:
        try:
            entry = float(entry_price)
            stop = float(stop_loss)
            risk = abs(entry - stop)
            reward = risk * float(risk_reward)
            if trend == "BULLISH":
                tp = entry + reward
                if resistance and resistance > entry and (
                        resistance - entry) < (reward * 1.5):
                    tp = resistance * 0.995
            elif trend == "BEARISH":
                tp = entry - reward
                if resistance and resistance < entry and (
                        entry - resistance) < (reward * 1.5):
                    tp = resistance * 1.005
            else:
                tp = entry + reward
            return float(tp)
        except Exception as e:
            self.logger.error("Error calculate_take_profit: %s", e)
            return float(entry_price * 1.02)

    def calculate_trailing_stop(
            self,
            current_price: float,
            entry_price: float,
            volatility: float,
            profit_pct: float) -> Optional[float]:
        try:
            if profit_pct < 0.01:
                return None
            base_trail = profit_pct * 0.5
            if volatility > 6:
                base_trail *= 1.2
            elif volatility < 2:
                base_trail *= 0.85
            trailing_pct = max(0.005, min(base_trail, 0.04))
            if current_price > entry_price:
                return float(current_price * (1 - trailing_pct))
            else:
                return float(current_price * (1 + trailing_pct))
        except Exception as e:
            self.logger.error("Error calculate_trailing_stop: %s", e)
            return None

    # -------------------------
    # Multi-timeframe combination
    # -------------------------
    def combine_timeframes(self,
                           signals: Dict[str,
                                         str],
                           weights: Optional[Dict[str,
                                                  float]] = None) -> Dict[str,
                                                                          Any]:
        """
        Combine signals across timeframes into an overall bias.
        signals: {'15m': 'BUY', '1h': 'SELL', '4h': 'BUY', '1d': 'BUY'}
        returns: {'bias': 'BULLISH'|'BEARISH'|'NEUTRAL', 'score': float}
        """
        try:
            default_weights = {"15m": 1.0, "1h": 1.5, "4h": 2.0, "1d": 3.0}
            w = weights or default_weights
            score = 0.0
            total = 0.0
            mapping = {"BUY": 1.0, "SELL": -1.0, "NEUTRAL": 0.0,
                       "COMPRAR": 1.0, "VENDER": -1.0, "NEUTRO": 0.0}
            for tf, sig in signals.items():
                weight = float(w.get(tf, 1.0))
                val = mapping.get(sig.upper(), 0.0)
                score += val * weight
                total += weight
            final = score / total if total else 0.0
            if final > 0.25:
                bias = "BULLISH"
            elif final < -0.25:
                bias = "BEARISH"
            else:
                bias = "NEUTRAL"
            return {"bias": bias, "score": float(final)}
        except Exception as e:
            self.logger.error("Error combine_timeframes: %s", e)
            return {"bias": "NEUTRAL", "score": 0.0}

    # -------------------------
    # Position / trade management
    # -------------------------
    def assess_trade(self,
                     symbol: str,
                     side: str,
                     entry_price: float,
                     indicators: Dict[str, float],
                     signals: Dict[str, str],
                     ml_score: Optional[float] = None,
                     suggested_stop: Optional[float] = None,
                     suggested_risk_reward: float = 2.0,
                     setup_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a potential trade and return a dict with recommended:
          - position_size (units)
          - stop_loss price
          - take_profit price
          - estimated_risk (notional)
          - reason / bias
        ml_score: probability in [0,1] (higher = favor buy/long bias)
        """
        try:
            # Normalizar inputs
            entry = float(entry_price)
            atr = float(indicators.get("atr", 0.0) or 0.0)
            volatility = float(indicators.get("volatility", 0.0) or 0.0)
            support = float(indicators.get("support", 0.0) or 0.0)
            resistance = float(indicators.get("resistance", 0.0) or 0.0)
            trend = indicators.get("trend", "NEUTRAL")

            # Combine timeframes
            combined = self.combine_timeframes(signals)
            bias = combined["bias"]
            score = combined["score"]

            # If ML score present, nudge risk/takeprofit
            ml_adj = 1.0
            if ml_score is not None:
                ml_score = float(ml_score)
                # if ml strongly supports side, allow a higher position
                if (side.upper() in ["BUY", "LONG"] and ml_score > 0.6) or (
                        side.upper() in ["SELL", "SHORT"] and ml_score < 0.4):
                    ml_adj = 1.2
                elif 0.45 < ml_score < 0.55:
                    ml_adj = 0.9

            # Determine stop loss
            if suggested_stop:
                stop = float(suggested_stop)
            else:
                stop = self.calculate_dynamic_stop_loss(
                    entry, atr, support, resistance, volatility, trend)

            # position size
            position_units = self.calculate_position_size(
                entry, stop, volatility, trend_strength=abs(score) * 1.0)
            position_units *= ml_adj
            if position_units <= 0:
                return {
                    "allowed": False,
                    "reason": "position_size_zero_or_invalid"}

            # estimated notional risk
            est_risk_notional = abs(entry - stop) * position_units

            # check total risk budget
            total_risk_now = self.compute_total_risk_notional()
            if (total_risk_now +
                est_risk_notional) > (self.account_balance *
                                      self.max_total_risk):
                return {
                    "allowed": False,
                    "reason": "exceeds_total_risk_budget"}

            # check exposure pct
            exposure = self.compute_portfolio_exposure_notional()
            projected_exposure = exposure + (position_units * entry)
            if projected_exposure > (
                    self.account_balance *
                    self.max_exposure_pct):
                return {"allowed": False, "reason": "exceeds_max_exposure_pct"}

            # Calculate take profit
            tp = self.calculate_take_profit(
                entry,
                stop,
                risk_reward=suggested_risk_reward,
                resistance=resistance,
                trend=trend)

            return {
                "allowed": True,
                "symbol": symbol,
                "side": side,
                "entry_price": entry,
                "position_units": float(position_units),
                "stop_loss": float(stop),
                "take_profit": float(tp),
                "estimated_risk_notional": float(est_risk_notional),
                "bias": bias,
                "bias_score": float(score),
                "ml_adj": float(ml_adj),
                "setup_name": setup_name
            }
        except Exception as e:
            self.logger.error("Error assess_trade: %s", e, exc_info=True)
            return {"allowed": False, "reason": "exception"}

    def open_position(self, trade: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Registers an open position in the manager.
        trade should include: symbol, side, entry_price, position_units,
        stop_loss, take_profit, timestamp(optional), etc.
        """
        try:
            t = trade.copy()
            t['timestamp_open'] = t.get(
                'timestamp_open', datetime.utcnow().isoformat())
            t['status'] = 'OPEN'
            # compute notional
            t['notional'] = float(
                t.get('position_units', 0.0)) * float(t.get('entry_price'))
            self.open_positions.append(t)
            self._exposure_cache = None
            self.logger.info(
                "Opened position: %s %s units @ %.2f",
                t.get('symbol'),
                t.get('position_units'),
                t.get('entry_price'))
            return t
        except Exception as e:
            self.logger.error("Error open_position: %s", e)
            return None

    def close_position(self, pos_index: int, exit_price: float,
                       reason: str = "MANUAL") -> Optional[Dict[str, Any]]:
        """
        Close position by index in open_positions.
        Calculates PnL, updates history and account_balance.
        """
        try:
            if pos_index < 0 or pos_index >= len(self.open_positions):
                return None
            pos = self.open_positions.pop(pos_index)
            entry = float(pos.get('entry_price'))
            units = float(pos.get('position_units', 0.0))
            side = pos.get('side', 'LONG').upper()
            pnl = (exit_price - entry) * units if side in [
                'BUY', 'LONG'] else (entry - exit_price) * units
            pnl = float(pnl)
            pos['exit_price'] = float(exit_price)
            pos['timestamp_close'] = datetime.utcnow().isoformat()
            pos['pnl'] = pnl
            pos['status'] = 'CLOSED'
            pos['close_reason'] = reason

            # update account and history
            self.trade_history.append(pos)
            self._recent_closed.append(pos)
            self.account_balance += pnl
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            month_str = datetime.utcnow().strftime("%Y-%m")
            self.daily_pnl[date_str] += pnl
            self.monthly_pnl[month_str] += pnl
            self._exposure_cache = None

            # update setup performance
            setup = pos.get('setup_name')
            if setup:
                perf = self.setup_performance[setup]
                perf['count'] += 1
                perf['total_pnl'] += pnl
                if pnl > 0:
                    perf['wins'] += 1
                else:
                    perf['losses'] += 1

            self.logger.info(
                "Closed position %s @ %.2f -> PnL: %.4f | "
                "New account balance: %.2f",
                pos.get('symbol'),
                exit_price,
                pnl,
                self.account_balance)
            return pos
        except Exception as e:
            self.logger.error("Error close_position: %s", e, exc_info=True)
            return None

    # -------------------------
    # Portfolio metrics
    # -------------------------
    def compute_portfolio_exposure_notional(self) -> float:
        """Sum absolute notional exposure of open positions."""
        if self._exposure_cache is not None:
            return float(self._exposure_cache)
        exposure = 0.0
        try:
            for p in self.open_positions:
                exposure += abs(float(p.get('notional', 0.0)))
        except Exception:
            pass
        self._exposure_cache = exposure
        return float(exposure)

    def compute_total_risk_notional(self) -> float:
        """Sum of estimated risk notional (distance to stop * units)."""
        total = 0.0
        try:
            for p in self.open_positions:
                entry = float(p.get('entry_price', 0.0))
                stop = float(p.get('stop_loss', entry))
                units = float(p.get('position_units', 0.0))
                total += abs(entry - stop) * units
        except Exception:
            pass
        return float(total)

    def current_exposure_pct(self) -> float:
        exposure = self.compute_portfolio_exposure_notional()
        try:
            return float(
                exposure /
                self.account_balance) if self.account_balance else 0.0
        except Exception:
            return 0.0

    # -------------------------
    # Simple backtest engine (walk-forward)
    # -------------------------
    def backtest(
            self,
            df: pd.DataFrame,
            strategy_fn,
            initial_balance: Optional[float] = None,
            verbose: bool = False) -> Dict[str, Any]:
        """
        Run a simple backtest:
        - df must contain columns: timestamp, open, high, low, close, volume
        - strategy_fn(row, history) -> dict: { 'signal': 'BUY'|'SELL'|'HOLD',
          'stop': price, 'rr': float }
        This engine is simple and intended to test risk manager parameters over
        historic OHLC.
        """
        try:
            if initial_balance is None:
                initial_balance = self.account_balance
            balance = float(initial_balance)
            open_pos = None
            closed_trades = []
            for i, row in df.iterrows():
                signal_obj = strategy_fn(row, closed_trades)
                if not signal_obj:
                    continue
                signal = signal_obj.get('signal', 'HOLD')
                if signal == 'BUY' and open_pos is None:
                    entry = float(row['close'])
                    stop = float(signal_obj.get('stop', entry * 0.98))
                    rr = float(signal_obj.get('rr', 2.0))
                    units = self.calculate_position_size(
                        entry, stop, volatility=signal_obj.get(
                            'volatility', 1.0), use_account_balance=balance)
                    if units <= 0:
                        continue
                    open_pos = {
                        'symbol': signal_obj.get(
                            'symbol',
                            'BACKTEST'),
                        'side': 'LONG',
                        'entry_price': entry,
                        'stop_loss': stop,
                        'position_units': units,
                        'take_profit': self.calculate_take_profit(
                            entry,
                            stop,
                            rr
                        )
                    }
                elif open_pos is not None:
                    # Check if stop or tp hit in this candle (simple)
                    low = float(row['low'])
                    high = float(row['high'])
                    if low <= open_pos['stop_loss']:
                        pnl = (
                            open_pos['stop_loss'] - open_pos['entry_price']
                        ) * open_pos['position_units']
                        open_pos['exit_price'] = open_pos['stop_loss']
                        open_pos['pnl'] = pnl
                        closed_trades.append(open_pos)
                        balance += pnl
                        open_pos = None
                    elif high >= open_pos['take_profit']:
                        pnl = (
                            open_pos['take_profit'] - open_pos['entry_price']
                        ) * open_pos['position_units']
                        open_pos['exit_price'] = open_pos['take_profit']
                        open_pos['pnl'] = pnl
                        closed_trades.append(open_pos)
                        balance += pnl
                        open_pos = None
                # optional: record equity curve if verbose
            total_pnl = sum([t.get('pnl', 0.0) for t in closed_trades])
            wins = sum(1 for t in closed_trades if t.get('pnl', 0.0) > 0)
            losses = sum(1 for t in closed_trades if t.get('pnl', 0.0) <= 0)
            result = {
                'initial_balance': initial_balance,
                'final_balance': balance,
                'total_pnl': total_pnl,
                'wins': wins,
                'losses': losses,
                'trades': len(closed_trades)
            }
            self.backtest_results = result
            if verbose:
                self.logger.info("Backtest result: %s", result)
            return result
        except Exception as e:
            self.logger.error("Error in backtest: %s", e, exc_info=True)
            return {}

    # -------------------------
    # Reporting & helpers for UI
    # -------------------------
    def summary(self) -> Dict[str, Any]:
        """Return snapshot summary for UI consumption."""
        try:
            exposure = self.compute_portfolio_exposure_notional()
            total_risk = self.compute_total_risk_notional()
            return {
                "account_balance": float(self.account_balance),
                "open_positions_count": len(self.open_positions),
                "open_positions": self.open_positions,
                "portfolio_exposure_notional": float(exposure),
                "portfolio_exposure_pct": float(
                    exposure / self.account_balance) if self.account_balance
                else 0.0,
                "total_risk_notional": float(total_risk),
                "total_risk_pct": float(
                    total_risk / self.account_balance) if self.account_balance
                else 0.0,
                "recent_trades": list(self._recent_closed)[-20:],
                "setup_performance": dict(self.setup_performance),
                "daily_pnl": dict(self.daily_pnl),
                "monthly_pnl": dict(self.monthly_pnl)
            }
        except Exception as e:
            self.logger.error("Error summary: %s", e)
            return {}

    # -------------------------
    # Sync with Binance Spot account (LIVE)
    # -------------------------
    def sync_spot_account(self, binance_connector, min_notional: float = 10.0):
        """Populate open_positions/exposure from live Binance spot account.
        - Computes notional exposure from non-stable holdings (e.g., BTC, ETH)
        - Approximates BTC cost basis via FIFO from get_my_trades
          to compute unrealized PnL
        """
        try:
            client = getattr(binance_connector, 'client', None)
            if client is None:
                return False

            acct = client.get_account()
            balances = acct.get('balances', [])
            new_positions = []

            # helper to fetch price safely
            def _price(sym):
                try:
                    t = client.get_symbol_ticker(symbol=sym)
                    return float(t['price'])
                except Exception:
                    return None

            # Consider these as stable (ignored for exposure)
            stable = {"USDT", "BUSD", "USDC", "FDUSD", "TUSD"}

            for b in balances:
                asset = b.get('asset')
                free = float(b.get('free', 0) or 0)
                locked = float(b.get('locked', 0) or 0)
                qty = free + locked
                if not asset or qty <= 0:
                    continue
                if asset.upper() in stable:
                    continue
                symbol = f"{asset.upper()}USDT"
                px = _price(symbol)
                if px is None:
                    continue
                notional = qty * px
                if notional < min_notional:
                    continue
                # default position info; entry price unknown unless BTC FIFO
                # computed below
                new_positions.append({
                    'symbol': symbol,
                    'side': 'LONG',
                    'entry_price': None,
                    'position_units': qty,
                    'take_profit': None,
                    'stop_loss': None
                })

            # Compute FIFO cost basis for each detected symbol (USDT pairs)
            try:
                for pos in new_positions:
                    sym = pos.get('symbol')
                    if not sym or not sym.endswith('USDT'):
                        continue
                    if float(pos.get('position_units', 0) or 0) <= 0:
                        continue
                    try:
                        trades = client.get_my_trades(symbol=sym, limit=1000)
                    except Exception:
                        continue
                    lots = []  # FIFO lots
                    trades_sorted = sorted(
                        trades, key=lambda t: int(t.get('time', 0)))
                    for tr in trades_sorted:
                        try:
                            qty = float(tr.get('qty', 0)
                                        or tr.get('qty'.upper(), 0))
                            price = float(tr.get('price', 0))
                            is_buyer = tr.get('isBuyer', False)
                            if qty <= 0 or price <= 0:
                                continue
                            if is_buyer:
                                lots.append({'qty': qty, 'price': price})
                            else:
                                remaining = qty
                                while remaining > 1e-10 and lots:
                                    take = min(lots[0]['qty'], remaining)
                                    lots[0]['qty'] -= take
                                    remaining -= take
                                    if lots[0]['qty'] <= 1e-10:
                                        lots.pop(0)
                        except Exception:
                            continue
                    net_qty = sum(lot['qty'] for lot in lots)
                    if net_qty > 1e-10:
                        cost = sum(lot['qty'] * lot['price'] for lot in lots)
                        avg_cost = cost / net_qty
                        pos['entry_price'] = float(avg_cost)
            except Exception as e:
                self.logger.debug("FIFO cost basis failed: %s", e)

            # replace open_positions and invalidate caches
            self.open_positions = new_positions
            self._exposure_cache = None
            return True
        except Exception as e:
            self.logger.error("sync_spot_account error: %s", e, exc_info=True)
            return False

    # -------------------------
    # Correlation utilities
    # -------------------------
    def update_correlation_matrix(
            self, price_series_dict: Dict[str, pd.Series]):
        """
        price_series_dict: {'BTCUSDT': pd.Series(close),
        'ETHUSDT': pd.Series(close), ...}
        Calculates correlations and stores them.
        """
        try:
            df = pd.DataFrame(price_series_dict).pct_change().dropna()
            corr = df.corr()
            for sym in corr.columns:
                self.correlation_matrix[sym] = corr[sym].to_dict()
            self.logger.info(
                "Correlation matrix updated for %d symbols", len(corr.columns))
            return True
        except Exception as e:
            self.logger.error("Error update_correlation_matrix: %s", e)
            return False

    def check_max_correlation(self, symbol: str) -> float:
        """Return max correlation of a symbol vs portfolio."""
        try:
            if symbol not in self.correlation_matrix:
                return 0.0
            corrs = self.correlation_matrix[symbol]
            max_corr = max(abs(v) for v in corrs.values()) if corrs else 0.0
            return float(max_corr)
        except Exception:
            return 0.0

    # -------------------------
    # Utility: integrate ML output
    # -------------------------
    def integrate_ml_score(self, ml_prob: Optional[float], side: str) -> float:
        """
        Convert ML probability to internal score used to nudge
        sizing/thresholds.
        ml_prob: probability of price up (0-1)
        side: 'BUY' or 'SELL'
        returns a multiplier (>=0.5 and <=1.5)
        """
        try:
            if ml_prob is None:
                return 1.0
            p = float(ml_prob)
            if side.upper() in ["BUY", "LONG"]:
                if p >= 0.7:
                    return 1.4
                if p >= 0.6:
                    return 1.2
                if p >= 0.55:
                    return 1.05
                if p <= 0.45:
                    return 0.8
            else:
                # SELL side -> invert
                if p <= 0.3:
                    return 1.4
                if p <= 0.4:
                    return 1.2
                if p <= 0.45:
                    return 1.05
                if p >= 0.55:
                    return 0.8
            return 1.0
        except Exception as e:
            self.logger.error("integrate_ml_score error: %s", e)
            return 1.0
