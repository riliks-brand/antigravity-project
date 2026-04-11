"""
Trade Manager — Elite v3.0
============================
Thread-safe State Machine for tracking every trade from signal to close.

States: SIGNAL → PENDING → OPEN → PARTIAL_CLOSED → CLOSED

Features:
- Thread-safe locks per trade (prevents race conditions)
- JSON persistence with corruption fallback (rebuild from MT5)
- Trailing Stop (ATR-based)
- Breakeven move after TP1
- Partial Close (50% at TP1)
- Signal Deduplication
- Cooldown after consecutive losses
- Latency & Slippage tracking
- Equity Curve Protection
"""

import json
import os
import time
import threading
import datetime
import csv
import logging
from enum import Enum
from config import Config

# ===== Setup Persistent Logger =====
logger = logging.getLogger("TradeManager")
logger.setLevel(logging.DEBUG)
_fh = logging.FileHandler(Config.LOG_FILE, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_fh)
_ch = logging.StreamHandler()
_ch.setFormatter(logging.Formatter("\033[96m%(asctime)s\033[0m [%(levelname)s] %(message)s"))
logger.addHandler(_ch)


class TradeState(Enum):
    SIGNAL = "SIGNAL"
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIAL_CLOSED = "PARTIAL_CLOSED"
    CLOSED = "CLOSED"


class ManagedTrade:
    """Represents a single managed trade with all its metadata."""

    def __init__(self, ticket, symbol, direction, volume, entry_price,
                 sl_price, tp1_price, tp2_price, magic, signal_time=None):
        self.ticket = ticket
        self.symbol = symbol
        self.direction = direction          # "BUY" or "SELL"
        self.original_volume = volume
        self.current_volume = volume
        self.entry_price = entry_price
        self.sl_price = sl_price
        self.tp1_price = tp1_price          # First target (partial close)
        self.tp2_price = tp2_price          # Second target (trailing)
        self.magic = magic
        self.state = TradeState.OPEN

        # Timing
        self.signal_time = signal_time or datetime.datetime.utcnow().isoformat()
        self.fill_time = datetime.datetime.utcnow().isoformat()
        self.close_time = None

        # Execution Quality
        self.expected_price = entry_price
        self.filled_price = entry_price     # Updated after fill confirmation
        self.slippage_points = 0.0
        self.latency_ms = 0.0

        # Management Flags
        self.tp1_hit = False
        self.sl_moved_to_be = False
        self.trailing_active = False
        self.trailing_sl = None

        # Outcome
        self.close_price = None
        self.pnl = 0.0
        self.close_reason = ""              # "TP1", "TP2", "SL", "TRAILING", "MANUAL", "KILL_SWITCH"

        # Thread Safety
        self.lock = threading.Lock()

    def to_dict(self):
        """Serialize to dict for JSON persistence."""
        return {
            "ticket": self.ticket,
            "symbol": self.symbol,
            "direction": self.direction,
            "original_volume": self.original_volume,
            "current_volume": self.current_volume,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp1_price": self.tp1_price,
            "tp2_price": self.tp2_price,
            "magic": self.magic,
            "state": self.state.value,
            "signal_time": self.signal_time,
            "fill_time": self.fill_time,
            "close_time": self.close_time,
            "expected_price": self.expected_price,
            "filled_price": self.filled_price,
            "slippage_points": self.slippage_points,
            "latency_ms": self.latency_ms,
            "tp1_hit": self.tp1_hit,
            "sl_moved_to_be": self.sl_moved_to_be,
            "trailing_active": self.trailing_active,
            "trailing_sl": self.trailing_sl,
            "close_price": self.close_price,
            "pnl": self.pnl,
            "close_reason": self.close_reason,
        }

    @staticmethod
    def from_dict(d):
        """Deserialize from dict."""
        trade = ManagedTrade(
            ticket=d["ticket"],
            symbol=d["symbol"],
            direction=d["direction"],
            volume=d["original_volume"],
            entry_price=d["entry_price"],
            sl_price=d["sl_price"],
            tp1_price=d["tp1_price"],
            tp2_price=d["tp2_price"],
            magic=d["magic"],
            signal_time=d.get("signal_time"),
        )
        trade.current_volume = d.get("current_volume", trade.original_volume)
        trade.state = TradeState(d.get("state", "OPEN"))
        trade.fill_time = d.get("fill_time")
        trade.close_time = d.get("close_time")
        trade.expected_price = d.get("expected_price", trade.entry_price)
        trade.filled_price = d.get("filled_price", trade.entry_price)
        trade.slippage_points = d.get("slippage_points", 0.0)
        trade.latency_ms = d.get("latency_ms", 0.0)
        trade.tp1_hit = d.get("tp1_hit", False)
        trade.sl_moved_to_be = d.get("sl_moved_to_be", False)
        trade.trailing_active = d.get("trailing_active", False)
        trade.trailing_sl = d.get("trailing_sl")
        trade.close_price = d.get("close_price")
        trade.pnl = d.get("pnl", 0.0)
        trade.close_reason = d.get("close_reason", "")
        return trade


class TradeManager:
    """
    Central trade management engine.
    Thread-safe, persistent, and crash-recoverable.
    """

    def __init__(self):
        self.active_trades = {}             # ticket_id -> ManagedTrade
        self.closed_trades = []             # History of closed trades
        self.global_lock = threading.Lock()

        # Cooldown State
        self.consecutive_losses = 0
        self.cooldown_until = None          # datetime when cooldown expires

        # Signal Deduplication
        self.last_signal_direction = None
        self.last_signal_candle_index = -999

        # Daily P&L tracking
        self.daily_pnl = 0.0
        self.daily_start_balance = None
        self.last_reset_date = None

        # Equity Curve
        self.equity_history = []

        # Load persisted state
        self._load_state()
        logger.info("TradeManager initialized. Active trades: %d", len(self.active_trades))

    # =========================================
    # PERSISTENCE — JSON with corruption fallback
    # =========================================

    def _load_state(self):
        """Load active trades from JSON. If corrupted, rebuild from MT5."""
        filepath = Config.ACTIVE_TRADES_FILE
        if not os.path.exists(filepath):
            return

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            for td in data.get("active_trades", []):
                trade = ManagedTrade.from_dict(td)
                if trade.state not in (TradeState.CLOSED,):
                    self.active_trades[trade.ticket] = trade

            self.consecutive_losses = data.get("consecutive_losses", 0)
            cooldown_str = data.get("cooldown_until")
            if cooldown_str:
                self.cooldown_until = datetime.datetime.fromisoformat(cooldown_str)
            self.daily_pnl = data.get("daily_pnl", 0.0)

            logger.info("[Persistence] Loaded %d active trades from JSON.", len(self.active_trades))

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("[Persistence] JSON CORRUPTED: %s. Rebuilding from MT5...", e)
            self._rebuild_from_mt5()

    def _rebuild_from_mt5(self):
        """Fallback: Rebuild active trade state from MT5 positions."""
        try:
            import MetaTrader5 as mt5
            positions = mt5.positions_get()
            if positions is None:
                logger.error("[Rebuild] MT5 returned None for positions.")
                return

            self.active_trades = {}
            for pos in positions:
                if pos.magic != Config.MAGIC_NUMBER:
                    continue

                direction = "BUY" if pos.type == 0 else "SELL"
                trade = ManagedTrade(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    direction=direction,
                    volume=pos.volume,
                    entry_price=pos.price_open,
                    sl_price=pos.sl,
                    tp1_price=pos.tp,
                    tp2_price=pos.tp,
                    magic=pos.magic,
                )
                trade.state = TradeState.OPEN
                trade.filled_price = pos.price_open
                self.active_trades[trade.ticket] = trade

            logger.info("[Rebuild] Recovered %d trades from MT5.", len(self.active_trades))
            self._save_state()

        except Exception as e:
            logger.error("[Rebuild] Failed to rebuild from MT5: %s", e)

    def _save_state(self):
        """Persist active trades to JSON atomically."""
        filepath = Config.ACTIVE_TRADES_FILE
        temp_path = filepath + ".tmp"

        data = {
            "active_trades": [t.to_dict() for t in self.active_trades.values()],
            "consecutive_losses": self.consecutive_losses,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "daily_pnl": self.daily_pnl,
            "last_updated": datetime.datetime.utcnow().isoformat(),
        }

        try:
            # Write to temp file first (atomic write pattern)
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)
            # Replace original only if write succeeded
            if os.path.exists(filepath):
                os.remove(filepath)
            os.rename(temp_path, filepath)
        except Exception as e:
            logger.error("[Persistence] Failed to save state: %s", e)

    # =========================================
    # TRADE REGISTRATION
    # =========================================

    def register_trade(self, ticket, symbol, direction, volume,
                       entry_price, expected_price, sl_price,
                       tp1_price, tp2_price, signal_time_ms, fill_time_ms):
        """
        Register a newly opened trade into the manager.
        Calculates latency and slippage automatically.
        """
        with self.global_lock:
            import MetaTrader5 as mt5
            info = mt5.symbol_info(symbol)
            point = info.point if info else 0.00001

            trade = ManagedTrade(
                ticket=ticket,
                symbol=symbol,
                direction=direction,
                volume=volume,
                entry_price=entry_price,
                sl_price=sl_price,
                tp1_price=tp1_price,
                tp2_price=tp2_price,
                magic=Config.MAGIC_NUMBER,
                signal_time=datetime.datetime.utcfromtimestamp(signal_time_ms / 1000).isoformat()
                           if signal_time_ms > 1e9 else None,
            )

            # Execution Quality Metrics
            trade.expected_price = expected_price
            trade.filled_price = entry_price
            trade.slippage_points = abs(entry_price - expected_price) / point if point > 0 else 0
            trade.latency_ms = fill_time_ms - signal_time_ms if signal_time_ms and fill_time_ms else 0

            trade.fill_time = datetime.datetime.utcnow().isoformat()

            self.active_trades[ticket] = trade
            self._save_state()

            # Log Execution Quality
            self._log_execution_quality(trade)

            logger.info(
                "[REGISTERED] Ticket #%s | %s %s %.5f | Vol: %.2f | SL: %.5f | TP1: %.5f | Slippage: %.1f pts | Latency: %.0f ms",
                ticket, direction, symbol, entry_price, volume,
                sl_price, tp1_price, trade.slippage_points, trade.latency_ms,
            )
            return trade

    # =========================================
    # TICK-LEVEL MANAGEMENT (called every tick)
    # =========================================

    def on_tick(self, symbol, bid, ask, current_atr):
        """
        Called on every price tick. Manages all active trades for the symbol.
        Thread-safe per trade.
        """
        import MetaTrader5 as mt5

        for ticket, trade in list(self.active_trades.items()):
            if trade.symbol != symbol:
                continue
            if trade.state == TradeState.CLOSED:
                continue

            with trade.lock:
                current_price = bid if trade.direction == "BUY" else ask

                # --- CHECK TP1 HIT (Partial Close) ---
                if not trade.tp1_hit:
                    tp1_hit = False
                    if trade.direction == "BUY" and current_price >= trade.tp1_price:
                        tp1_hit = True
                    elif trade.direction == "SELL" and current_price <= trade.tp1_price:
                        tp1_hit = True

                    if tp1_hit:
                        self._execute_partial_close(trade, current_price, mt5)
                        continue

                # --- TRAILING STOP LOGIC ---
                if trade.tp1_hit and trade.trailing_active:
                    self._update_trailing_stop(trade, current_price, current_atr, mt5)

                # --- CHECK SL HIT (including trailing) ---
                active_sl = trade.trailing_sl if trade.trailing_sl else trade.sl_price
                sl_hit = False
                if trade.direction == "BUY" and current_price <= active_sl:
                    sl_hit = True
                elif trade.direction == "SELL" and current_price >= active_sl:
                    sl_hit = True

                if sl_hit:
                    reason = "TRAILING_SL" if trade.trailing_active else "SL"
                    self._close_trade(trade, current_price, reason, mt5)

                # --- CHECK TP2 HIT (Final Target) ---
                if trade.tp1_hit:
                    tp2_hit = False
                    if trade.direction == "BUY" and current_price >= trade.tp2_price:
                        tp2_hit = True
                    elif trade.direction == "SELL" and current_price <= trade.tp2_price:
                        tp2_hit = True

                    if tp2_hit:
                        self._close_trade(trade, current_price, "TP2", mt5)

    def _execute_partial_close(self, trade, current_price, mt5):
        """Close 50% of the position at TP1 and move SL to breakeven."""
        close_volume = round(trade.current_volume * Config.PARTIAL_CLOSE_PCT, 2)
        info = mt5.symbol_info(trade.symbol)
        if info:
            close_volume = max(close_volume, info.volume_min)
            lot_step = info.volume_step
            close_volume = round(round(close_volume / lot_step) * lot_step, 2)

        # Build partial close request
        close_type = mt5.ORDER_TYPE_SELL if trade.direction == "BUY" else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(trade.symbol).bid if trade.direction == "BUY" else mt5.symbol_info_tick(trade.symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": trade.symbol,
            "volume": float(close_volume),
            "type": close_type,
            "position": trade.ticket,
            "price": price,
            "deviation": Config.SLIPPAGE_TOLERANCE,
            "magic": Config.MAGIC_NUMBER,
            "comment": "TP1_Partial_Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            trade.tp1_hit = True
            trade.current_volume = round(trade.current_volume - close_volume, 2)
            trade.state = TradeState.PARTIAL_CLOSED

            logger.info(
                "[TP1 HIT] Ticket #%s | Closed %.2f lots at %.5f | Remaining: %.2f lots",
                trade.ticket, close_volume, current_price, trade.current_volume,
            )

            # Move SL to Breakeven
            if Config.MOVE_SL_TO_BE_AFTER_TP1:
                self._move_sl_to_breakeven(trade, mt5)

            # Activate trailing stop
            trade.trailing_active = True
            trade.trailing_sl = trade.entry_price  # Start trailing from BE

            self._save_state()
        else:
            retcode = result.retcode if result else "None"
            comment = result.comment if result else "No result"
            logger.error("[TP1 PARTIAL CLOSE FAILED] Ticket #%s | Code: %s | %s", trade.ticket, retcode, comment)

            # Retry with FOK
            if result and result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
                request["type_filling"] = mt5.ORDER_FILLING_FOK
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    trade.tp1_hit = True
                    trade.current_volume = round(trade.current_volume - close_volume, 2)
                    trade.state = TradeState.PARTIAL_CLOSED
                    trade.trailing_active = True
                    trade.trailing_sl = trade.entry_price
                    if Config.MOVE_SL_TO_BE_AFTER_TP1:
                        self._move_sl_to_breakeven(trade, mt5)
                    self._save_state()
                    logger.info("[TP1 RETRY OK] Ticket #%s partial close succeeded with FOK.", trade.ticket)

    def _move_sl_to_breakeven(self, trade, mt5):
        """Modify the position's SL to the entry price (breakeven)."""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": trade.symbol,
            "position": trade.ticket,
            "sl": trade.entry_price,
            "tp": trade.tp2_price,
            "magic": Config.MAGIC_NUMBER,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            trade.sl_price = trade.entry_price
            trade.sl_moved_to_be = True
            logger.info("[BREAKEVEN] Ticket #%s SL moved to entry: %.5f", trade.ticket, trade.entry_price)
        else:
            retcode = result.retcode if result else "None"
            logger.warning("[BREAKEVEN FAILED] Ticket #%s | Code: %s", trade.ticket, retcode)

    def _update_trailing_stop(self, trade, current_price, current_atr, mt5):
        """Update trailing stop based on ATR."""
        trail_distance = current_atr * Config.TRAILING_STOP_ATR_MULT

        if trade.direction == "BUY":
            new_trail_sl = current_price - trail_distance
            if trade.trailing_sl is None or new_trail_sl > trade.trailing_sl:
                trade.trailing_sl = new_trail_sl
                # Modify on MT5
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": trade.symbol,
                    "position": trade.ticket,
                    "sl": new_trail_sl,
                    "tp": trade.tp2_price,
                    "magic": Config.MAGIC_NUMBER,
                }
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    trade.sl_price = new_trail_sl
                    logger.debug("[TRAIL] Ticket #%s SL ↑ %.5f", trade.ticket, new_trail_sl)
                self._save_state()

        elif trade.direction == "SELL":
            new_trail_sl = current_price + trail_distance
            if trade.trailing_sl is None or new_trail_sl < trade.trailing_sl:
                trade.trailing_sl = new_trail_sl
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": trade.symbol,
                    "position": trade.ticket,
                    "sl": new_trail_sl,
                    "tp": trade.tp2_price,
                    "magic": Config.MAGIC_NUMBER,
                }
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    trade.sl_price = new_trail_sl
                    logger.debug("[TRAIL] Ticket #%s SL ↓ %.5f", trade.ticket, new_trail_sl)
                self._save_state()

    def _close_trade(self, trade, close_price, reason, mt5):
        """Fully close the remaining position."""
        close_type = mt5.ORDER_TYPE_SELL if trade.direction == "BUY" else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(trade.symbol).bid if trade.direction == "BUY" else mt5.symbol_info_tick(trade.symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": trade.symbol,
            "volume": float(trade.current_volume),
            "type": close_type,
            "position": trade.ticket,
            "price": price,
            "deviation": Config.SLIPPAGE_TOLERANCE,
            "magic": Config.MAGIC_NUMBER,
            "comment": f"Close_{reason}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            trade.state = TradeState.CLOSED
            trade.close_price = close_price
            trade.close_reason = reason
            trade.close_time = datetime.datetime.utcnow().isoformat()

            # Calculate P&L
            if trade.direction == "BUY":
                trade.pnl = (close_price - trade.entry_price) * trade.original_volume
            else:
                trade.pnl = (trade.entry_price - close_price) * trade.original_volume

            # Update daily P&L
            self.daily_pnl += trade.pnl

            # Update consecutive loss counter
            if trade.pnl < 0:
                self.consecutive_losses += 1
                if self.consecutive_losses >= Config.COOLDOWN_AFTER_LOSSES:
                    self.cooldown_until = datetime.datetime.utcnow() + datetime.timedelta(
                        minutes=Config.COOLDOWN_DURATION_MINUTES
                    )
                    logger.warning(
                        "[COOLDOWN ACTIVATED] %d consecutive losses. Pausing until %s",
                        self.consecutive_losses, self.cooldown_until.isoformat(),
                    )
            else:
                self.consecutive_losses = 0

            # Move to history
            self.closed_trades.append(trade.to_dict())
            del self.active_trades[trade.ticket]
            self._save_state()
            self._log_trade_history(trade)

            logger.info(
                "[CLOSED] Ticket #%s | %s | Reason: %s | P&L: %.2f | Daily P&L: %.2f",
                trade.ticket, trade.direction, reason, trade.pnl, self.daily_pnl,
            )
        else:
            retcode = result.retcode if result else "None"
            comment = result.comment if result else "No result"
            logger.error("[CLOSE FAILED] Ticket #%s | Code: %s | %s", trade.ticket, retcode, comment)

            # Retry with FOK
            if result and result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
                request["type_filling"] = mt5.ORDER_FILLING_FOK
                retry = mt5.order_send(request)
                if retry and retry.retcode == mt5.TRADE_RETCODE_DONE:
                    trade.state = TradeState.CLOSED
                    trade.close_price = close_price
                    trade.close_reason = reason
                    trade.close_time = datetime.datetime.utcnow().isoformat()
                    if trade.direction == "BUY":
                        trade.pnl = (close_price - trade.entry_price) * trade.original_volume
                    else:
                        trade.pnl = (trade.entry_price - close_price) * trade.original_volume
                    self.daily_pnl += trade.pnl
                    if trade.pnl < 0:
                        self.consecutive_losses += 1
                    else:
                        self.consecutive_losses = 0
                    self.closed_trades.append(trade.to_dict())
                    del self.active_trades[trade.ticket]
                    self._save_state()
                    self._log_trade_history(trade)
                    logger.info("[CLOSE RETRY OK] Ticket #%s closed with FOK.", trade.ticket)

    # =========================================
    # GUARDS — Pre-trade validation
    # =========================================

    def can_trade(self, direction, candle_index):
        """
        Master gate: checks ALL conditions before allowing a trade.
        Returns (allowed: bool, reason: str)
        """
        # 1. Cooldown check
        if self.cooldown_until:
            now = datetime.datetime.utcnow()
            if now < self.cooldown_until:
                remaining = (self.cooldown_until - now).total_seconds() / 60
                return False, f"COOLDOWN active. {remaining:.0f} min remaining after {self.consecutive_losses} losses."
            else:
                # Cooldown expired
                self.cooldown_until = None
                self.consecutive_losses = 0
                logger.info("[COOLDOWN EXPIRED] Resuming trading.")

        # 2. Max concurrent trades
        open_count = sum(1 for t in self.active_trades.values()
                         if t.state in (TradeState.OPEN, TradeState.PARTIAL_CLOSED))
        if open_count >= Config.MAX_CONCURRENT_TRADES:
            return False, f"MAX_CONCURRENT_TRADES reached ({open_count}/{Config.MAX_CONCURRENT_TRADES})."

        # 3. Daily loss kill switch
        if self.daily_start_balance and self.daily_start_balance > 0:
            daily_loss_pct = abs(self.daily_pnl) / self.daily_start_balance * 100
            if self.daily_pnl < 0 and daily_loss_pct >= Config.MAX_DAILY_LOSS_PCT:
                return False, f"KILL SWITCH: Daily loss {daily_loss_pct:.1f}% exceeds {Config.MAX_DAILY_LOSS_PCT}%."

        # 4. Signal deduplication
        if (direction == self.last_signal_direction and
                abs(candle_index - self.last_signal_candle_index) < Config.MIN_CANDLES_BETWEEN_TRADES):
            candles_since = abs(candle_index - self.last_signal_candle_index)
            return False, f"DUPLICATE SIGNAL: Same {direction} within {candles_since} candles (min: {Config.MIN_CANDLES_BETWEEN_TRADES})."

        return True, "OK"

    def update_signal_tracker(self, direction, candle_index):
        """Mark this signal as the latest for deduplication."""
        self.last_signal_direction = direction
        self.last_signal_candle_index = candle_index

    def reset_daily_stats(self, current_balance):
        """Reset daily P&L. Call at the start of each trading day."""
        today = datetime.date.today()
        if self.last_reset_date != today:
            self.daily_pnl = 0.0
            self.daily_start_balance = current_balance
            self.last_reset_date = today
            logger.info("[DAILY RESET] Balance: %.2f | Date: %s", current_balance, today)

    # =========================================
    # EQUITY CURVE PROTECTION
    # =========================================

    def get_risk_multiplier(self, current_equity):
        """
        Returns a risk multiplier (0.0 to 1.0) based on equity curve health.
        If equity drops below its moving average, reduce risk.
        """
        self.equity_history.append(current_equity)

        # Keep only the last N data points
        if len(self.equity_history) > Config.EQUITY_MA_PERIOD * 2:
            self.equity_history = self.equity_history[-Config.EQUITY_MA_PERIOD * 2:]

        if len(self.equity_history) < Config.EQUITY_MA_PERIOD:
            return 1.0  # Not enough data, trade at full risk

        ma = sum(self.equity_history[-Config.EQUITY_MA_PERIOD:]) / Config.EQUITY_MA_PERIOD

        if current_equity < ma:
            logger.warning(
                "[EQUITY PROTECTION] Equity %.2f < MA(%.0f) %.2f → Risk reduced to %.0f%%",
                current_equity, Config.EQUITY_MA_PERIOD, ma, Config.EQUITY_RISK_REDUCTION * 100,
            )
            return Config.EQUITY_RISK_REDUCTION

        return 1.0

    # =========================================
    # SPREAD FILTER
    # =========================================

    @staticmethod
    def check_spread(symbol):
        """Check if spread is acceptable. Returns (ok, spread_points)."""
        import MetaTrader5 as mt5
        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        if not tick or not info or info.point <= 0:
            return False, -1

        spread_points = (tick.ask - tick.bid) / info.point
        if spread_points > Config.MAX_SPREAD_POINTS:
            return False, spread_points
        return True, spread_points

    # =========================================
    # SESSION FILTER
    # =========================================

    @staticmethod
    def is_in_trading_session(symbol=None):
        """
        Check if current server time is within an active session.
        Uses MT5 server time, NOT local clock.
        """
        import MetaTrader5 as mt5

        if not Config.TRADE_ONLY_IN_SESSIONS:
            return True, "Session filter disabled"

        sym = symbol or Config.FOREX_SYMBOL
        tick = mt5.symbol_info_tick(sym)
        if not tick:
            return True, "Cannot get server time, allowing trade"

        server_time = datetime.datetime.utcfromtimestamp(tick.time)
        hour = server_time.hour

        london = Config.SESSION_LONDON[0] <= hour < Config.SESSION_LONDON[1]
        ny = Config.SESSION_NY[0] <= hour < Config.SESSION_NY[1]
        asia = Config.SESSION_ASIA[0] <= hour < Config.SESSION_ASIA[1]

        if london or ny or asia:
            session_name = "London" if london else ("New York" if ny else "Asia")
            return True, f"In {session_name} session (Server Hour: {hour})"
        else:
            return False, f"Outside all sessions (Server Hour: {hour})"

    # =========================================
    # LOGGING — Execution Quality + Trade History
    # =========================================

    def _log_execution_quality(self, trade):
        """Log slippage & latency to dedicated file."""
        try:
            filepath = Config.EXECUTION_QUALITY_LOG
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(
                    f"{trade.fill_time} | Ticket #{trade.ticket} | {trade.direction} {trade.symbol} | "
                    f"Expected: {trade.expected_price:.5f} | Filled: {trade.filled_price:.5f} | "
                    f"Slippage: {trade.slippage_points:.1f} pts | Latency: {trade.latency_ms:.0f} ms\n"
                )
        except Exception as e:
            logger.error("[ExecQuality] Log write failed: %s", e)

    def _log_trade_history(self, trade):
        """Append closed trade to CSV history."""
        try:
            filepath = Config.TRADING_HISTORY_FILE
            file_exists = os.path.isfile(filepath)

            fieldnames = [
                "close_time", "ticket", "symbol", "direction",
                "volume", "entry_price", "close_price",
                "sl_price", "tp1_price", "tp2_price",
                "slippage_points", "latency_ms",
                "pnl", "close_reason",
                "tp1_hit", "trailing_active",
            ]

            with open(filepath, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "close_time": trade.close_time,
                    "ticket": trade.ticket,
                    "symbol": trade.symbol,
                    "direction": trade.direction,
                    "volume": trade.original_volume,
                    "entry_price": trade.entry_price,
                    "close_price": trade.close_price,
                    "sl_price": trade.sl_price,
                    "tp1_price": trade.tp1_price,
                    "tp2_price": trade.tp2_price,
                    "slippage_points": trade.slippage_points,
                    "latency_ms": trade.latency_ms,
                    "pnl": trade.pnl,
                    "close_reason": trade.close_reason,
                    "tp1_hit": trade.tp1_hit,
                    "trailing_active": trade.trailing_active,
                })
        except Exception as e:
            logger.error("[TradeHistory] Log write failed: %s", e)

    def log_rejected_trade(self, direction, reason, probability, similarity_pct=0):
        """Log rejected signals to detect over-filtering."""
        try:
            filepath = Config.REJECTED_TRADES_LOG
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(
                    f"{datetime.datetime.utcnow().isoformat()} | {direction} | "
                    f"Prob: {probability:.4f} | Similarity: {similarity_pct:.1f}% | "
                    f"Reason: {reason}\n"
                )
        except Exception as e:
            logger.error("[RejectedTrades] Log write failed: %s", e)

    # =========================================
    # STATUS & MONITORING
    # =========================================

    def print_status(self):
        """Print a formatted status report of all managed trades."""
        print(f"\n\033[96m{'='*65}\033[0m")
        print(f"\033[96m       📊 TRADE MANAGER STATUS\033[0m")
        print(f"\033[96m{'='*65}\033[0m")
        print(f"  Active Trades   : {len(self.active_trades)}")
        print(f"  Daily P&L       : {'🟢' if self.daily_pnl >= 0 else '🔴'} ${self.daily_pnl:.2f}")
        print(f"  Consec. Losses  : {self.consecutive_losses}")
        print(f"  Cooldown Active : {'YES ⏸️' if self.cooldown_until and datetime.datetime.utcnow() < self.cooldown_until else 'NO ✅'}")

        for ticket, trade in self.active_trades.items():
            state_emoji = {
                TradeState.OPEN: "🟢",
                TradeState.PARTIAL_CLOSED: "🟡",
                TradeState.CLOSED: "⚪",
            }.get(trade.state, "❓")

            print(f"\n  {state_emoji} #{ticket} | {trade.direction} {trade.symbol}")
            print(f"     Entry: {trade.entry_price:.5f} | SL: {trade.sl_price:.5f} | TP1: {trade.tp1_price:.5f}")
            print(f"     Volume: {trade.current_volume:.2f} / {trade.original_volume:.2f}")
            print(f"     TP1 Hit: {'✅' if trade.tp1_hit else '❌'} | Trailing: {'✅' if trade.trailing_active else '❌'}")

        print(f"\033[96m{'='*65}\033[0m\n")

    def get_stats(self):
        """Return performance statistics for monitoring."""
        if not self.closed_trades:
            return {"total": 0, "win_rate": 0, "profit_factor": 0, "max_dd": 0}

        total = len(self.closed_trades)
        wins = sum(1 for t in self.closed_trades if t.get("pnl", 0) > 0)
        losses = total - wins

        gross_profit = sum(t.get("pnl", 0) for t in self.closed_trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t.get("pnl", 0) for t in self.closed_trades if t.get("pnl", 0) < 0))

        win_rate = (wins / total * 100) if total > 0 else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        # Max Drawdown calculation
        cumulative = 0
        peak = 0
        max_dd = 0
        for t in self.closed_trades:
            cumulative += t.get("pnl", 0)
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_dd": max_dd,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "daily_pnl": self.daily_pnl,
        }
