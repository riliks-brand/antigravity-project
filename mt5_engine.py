"""
MT5 Engine — Elite v3.0
========================
Professional MetaTrader 5 execution engine with:
- Smart retry decorator
- Dynamic lot sizing (balance-based)
- Spread filter
- Heartbeat monitor
- Kill switch
- Candle close confirmation
"""

import MetaTrader5 as mt5
import time
import datetime
import functools
import logging
from config import Config

logger = logging.getLogger("MT5Engine")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler(Config.LOG_FILE, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("\033[94m%(asctime)s\033[0m [%(levelname)s] %(message)s"))
    logger.addHandler(_ch)


# =========================================
# SMART RETRY DECORATOR
# =========================================

def smart_retry(max_retries=None):
    """Decorator: retries MT5 operations on failure."""
    if max_retries is None:
        max_retries = Config.MAX_RETRIES

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_retries + 2):  # +1 for initial attempt
                try:
                    result = func(*args, **kwargs)
                    if result is not None:
                        return result
                except Exception as e:
                    last_error = e
                    logger.warning("[Retry] %s attempt %d/%d failed: %s",
                                   func.__name__, attempt, max_retries + 1, e)

                if attempt <= max_retries:
                    wait = attempt * 0.5
                    logger.info("[Retry] Waiting %.1fs before retry...", wait)
                    time.sleep(wait)

            logger.error("[Retry] %s EXHAUSTED all %d retries. Last error: %s",
                         func.__name__, max_retries + 1, last_error)
            return None
        return wrapper
    return decorator


# =========================================
# CONNECTION & HEARTBEAT
# =========================================

def connect_to_exness():
    """
    Initializes connection and logs into the MT5 Terminal.
    Prioritizes attaching to an already open terminal.
    """
    # Force shutdown of any stale handles
    mt5.shutdown()
    time.sleep(0.5)

    max_retries = 3
    connected = False

    logger.info("[MT5] Attempting to link with the open terminal...")

    for attempt in range(max_retries):
        if mt5.initialize():
            connected = True
            break
        if mt5.initialize(path=Config.MT5_PATH):
            connected = True
            break

        logger.warning("[MT5] Connection attempt %d failed. Retrying in 2s...", attempt + 1)
        time.sleep(2)

    if not connected:
        error = mt5.last_error()
        logger.error("MT5 Initialization failed. Error code: %s", error)
        if error and error[0] == -6:
            logger.info("[Tip] Terminal reports Auth Error. Please close MT5 and reopen it.")
        else:
            logger.info("[Tip] Make sure MT5 is open and 'Allow Algorithmic Trading' is Checked.")
        return False

    # Attempt Login
    authorized = mt5.login(
        login=Config.LOGIN,
        password=Config.PASSWORD,
        server=Config.SERVER
    )

    if not authorized:
        logger.error("MT5 Login failed for account %s. Error: %s", Config.LOGIN, mt5.last_error())
        return False

    account = mt5.account_info()
    logger.info("Connected & Logged into %s | Balance: %.2f %s | Leverage: 1:%d",
                Config.SERVER, account.balance, account.currency, account.leverage)
    return True


def heartbeat():
    """
    Verify MT5 connection is alive.
    Returns True if healthy, False if dead.
    Attempts auto-reconnect on failure.
    """
    try:
        info = mt5.account_info()
        if info is None:
            logger.warning("[Heartbeat] MT5 connection lost. Attempting reconnect...")
            return connect_to_exness()
        return True
    except Exception as e:
        logger.error("[Heartbeat] Exception: %s. Attempting reconnect...", e)
        return connect_to_exness()


def get_server_time(symbol=None):
    """Get current server time from MT5 tick (NOT local clock)."""
    sym = symbol or Config.FOREX_SYMBOL
    tick = mt5.symbol_info_tick(sym)
    if tick:
        return datetime.datetime.utcfromtimestamp(tick.time)
    return datetime.datetime.utcnow()  # Fallback


# =========================================
# SYMBOL INFO & TICK
# =========================================

def get_tick_info(symbol):
    """
    Retrieves latest tick and symbol properties.
    Ensures symbol is selected in Market Watch.
    """
    if not mt5.symbol_select(symbol, True):
        logger.warning("[MT5] Symbol '%s' not found. Trying to select...", symbol)

    info = mt5.symbol_info(symbol)
    if not info:
        logger.error("Failed to retrieve info for symbol %s.", symbol)
        return None, None

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        logger.warning("[MT5] Waiting for first tick for %s...", symbol)
        return None, info

    return tick, info


def get_spread_points(symbol):
    """Get current spread in points."""
    tick, info = get_tick_info(symbol)
    if not tick or not info or info.point <= 0:
        return -1
    return (tick.ask - tick.bid) / info.point


# =========================================
# DYNAMIC LOT SIZING (Balance-based)
# =========================================

def calculate_lot_size(symbol, sl_points, risk_multiplier=1.0):
    """
    Calculates lot size based on:
    - Account balance
    - Risk % per trade (from Config)
    - Stop loss distance (in points)
    - Equity curve risk multiplier

    Formula: lot = (balance * risk%) / (SL_points * tick_value)
    """
    account = mt5.account_info()
    if not account:
        logger.error("[LotSize] Cannot get account info.")
        return 0.0

    tick, info = get_tick_info(symbol)
    if not info:
        return 0.0

    balance = account.balance
    risk_pct = Config.RISK_PERCENT_PER_TRADE / 100.0
    risk_amount = balance * risk_pct * risk_multiplier

    tick_value = info.trade_tick_value
    if tick_value is None or tick_value <= 0:
        tick_value = 1.0  # Fallback

    if sl_points <= 0:
        logger.warning("[LotSize] SL points is <= 0. Using minimum lot.")
        return info.volume_min

    lot_size = risk_amount / (sl_points * tick_value)

    # Clamp to broker limits
    lot_step = info.volume_step
    lot_size = round(lot_size / lot_step) * lot_step
    lot_size = max(lot_size, info.volume_min)
    lot_size = min(lot_size, info.volume_max)

    digits = 0
    if "." in str(lot_step):
        digits = len(str(lot_step).split('.')[1])

    lot_size = round(float(lot_size), digits)

    logger.info("[LotSize] Balance: %.2f | Risk: %.2f$ (x%.1f) | SL: %d pts | Lot: %.2f",
                balance, risk_amount, risk_multiplier, sl_points, lot_size)
    return lot_size


# =========================================
# EXECUTION ENGINE
# =========================================

@smart_retry(max_retries=Config.MAX_RETRIES)
def execute_forex_trade(action, symbol, sl_points, tp_points,
                        risk_multiplier=1.0, signal_time_ms=None):
    """
    Builds the MT5 order payload and executes the trade.
    Returns (success, result_dict) where result_dict contains ticket, prices, etc.

    Args:
        action: "BUY" or "SELL"
        symbol: MT5 symbol name
        sl_points: Stop loss in points
        tp_points: Take profit in points
        risk_multiplier: From equity curve protection (0.0-1.0)
        signal_time_ms: Unix timestamp (ms) when signal was generated
    """
    tick, info = get_tick_info(symbol)
    if not tick or not info:
        return None

    # --- SPREAD FILTER ---
    spread_points = (tick.ask - tick.bid) / info.point if info.point > 0 else 0
    if spread_points > Config.MAX_SPREAD_POINTS:
        logger.warning("[SPREAD FILTER] Spread %.1f > Max %d. Trade REJECTED.",
                       spread_points, Config.MAX_SPREAD_POINTS)
        return None

    # --- DYNAMIC LOT SIZING ---
    lot = calculate_lot_size(symbol, sl_points, risk_multiplier)
    if lot <= 0:
        return None

    # --- BUILD ORDER ---
    price = tick.ask if action == 'BUY' else tick.bid
    point = info.point

    if action == 'BUY':
        sl_price = price - (sl_points * point)
        tp_price = price + (tp_points * point)
        order_type = mt5.ORDER_TYPE_BUY
    else:
        sl_price = price + (sl_points * point)
        tp_price = price - (tp_points * point)
        order_type = mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot),
        "type": order_type,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": Config.SLIPPAGE_TOLERANCE,
        "magic": Config.MAGIC_NUMBER,
        "comment": "Elite_Bot_v3",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    fill_start_ms = time.time() * 1000
    result = mt5.order_send(request)
    fill_end_ms = time.time() * 1000

    if result is None:
        logger.error("[Execute] MT5 order_send returned None.")
        return None

    # Handle fill mode rejection
    if result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
        logger.info("[Execute] Retrying with FOK filling...")
        request["type_filling"] = mt5.ORDER_FILLING_FOK
        result = mt5.order_send(request)
        if result is None:
            return None

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error("[Execute] Order failed: %s (code: %s)", result.comment, result.retcode)
        return None

    # --- SUCCESS ---
    actual_fill_price = result.price if hasattr(result, 'price') and result.price > 0 else price
    latency_ms = fill_end_ms - (signal_time_ms if signal_time_ms else fill_start_ms)

    logger.info(
        "[EXECUTED] ✅ %s %s | Ticket: #%s | Lot: %.2f | Price: %.5f | SL: %.5f | TP: %.5f | Spread: %.1f | Latency: %.0fms",
        action, symbol, result.order, lot, actual_fill_price,
        sl_price, tp_price, spread_points, latency_ms,
    )

    return {
        "success": True,
        "ticket": result.order,
        "symbol": symbol,
        "direction": action,
        "volume": lot,
        "expected_price": price,
        "filled_price": actual_fill_price,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "tp2_price": tp_price,  # Will be overridden by caller for TP2
        "spread_points": spread_points,
        "signal_time_ms": signal_time_ms or fill_start_ms,
        "fill_time_ms": fill_end_ms,
        "latency_ms": latency_ms,
    }


# =========================================
# KILL SWITCH
# =========================================

def check_kill_switch():
    """
    Returns True if trading should STOP (daily loss exceeded).
    """
    account = mt5.account_info()
    if not account:
        return False

    daily_change_pct = ((account.equity - account.balance) / account.balance * 100) if account.balance > 0 else 0

    if daily_change_pct < -Config.MAX_DAILY_LOSS_PCT:
        logger.critical(
            "[KILL SWITCH] ⛔ Daily loss %.1f%% exceeds %.1f%%. ALL TRADING HALTED.",
            abs(daily_change_pct), Config.MAX_DAILY_LOSS_PCT,
        )
        return True
    return False


# =========================================
# CLOSE ALL POSITIONS (Emergency)
# =========================================

def close_all_positions():
    """Emergency: close all bot-managed positions."""
    positions = mt5.positions_get()
    if not positions:
        logger.info("[CloseAll] No positions found.")
        return

    closed = 0
    for pos in positions:
        if pos.magic != Config.MAGIC_NUMBER:
            continue

        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(pos.symbol)
        price = tick.bid if pos.type == 0 else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": price,
            "deviation": Config.SLIPPAGE_TOLERANCE,
            "magic": Config.MAGIC_NUMBER,
            "comment": "Emergency_Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            closed += 1
            logger.info("[CloseAll] Closed #%s", pos.ticket)
        else:
            logger.error("[CloseAll] Failed to close #%s", pos.ticket)

    logger.info("[CloseAll] Closed %d/%d positions.", closed, len([p for p in positions if p.magic == Config.MAGIC_NUMBER]))


# =========================================
# ACCOUNT INFO HELPERS
# =========================================

def get_account_balance():
    """Get current account balance."""
    account = mt5.account_info()
    return account.balance if account else 0.0


def get_account_equity():
    """Get current account equity."""
    account = mt5.account_info()
    return account.equity if account else 0.0
