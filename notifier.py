"""
Notifier — Elite v3.1
=======================
Spam-controlled Telegram notification system.

Features:
- Rate limiting (max N messages/minute)
- Message grouping (batches rapid-fire events)
- Emergency bypass (kill switch alerts always sent)
- Daily summary with P&L and feature importance
- Silent fallback to file logging if Telegram fails
"""

import os
import time
import datetime
import logging
import threading
from collections import deque
from config import Config

from logging_setup import setup_module_logger
logger = setup_module_logger("Notifier", Config.LOG_FILE, console_color="\033[96m")


# =========================================
# LOAD CREDENTIALS FROM .env
# =========================================

def _load_telegram_credentials():
    """Load bot token and chat ID from .env file."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), Config.TELEGRAM_ENV_FILE)

    if not os.path.exists(env_path):
        return None, None

    token = None
    chat_id = None

    try:
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "TELEGRAM_BOT_TOKEN":
                    token = value
                elif key == "TELEGRAM_CHAT_ID":
                    chat_id = value
    except Exception as e:
        logger.warning("[Notifier] Failed to read .env: %s", e)

    return token, chat_id


# =========================================
# TELEGRAM SENDER (Thread-Safe, Rate-Limited)
# =========================================

class TelegramNotifier:
    """Thread-safe, rate-limited Telegram notification system."""

    def __init__(self):
        self.enabled = Config.TELEGRAM_ENABLED
        self.token, self.chat_id = _load_telegram_credentials()
        self.lock = threading.Lock()

        # Rate limiting
        self.message_timestamps = deque(maxlen=100)
        self.max_per_minute = Config.TELEGRAM_MAX_MESSAGES_PER_MINUTE

        # Message queue for grouping
        self.pending_messages = []
        self.last_flush_time = time.time()

        # Validate
        if self.enabled:
            if not self.token or not self.chat_id:
                logger.warning("[Notifier] TELEGRAM_ENABLED=True but no credentials found. "
                               "Run: python setup_telegram.py")
                self.enabled = False
            else:
                logger.info("[Notifier] Telegram notifications ACTIVE.")

    def _is_rate_limited(self):
        """Check if we've exceeded the message rate limit."""
        now = time.time()
        # Remove timestamps older than 60 seconds
        while self.message_timestamps and now - self.message_timestamps[0] > 60:
            self.message_timestamps.popleft()
        return len(self.message_timestamps) >= self.max_per_minute

    def _send_raw(self, text, parse_mode="Markdown"):
        """Low-level send to Telegram API. Returns True on success."""
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }
            response = requests.post(url, json=payload, timeout=10)
            data = response.json()

            if data.get("ok"):
                self.message_timestamps.append(time.time())
                return True
            else:
                logger.warning("[Notifier] Telegram API error: %s", data.get("description"))
                return False

        except ImportError:
            logger.warning("[Notifier] 'requests' not installed. Falling back to log.")
            return False
        except Exception as e:
            logger.warning("[Notifier] Send failed: %s. Falling back to log.", e)
            return False

    def send(self, text, level="TRADE", force=False):
        """
        Public send method with rate limiting and level filtering.

        Args:
            text: Message text (Markdown supported)
            level: "TRADE", "CLOSE", "EMERGENCY", "DAILY", "INFO"
            force: If True, bypass rate limiting (for emergencies)
        """
        if not self.enabled:
            logger.debug("[Notifier] (Disabled) %s: %s", level, text[:80])
            return

        with self.lock:
            # Emergency messages bypass rate limiting
            if force or level == "EMERGENCY":
                self._send_raw(text)
                return

            # Check rate limit
            if self._is_rate_limited():
                logger.info("[Notifier] Rate limited. Queuing message.")
                self.pending_messages.append(text)
                return

            # Check if level is allowed
            if level not in Config.TELEGRAM_ALERT_LEVELS:
                return

            self._send_raw(text)

    def flush_queue(self):
        """Send any queued messages as a grouped batch."""
        if not self.enabled or not self.pending_messages:
            return

        with self.lock:
            if len(self.pending_messages) == 0:
                return

            # Group pending messages
            grouped = "\n\n".join(self.pending_messages[-5:])  # Max 5 grouped
            header = f"📥 *Grouped Alerts* ({len(self.pending_messages)} messages)\n\n"
            self._send_raw(header + grouped)
            self.pending_messages.clear()

    # =========================================
    # HIGH-LEVEL ALERT METHODS
    # =========================================

    def trade_opened(self, direction, symbol, xgb_prob=None, rf_prob=0.5, final_prob=0.5,
                     entry_price=0, sl_price=0, tp_price=0, lot_size=0, penalty=0,
                     lstm_prob=None, **kwargs):
        """Send trade entry alert. Accepts xgb_prob or lstm_prob (backward compat)."""
        # Backward compatibility: accept lstm_prob as alias
        model_prob = xgb_prob if xgb_prob is not None else (lstm_prob if lstm_prob is not None else 0.5)
        emoji = "\U0001f7e2" if direction == "BUY" else "\U0001f534"
        text = (
            f"{emoji} *{direction} {symbol}*\n\n"
            f"\U0001f9e0 XGB: `{model_prob:.1%}` | RF: `{rf_prob:.1%}`\n"
            f"\U0001f3af Final: `{final_prob:.1%}` (penalty: `{penalty:.3f}`)\n\n"
            f"\U0001f4b0 Entry: `{entry_price:.5f}`\n"
            f"\U0001f534 SL: `{sl_price:.5f}`\n"
            f"\U0001f7e2 TP: `{tp_price:.5f}`\n"
            f"\U0001f4e6 Lot: `{lot_size:.2f}`"
        )
        self.send(text, level="TRADE")

    def trade_closed(self, ticket, symbol, direction, pnl, reason, daily_pnl):
        """Send trade closure alert."""
        emoji = "\U0001f4b0" if pnl >= 0 else "\U0001f4a5"
        pnl_sign = "+" if pnl >= 0 else ""
        text = (
            f"{emoji} *Trade #{ticket} CLOSED*\n\n"
            f"\U0001f4c8 {direction} {symbol}\n"
            f"\U0001f4b5 P&L: `{pnl_sign}${pnl:.2f}`\n"
            f"\U0001f4cb Reason: `{reason}`\n"
            f"\U0001f4ca Daily P&L: `{pnl_sign if daily_pnl >= 0 else ''}${daily_pnl:.2f}`"
        )
        self.send(text, level="CLOSE")

    def emergency(self, message):
        """Send emergency alert (bypasses rate limiting)."""
        text = f"\U0001f6a8 *EMERGENCY*\n\n{message}"
        self.send(text, level="EMERGENCY", force=True)

    def kill_switch_activated(self, daily_loss_pct, balance):
        """Kill switch notification."""
        text = (
            f"\u26d4 *KILL SWITCH ACTIVATED*\n\n"
            f"\U0001f4c9 Daily Loss: `{daily_loss_pct:.1f}%`\n"
            f"\U0001f4b0 Balance: `${balance:.2f}`\n"
            f"\U0001f6d1 All positions closed. Bot paused until tomorrow."
        )
        self.send(text, level="EMERGENCY", force=True)

    def connection_lost(self):
        """MT5 connection lost notification."""
        self.emergency("\U0001f4e1 MT5 Connection LOST. Attempting reconnect...")

    def connection_restored(self):
        """MT5 connection restored notification."""
        text = "\u2705 *MT5 Connection Restored*\nBot resuming normal operations."
        self.send(text, level="INFO")

    def daily_summary(self, stats, feature_importance=None):
        """
        Send end-of-day summary with P&L and optional feature importance.
        """
        total = stats.get("total", 0)
        win_rate = stats.get("win_rate", 0)
        pf = stats.get("profit_factor", 0)
        daily_pnl = stats.get("daily_pnl", 0)
        max_dd = stats.get("max_dd", 0)

        emoji = "\U0001f7e2" if daily_pnl >= 0 else "\U0001f534"

        text = (
            f"\U0001f4ca *DAILY SUMMARY*\n"
            f"{'='*30}\n\n"
            f"{emoji} Daily P&L: `{'+'if daily_pnl>=0 else ''}${daily_pnl:.2f}`\n"
            f"\U0001f3af Win Rate: `{win_rate:.1f}%`\n"
            f"\U0001f4c8 Trades: `{total}`\n"
            f"\U0001f4b9 Profit Factor: `{pf:.2f}`\n"
            f"\U0001f4c9 Max Drawdown: `${max_dd:.2f}`\n"
        )

        # Feature importance section
        if feature_importance:
            text += f"\n\U0001f9e0 *Top Features Driving Decisions:*\n"
            for i, (feat, imp) in enumerate(feature_importance[:5], 1):
                bar = "\u2588" * int(imp * 20)
                text += f"  {i}. `{feat}`: {bar} `{imp:.1%}`\n"

        self.send(text, level="DAILY")

    def signal_rejected(self, direction, reason, prob):
        """Log rejected signal (not sent to Telegram, just logged)."""
        logger.info("[Notifier] Rejected %s (prob=%.4f): %s", direction, prob, reason)


# =========================================
# SINGLETON INSTANCE
# =========================================

_notifier_instance = None
_notifier_lock = threading.Lock()


def get_notifier():
    """Get the global TelegramNotifier instance (thread-safe singleton)."""
    global _notifier_instance
    if _notifier_instance is None:
        with _notifier_lock:
            if _notifier_instance is None:
                _notifier_instance = TelegramNotifier()
    return _notifier_instance
