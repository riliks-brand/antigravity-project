"""
News Filter — Elite v3.0
==========================
Scrapes economic calendar events and blocks trading
around high-impact news releases.

Features:
- ForexFactory scraping with daily caching
- Configurable buffer (±15 min by default)
- Currency-specific filtering
- Fallback to allow trading if scraping fails
"""

import datetime
import os
import json
import logging
import re
from config import Config

logger = logging.getLogger("NewsFilter")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler(Config.LOG_FILE, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("\033[93m%(asctime)s\033[0m [%(levelname)s] %(message)s"))
    logger.addHandler(_ch)

# Cache file for today's events
NEWS_CACHE_FILE = "news_cache.json"


def _get_currencies_for_symbol(symbol):
    """
    Extract relevant currencies from a symbol.
    e.g., "EURUSD" -> ["EUR", "USD"]
    """
    symbol = symbol.upper().replace("M", "").replace(".", "")
    # Common pairs
    currency_pairs = {
        "EURUSD": ["EUR", "USD"],
        "GBPUSD": ["GBP", "USD"],
        "USDJPY": ["USD", "JPY"],
        "USDCHF": ["USD", "CHF"],
        "AUDUSD": ["AUD", "USD"],
        "USDCAD": ["USD", "CAD"],
        "NZDUSD": ["NZD", "USD"],
        "EURJPY": ["EUR", "JPY"],
        "GBPJPY": ["GBP", "JPY"],
        "EURGBP": ["EUR", "GBP"],
        "XAUUSD": ["XAU", "USD"],
        "BTCUSD": ["BTC", "USD"],
    }

    if symbol in currency_pairs:
        return currency_pairs[symbol]

    # Fallback: try to extract 3-char codes
    if len(symbol) >= 6:
        return [symbol[:3], symbol[3:6]]

    return ["USD"]  # Default


def fetch_news_events():
    """
    Fetch today's high-impact economic events.

    Strategy:
    1. Check local cache first
    2. If stale, fetch from ForexFactory
    3. If fetch fails, return empty (allow trading)

    Returns:
        list of dicts: [{"time": datetime, "currency": str, "impact": str, "event": str}, ...]
    """
    today = datetime.date.today().isoformat()

    # Check cache
    if os.path.exists(NEWS_CACHE_FILE):
        try:
            with open(NEWS_CACHE_FILE, "r") as f:
                cache = json.load(f)
            if cache.get("date") == today and cache.get("events"):
                logger.debug("[News] Using cached events for %s (%d events).",
                             today, len(cache["events"]))
                return _parse_cached_events(cache["events"])
        except (json.JSONDecodeError, KeyError):
            logger.warning("[News] Cache corrupted. Refetching.")

    # Fetch fresh data
    events = _scrape_forexfactory()

    # Cache it
    if events:
        try:
            cache_data = {
                "date": today,
                "events": [
                    {
                        "time": e["time"].isoformat() if isinstance(e["time"], datetime.datetime) else e["time"],
                        "currency": e["currency"],
                        "impact": e["impact"],
                        "event": e["event"],
                    }
                    for e in events
                ],
            }
            with open(NEWS_CACHE_FILE, "w") as f:
                json.dump(cache_data, f, indent=2)
            logger.info("[News] Cached %d events for %s.", len(events), today)
        except Exception as e:
            logger.warning("[News] Failed to cache events: %s", e)

    return events


def _scrape_forexfactory():
    """
    Scrape ForexFactory calendar for today's high-impact events.
    Returns list of event dicts or empty list on failure.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        url = "https://www.forexfactory.com/calendar?day=today"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            logger.warning("[News] ForexFactory returned status %d.", response.status_code)
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        events = []
        current_time = None

        # Find calendar rows
        rows = soup.select("tr.calendar__row")
        for row in rows:
            # Time cell
            time_cell = row.select_one("td.calendar__time")
            if time_cell:
                time_text = time_cell.get_text(strip=True)
                if time_text:
                    current_time = _parse_ff_time(time_text)

            # Impact
            impact_cell = row.select_one("td.calendar__impact span")
            if not impact_cell:
                continue

            impact_classes = impact_cell.get("class", [])
            impact = "low"
            for cls in impact_classes:
                if "high" in cls.lower() or "red" in cls.lower():
                    impact = "high"
                    break
                elif "medium" in cls.lower() or "orange" in cls.lower() or "ora" in cls.lower():
                    impact = "medium"
                    break

            # Only care about high impact
            if impact != "high":
                continue

            # Currency
            currency_cell = row.select_one("td.calendar__currency")
            currency = currency_cell.get_text(strip=True) if currency_cell else ""

            # Event name
            event_cell = row.select_one("td.calendar__event")
            event_name = event_cell.get_text(strip=True) if event_cell else ""

            if current_time and currency:
                events.append({
                    "time": current_time,
                    "currency": currency.upper(),
                    "impact": impact,
                    "event": event_name,
                })

        logger.info("[News] Scraped %d high-impact events from ForexFactory.", len(events))
        return events

    except ImportError:
        logger.warning("[News] 'requests' or 'beautifulsoup4' not installed. News filter disabled.")
        return []
    except Exception as e:
        logger.warning("[News] Scraping failed: %s. Allowing trading.", e)
        return []


def _parse_ff_time(time_str):
    """Parse ForexFactory time string (e.g., '8:30am') to datetime."""
    try:
        today = datetime.date.today()
        # Handle formats like "8:30am", "2:00pm", "Tentative", "All Day"
        time_str = time_str.strip().lower()
        if "tentative" in time_str or "all day" in time_str or not time_str:
            return None

        # Parse AM/PM format
        match = re.match(r'(\d{1,2}):(\d{2})(am|pm)', time_str)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            ampm = match.group(3)

            if ampm == "pm" and hour != 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0

            # ForexFactory shows times in ET (Eastern Time) → convert to UTC (+5 or +4 DST)
            # Approximation: use +4 for summer, +5 for winter
            utc_offset = 4  # Summer (DST)
            hour = (hour + utc_offset) % 24

            return datetime.datetime(today.year, today.month, today.day, hour, minute)

        return None
    except Exception:
        return None


def _parse_cached_events(raw_events):
    """Parse events from cache (times stored as ISO strings)."""
    events = []
    for e in raw_events:
        ev_time = e.get("time")
        if ev_time and isinstance(ev_time, str):
            try:
                ev_time = datetime.datetime.fromisoformat(ev_time)
            except ValueError:
                continue

        events.append({
            "time": ev_time,
            "currency": e.get("currency", ""),
            "impact": e.get("impact", "high"),
            "event": e.get("event", ""),
        })
    return events


# =========================================
# MAIN API
# =========================================

def is_news_window(symbol=None, server_time=None):
    """
    Check if we are within the news blackout window.

    Args:
        symbol: Trading symbol (to match currencies)
        server_time: Current server time (datetime). If None, uses UTC now.

    Returns:
        (is_blocked: bool, reason: str)
    """
    if not Config.NEWS_FILTER_ENABLED:
        return False, "News filter disabled"

    symbol = symbol or Config.FOREX_SYMBOL
    now = server_time or datetime.datetime.utcnow()
    currencies = _get_currencies_for_symbol(symbol)

    events = fetch_news_events()
    if not events:
        return False, "No events found (or fetch failed)"

    before_mins = Config.NEWS_BLOCK_MINUTES_BEFORE
    after_mins = Config.NEWS_BLOCK_MINUTES_AFTER

    for event in events:
        ev_time = event.get("time")
        if not ev_time or not isinstance(ev_time, datetime.datetime):
            continue

        # Check currency relevance
        ev_currency = event.get("currency", "").upper()
        if ev_currency not in currencies:
            continue

        # Check time window
        window_start = ev_time - datetime.timedelta(minutes=before_mins)
        window_end = ev_time + datetime.timedelta(minutes=after_mins)

        if window_start <= now <= window_end:
            event_name = event.get("event", "Unknown")
            mins_to_event = (ev_time - now).total_seconds() / 60

            if mins_to_event > 0:
                reason = f"NEWS BLOCK: '{event_name}' ({ev_currency}) in {mins_to_event:.0f} min"
            else:
                reason = f"NEWS BLOCK: '{event_name}' ({ev_currency}) was {abs(mins_to_event):.0f} min ago"

            logger.warning("[NEWS] ⚠️ %s", reason)
            return True, reason

    return False, "No relevant news in window"


def get_upcoming_events(symbol=None, hours_ahead=4):
    """
    Get list of upcoming high-impact events for monitoring display.
    """
    symbol = symbol or Config.FOREX_SYMBOL
    now = datetime.datetime.utcnow()
    currencies = _get_currencies_for_symbol(symbol)
    events = fetch_news_events()

    upcoming = []
    for event in events:
        ev_time = event.get("time")
        if not ev_time or not isinstance(ev_time, datetime.datetime):
            continue
        if event.get("currency", "").upper() not in currencies:
            continue
        if now <= ev_time <= now + datetime.timedelta(hours=hours_ahead):
            mins = (ev_time - now).total_seconds() / 60
            upcoming.append({
                "event": event.get("event", ""),
                "currency": event.get("currency", ""),
                "minutes_until": round(mins),
                "time": ev_time.strftime("%H:%M UTC"),
            })

    return upcoming
