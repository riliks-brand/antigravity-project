"""
OTC Scraper — WebSocket Interception + DOM Live Price Module
============================================================
Connects to Olymp Trade's running browser via CDP (Chrome DevTools Protocol),
intercepts WebSocket messages to grab historical candle data on load,
and polls the DOM every second for real-time price ticks.

The scraped ticks are aggregated into OHLCV candles in-memory.
"""

import os
import time
import json
import threading
import datetime
import numpy as np
import pandas as pd
from collections import deque
from playwright.sync_api import sync_playwright


class OTCScraper:
    """
    Singleton-style OTC data provider.
    
    Usage:
        scraper = OTCScraper()
        scraper.start()                    # Starts background threads
        df = scraper.get_candles(1000)      # Returns last 1000 candles as DataFrame
        price = OTCScraper.get_last_price() # Static access to latest tick
    """
    
    _instance = None
    _last_price = 0.0
    _lock = threading.Lock()
    
    # In-memory tick and candle storage
    _ticks = deque(maxlen=500000)       # Raw 1s ticks: (timestamp, price)
    _candles = deque(maxlen=10000)      # Aggregated OHLCV candles
    _ws_history_loaded = threading.Event()
    _running = False
    
    # Candle interval in seconds (default 60s = 1 minute candles)
    CANDLE_INTERVAL = 60
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, cdp_port=9225, candle_interval=60):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.cdp_port = cdp_port
        OTCScraper.CANDLE_INTERVAL = candle_interval
        self._page = None
        self._browser = None
        self._playwright = None
        self._dom_thread = None
        self._aggregator_thread = None
    
    def start(self):
        """
        Starts the OTC scraper:
        1. Connects to the browser via CDP
        2. Sets up WebSocket interception for historical candle data
        3. Starts DOM polling thread for real-time ticks
        4. Starts candle aggregation thread
        """
        if OTCScraper._running:
            print("\033[93m[OTC Scraper] Already running.\033[0m")
            return
        
        OTCScraper._running = True
        
        print(f"\n\033[96m{'='*55}\033[0m")
        print(f"\033[96m       🎯 OTC SCRAPER: INITIALIZING\033[0m")
        print(f"\033[96m{'='*55}\033[0m")
        
        # Connect to the existing browser via CDP
        print(f"[OTC Scraper] Connecting to browser on CDP port {self.cdp_port}...")
        self._playwright = sync_playwright().start()
        
        try:
            self._browser = self._playwright.chromium.connect_over_cdp(f"http://localhost:{self.cdp_port}")
            self._page = self._browser.contexts[0].pages[0]
            print("\033[92m[OTC Scraper] Connected to browser successfully.\033[0m")
        except Exception as e:
            print(f"\033[91m[OTC Scraper] Failed to connect to browser: {e}\033[0m")
            print("[OTC Scraper] Make sure the browser is running with --remote-debugging-port=9225")
            OTCScraper._running = False
            return
        
        # Step 1: Intercept WebSocket for historical candles
        self._setup_ws_interception()
        
        # Step 2: Start DOM polling for live ticks
        self._dom_thread = threading.Thread(target=self._dom_poll_loop, daemon=True)
        self._dom_thread.start()
        
        # Step 3: Start candle aggregator
        self._aggregator_thread = threading.Thread(target=self._aggregator_loop, daemon=True)
        self._aggregator_thread.start()
        
        print(f"\033[92m[OTC Scraper] All systems operational. Candle interval: {self.CANDLE_INTERVAL}s\033[0m")
        print(f"\033[96m{'='*55}\033[0m\n")
    
    def _setup_ws_interception(self):
        """
        Intercepts WebSocket frames from the Olymp Trade platform to extract
        historical candle data that the platform sends on page load.
        """
        print("[OTC Scraper] Setting up WebSocket interception...")
        
        try:
            # Use CDP session to listen to WebSocket frames
            cdp_session = self._page.context.new_cdp_session(self._page)
            cdp_session.send("Network.enable")
            
            # Track WebSocket connections
            ws_connections = {}
            
            def on_ws_created(params):
                ws_id = params.get('requestId', '')
                url = params.get('url', '')
                print(f"\033[94m[WS Intercept] WebSocket opened: {url[:80]}...\033[0m")
                ws_connections[ws_id] = url
            
            def on_ws_frame(params):
                """Parse incoming WebSocket frames for candle/price data."""
                try:
                    payload = params.get('response', {}).get('payloadData', '')
                    if not payload:
                        return
                    
                    # Try to parse as JSON (Olymp Trade sends JSON frames)
                    try:
                        data = json.loads(payload)
                    except (json.JSONDecodeError, ValueError):
                        return
                    
                    # Look for candle data structures
                    # Olymp Trade typically sends candles in arrays with OHLC fields
                    candles_extracted = self._extract_candles_from_ws(data)
                    
                    if candles_extracted > 0:
                        print(f"\033[92m[WS Intercept] Extracted {candles_extracted} historical candles from WebSocket!\033[0m")
                        OTCScraper._ws_history_loaded.set()
                        
                except Exception as e:
                    pass  # Silently skip non-parseable frames
            
            cdp_session.on("Network.webSocketCreated", on_ws_created)
            cdp_session.on("Network.webSocketFrameReceived", on_ws_frame)
            
            print("\033[92m[WS Intercept] WebSocket listeners active. Waiting for historical data...\033[0m")
            
            # Give the platform time to send historical candles
            # (They're typically sent within 3-5 seconds of page load)
            time.sleep(5)
            
            if not OTCScraper._ws_history_loaded.is_set():
                print("\033[93m[WS Intercept] No historical WS data detected yet. Will rely on DOM polling + gradual collection.\033[0m")
            
        except Exception as e:
            print(f"\033[93m[WS Intercept] WebSocket interception setup warning: {e}\033[0m")
            print("[WS Intercept] Falling back to pure DOM scraping mode.")
    
    def _extract_candles_from_ws(self, data):
        """
        Attempts to extract candle data from various JSON structures
        that Olymp Trade may use in its WebSocket communication.
        Returns the number of candles extracted.
        """
        count = 0
        
        # Strategy 1: Look for 'candles' or 'history' keys
        candle_keys = ['candles', 'history', 'bars', 'data', 'chart']
        
        if isinstance(data, dict):
            for key in candle_keys:
                if key in data and isinstance(data[key], list):
                    for candle in data[key]:
                        if self._parse_single_candle(candle):
                            count += 1
            
            # Strategy 2: Look for nested structures
            for key, value in data.items():
                if isinstance(value, dict):
                    for sub_key in candle_keys:
                        if sub_key in value and isinstance(value[sub_key], list):
                            for candle in value[sub_key]:
                                if self._parse_single_candle(candle):
                                    count += 1
        
        elif isinstance(data, list):
            # Direct array of candles
            for item in data:
                if self._parse_single_candle(item):
                    count += 1
        
        return count
    
    def _parse_single_candle(self, candle):
        """
        Parses a single candle dict with various possible key names.
        Returns True if successfully parsed and stored.
        """
        if not isinstance(candle, dict):
            return False
        
        # Map common field name variations
        open_keys  = ['open', 'o', 'Open', 'openPrice']
        high_keys  = ['high', 'h', 'High', 'highPrice', 'max']
        low_keys   = ['low', 'l', 'Low', 'lowPrice', 'min']
        close_keys = ['close', 'c', 'Close', 'closePrice']
        time_keys  = ['time', 't', 'timestamp', 'ts', 'date', 'at']
        vol_keys   = ['volume', 'v', 'vol', 'Volume']
        
        def find_val(keys):
            for k in keys:
                if k in candle:
                    return candle[k]
            return None
        
        o = find_val(open_keys)
        h = find_val(high_keys)
        l = find_val(low_keys)
        c = find_val(close_keys)
        t = find_val(time_keys)
        v = find_val(vol_keys) or 0
        
        if o is not None and h is not None and l is not None and c is not None:
            try:
                ts = t if t else time.time()
                # If timestamp is in milliseconds, convert
                if isinstance(ts, (int, float)) and ts > 1e12:
                    ts = ts / 1000.0
                
                with OTCScraper._lock:
                    OTCScraper._candles.append({
                        'time': datetime.datetime.fromtimestamp(float(ts)),
                        'open': float(o),
                        'high': float(h),
                        'low': float(l),
                        'close': float(c),
                        'real_volume': float(v)
                    })
                    OTCScraper._last_price = float(c)
                return True
            except (ValueError, TypeError):
                return False
        
        return False
    
    def _dom_poll_loop(self):
        """
        Background thread: polls the DOM every 1 second to extract the live price.
        """
        print("[OTC Scraper] DOM polling thread started.")
        
        # JavaScript to extract the live price from Olymp Trade's DOM
        # This tries multiple selectors that the platform may use
        js_extract_price = """
        () => {
            // Strategy 1: Common quote value selectors
            const selectors = [
                '.quote__val',
                '.trading-deal__price-value',
                '[data-test="asset-price"]',
                '.current-price',
                '.price-value',
                '.instrument-price',
                '.chart-price',
                '.deal-price__value'
            ];
            
            for (const sel of selectors) {
                const el = document.querySelector(sel);
                if (el) {
                    const text = el.textContent.replace(/[^0-9.]/g, '');
                    const price = parseFloat(text);
                    if (price > 0) return price;
                }
            }
            
            // Strategy 2: Find any element with a price-like large number
            // Look for elements that update frequently (likely the price)
            const allSpans = document.querySelectorAll('span, div');
            for (const el of allSpans) {
                if (el.children.length === 0) {
                    const text = el.textContent.trim();
                    // Match price patterns like "1.23456" or "83421.50"
                    if (/^\\d{1,6}\\.\\d{2,6}$/.test(text)) {
                        const fontSize = window.getComputedStyle(el).fontSize;
                        const size = parseInt(fontSize);
                        // Price elements are typically displayed in larger font
                        if (size >= 18) {
                            return parseFloat(text);
                        }
                    }
                }
            }
            
            return null;
        }
        """
        
        consecutive_failures = 0
        
        while OTCScraper._running:
            try:
                price = self._page.evaluate(js_extract_price)
                
                if price and price > 0:
                    now = time.time()
                    with OTCScraper._lock:
                        OTCScraper._ticks.append((now, price))
                        OTCScraper._last_price = price
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures == 10:
                        print("\033[93m[OTC Scraper] Warning: 10 consecutive DOM poll failures. Price element may have changed.\033[0m")
                    if consecutive_failures == 60:
                        print("\033[91m[OTC Scraper] Critical: 60 consecutive failures. Check if the platform DOM has updated.\033[0m")
                
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures % 30 == 0:
                    print(f"\033[91m[OTC DOM] Polling error (attempt {consecutive_failures}): {str(e)[:60]}\033[0m")
            
            time.sleep(1)
    
    def _aggregator_loop(self):
        """
        Background thread: aggregates raw ticks into OHLCV candles.
        Runs every CANDLE_INTERVAL seconds.
        """
        print(f"[OTC Scraper] Candle aggregator started ({self.CANDLE_INTERVAL}s interval).")
        
        while OTCScraper._running:
            time.sleep(self.CANDLE_INTERVAL)
            
            now = time.time()
            window_start = now - self.CANDLE_INTERVAL
            
            with OTCScraper._lock:
                # Collect ticks within the last candle window
                window_ticks = [(t, p) for t, p in OTCScraper._ticks if t >= window_start and t < now]
            
            if not window_ticks:
                continue
            
            prices = [p for _, p in window_ticks]
            candle = {
                'time': datetime.datetime.fromtimestamp(window_start),
                'open': prices[0],
                'high': max(prices),
                'low': min(prices),
                'close': prices[-1],
                'real_volume': len(prices)  # Tick volume
            }
            
            with OTCScraper._lock:
                OTCScraper._candles.append(candle)
            
            total_candles = len(OTCScraper._candles)
            if total_candles % 50 == 0:
                print(f"\033[94m[OTC Scraper] Total candles in memory: {total_candles}\033[0m")
    
    def get_candles(self, count=1000):
        """
        Returns the last `count` candles as a pandas DataFrame
        matching MT5/yfinance format (columns: open, high, low, close, real_volume).
        """
        with OTCScraper._lock:
            candle_list = list(OTCScraper._candles)
        
        if not candle_list:
            print("\033[93m[OTC Scraper] No candles available yet.\033[0m")
            return None
        
        # Take the last `count` candles
        candle_list = candle_list[-count:]
        
        df = pd.DataFrame(candle_list)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        
        print(f"\033[92m[OTC Scraper] Returning {len(df)} candles. Latest price: {df['close'].iloc[-1]:.5f}\033[0m")
        return df
    
    @staticmethod
    def get_last_price():
        """Returns the most recent price tick."""
        return OTCScraper._last_price
    
    @staticmethod
    def get_candle_count():
        """Returns the number of candles currently in memory."""
        return len(OTCScraper._candles)
    
    @staticmethod
    def is_ready(min_candles=120):
        """
        Returns True if we have enough candles for the LSTM sequence.
        Default minimum = Config.SEQUENCE_LENGTH (120).
        """
        return len(OTCScraper._candles) >= min_candles
    
    def stop(self):
        """Stops the scraper gracefully."""
        OTCScraper._running = False
        print("[OTC Scraper] Shutting down...")
        try:
            self._playwright.stop()
        except:
            pass


# ===== MODULE-LEVEL SINGLETON =====
_scraper_instance = None

def get_otc_scraper(candle_interval=60):
    """
    Returns the singleton OTCScraper instance.
    Creates and starts it on first call.
    """
    global _scraper_instance
    if _scraper_instance is None:
        _scraper_instance = OTCScraper(candle_interval=candle_interval)
        _scraper_instance.start()
    return _scraper_instance
