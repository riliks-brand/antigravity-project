import MetaTrader5 as mt5
from config import Config
from playwright.sync_api import sync_playwright

class TradeExecutor:
    def __init__(self):
        pass

    def execute_mt5(self, action="buy", volume=0.01):
        """
        Executes a paper trade on MT5 using credentials in config.py
        Action can be 'buy' or 'sell'
        """
        print("Initializing MT5 Connection...")
        if not mt5.initialize(login=Config.LOGIN, server=Config.SERVER, password=Config.PASSWORD):
            print("MT5 initialize() failed, error code =", mt5.last_error())
            return None

        symbol = Config.SYMBOL
        
        # Select symbol
        if not mt5.symbol_select(symbol, True):
            print(f"Failed to select {symbol}")
            mt5.shutdown()
            return None

        # Get point and price
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"{symbol} not found")
            mt5.shutdown()
            return None
            
        point = symbol_info.point
        digits = symbol_info.digits
        if action.lower() == "buy":
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
            sl = round(price - 200 * point, digits)
            tp = round(price + 200 * point, digits)
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
            sl = round(price + 200 * point, digits)
            tp = round(price - 200 * point, digits)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": "Antigravity Paper Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        print(f"Sending MT5 Order: {action.upper()} {volume} {symbol} at {price}")
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print("OrderSend failed, retcode =", result.retcode)
            # fallback filling modes
            print("Retrying with ORDER_FILLING_RETURN...")
            request["type_filling"] = mt5.ORDER_FILLING_RETURN
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                 request["type_filling"] = mt5.ORDER_FILLING_FOK
                 print("Retrying with ORDER_FILLING_FOK...")
                 result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                result_dict = result._asdict()
                print(f"Failed again. Error Dictionary: {result_dict}")
                mt5.shutdown()
                return None
                
        print(f"Trade Success! Ticket ID: {result.order}")
        mt5.shutdown()
        return {
            "ticket": result.order,
            "entry_price": price,
            "sl": sl,
            "tp": tp
        }

    def execute_web(self, action="buy", url="https://olymptrade.com/platform", user_data_dir=None):
        """
        Executes a paper trade on Olymp Trade via native CDP connection.
        Uses domcontentloaded for speed, aggressive JS button finding, and 3-attempt retry logic.
        """
        import os
        import time
        import datetime
        import subprocess
        
        print(f"Executing web trade on {url} - Action: {action.upper()}")
        
        browser_exe = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        if not os.path.exists(browser_exe):
            browser_exe = r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
            if not os.path.exists(browser_exe):
                browser_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
                
        profile_path = os.path.abspath("./cdp_profile")
        
        # Close any lingering process that might block the debug port
        import psutil
        for p_proc in psutil.process_iter(['name', 'cmdline']):
            try:
                if p_proc.info['cmdline'] and profile_path in " ".join(p_proc.info['cmdline']):
                    p_proc.kill()
            except:
                pass
        
        time.sleep(1) # Give OS time to release port
        
        # Native Browser Launch
        proc = subprocess.Popen([
            browser_exe,
            "--remote-debugging-port=9225",
            f"--user-data-dir={profile_path}",
            url
        ])
        
        time.sleep(4) # Let real browser open and pass Cloudflare
        
        with sync_playwright() as p:
            try:
                browser = p.chromium.connect_over_cdp("http://localhost:9225")
                page = browser.contexts[0].pages[0]
                
                # Skip full load — domcontentloaded is enough for buttons
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=10000)
                except:
                    pass # Don't block if already loaded
                
                # Check for Demo account
                try:
                    if "demo" not in page.title().lower() and not page.locator("text=Demo account").is_visible():
                        print("WARNING: Could not verify Demo account. Please be careful.")
                except:
                    print("WARNING: Could not read page title (page still loading). Proceeding.")
                
                # ===== AGGRESSIVE RETRY LOGIC (3 Attempts) =====
                MAX_RETRIES = 3
                for attempt in range(1, MAX_RETRIES + 1):
                    print(f"[Attempt {attempt}/{MAX_RETRIES}] Searching for trade buttons...")
                    
                    # Try to find the button using raw JavaScript — bypasses ALL selector issues
                    found = page.evaluate("""() => {
                        // Strategy 1: Find by data-test attributes
                        let upBtn = document.querySelector('[data-test="deal-form_create-deal_up-button"]');
                        let downBtn = document.querySelector('[data-test="deal-form_create-deal_down-button"]');
                        if (upBtn && downBtn) return 'data-test';
                        
                        // Strategy 2: Find by visible text content "Up" / "Down" inside buttons
                        const allBtns = document.querySelectorAll('button');
                        for (const btn of allBtns) {
                            const txt = btn.textContent.trim().toLowerCase();
                            if (txt === 'up' || txt === 'down') return 'text-match';
                        }
                        
                        // Strategy 3: Find by button classes containing 'up' or 'down'
                        const upClass = document.querySelector('[class*="up-button"], [class*="call"], [class*="green"]');
                        const downClass = document.querySelector('[class*="down-button"], [class*="put"], [class*="red"]');
                        if (upClass || downClass) return 'class-match';
                        
                        return null;
                    }""")
                    
                    if found:
                        print(f"[Attempt {attempt}] Buttons found via strategy: {found}")
                        break
                    else:
                        print(f"[Attempt {attempt}] Buttons NOT found. ", end="")
                        if attempt < MAX_RETRIES:
                            print("Refreshing page and retrying in 3s...")
                            page.reload(wait_until="domcontentloaded", timeout=15000)
                            time.sleep(3)
                        else:
                            print("All retries exhausted.")
                            raise Exception(f"Could not find trade buttons after {MAX_RETRIES} attempts")
                
                # ===== EXECUTE THE CLICK =====
                sig_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                print(f"[Signal] {sig_time} -> {action.upper()}")
                
                if action.lower() == "buy":
                    print("[Action] Clicking UP/BUY button...")
                    clicked = page.evaluate("""() => {
                        // Priority 1: data-test
                        let btn = document.querySelector('[data-test="deal-form_create-deal_up-button"]');
                        if (btn) { btn.click(); return 'data-test-up'; }
                        // Priority 2: text match
                        for (const b of document.querySelectorAll('button')) {
                            if (b.textContent.trim().toLowerCase() === 'up') { b.click(); return 'text-up'; }
                        }
                        // Priority 3: class match
                        btn = document.querySelector('[class*="up-button"], [class*="call"]');
                        if (btn) { btn.click(); return 'class-up'; }
                        return null;
                    }""")
                else:
                    print("[Action] Clicking DOWN/SELL button...")
                    clicked = page.evaluate("""() => {
                        let btn = document.querySelector('[data-test="deal-form_create-deal_down-button"]');
                        if (btn) { btn.click(); return 'data-test-down'; }
                        for (const b of document.querySelectorAll('button')) {
                            if (b.textContent.trim().toLowerCase() === 'down') { b.click(); return 'text-down'; }
                        }
                        btn = document.querySelector('[class*="down-button"], [class*="put"]');
                        if (btn) { btn.click(); return 'class-down'; }
                        return null;
                    }""")
                
                exec_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                
                if clicked:
                    print(f"[JS Execution] Click confirmed via: {clicked} at {exec_time}")
                    print(f"Trade Success on Web UI: {action.upper()}")
                    return True, f"Success via {clicked}"
                else:
                    page.screenshot(path="error_screenshot.png")
                    print("--> ALERT: JS click returned null. Screenshot saved.")
                    return False, "JS click returned null - button not found in DOM"
                    
            except Exception as e:
                error_msg = str(e).split('\n')[0]
                print(f"Failed to execute web trade: {error_msg}")
                try:
                    page.screenshot(path="error_screenshot.png")
                    print("--> ALERT: Saved debug screenshot to error_screenshot.png")
                except:
                    pass
                return False, error_msg
            finally:
                time.sleep(2)
                proc.terminate()
