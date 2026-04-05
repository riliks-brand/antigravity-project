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

    def warm_up_browser(self, url="https://olymptrade.com/platform"):
        """
        Opens the browser early (Warm-up Phase) so Cloudflare passes before the candle closes.
        Returns (proc, playwright_instance, browser, page) to be reused by execute_web.
        """
        import os
        import time
        import subprocess
        import psutil
        
        browser_exe = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        if not os.path.exists(browser_exe):
            browser_exe = r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
            if not os.path.exists(browser_exe):
                browser_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
                
        profile_path = os.path.abspath("./cdp_profile")
        
        # Kill any lingering browser using our profile
        for p_proc in psutil.process_iter(['name', 'cmdline']):
            try:
                if p_proc.info['cmdline'] and profile_path in " ".join(p_proc.info['cmdline']):
                    p_proc.kill()
            except:
                pass
        
        time.sleep(1)
        
        # Native Browser Launch (Forcing WebGL/GPU for Olymp Trade Charts)
        proc = subprocess.Popen([
            browser_exe,
            "--remote-debugging-port=9225",
            f"--user-data-dir={profile_path}",
            "--ignore-gpu-blocklist",
            "--enable-webgl",
            "--no-sandbox",
            url
        ])
        
        time.sleep(10)  # Extended wait to ensure WebGL/Canvas loads fully
        
        p = sync_playwright().start()
        browser = p.chromium.connect_over_cdp("http://localhost:9225")
        page = browser.contexts[0].pages[0]
        
        # Wait for DOM content only (speed)
        try:
            page.wait_for_load_state("domcontentloaded", timeout=15000)
        except:
            pass
        
        print("[Warm-up] Browser is warm. Cloudflare negotiation complete. Standing by.")
        return proc, p, browser, page

    def execute_web(self, action="buy", duration="2 min", amount="10", url="https://olymptrade.com/platform", 
                    warm_session=None):
        """
        Executes a Fixed Time trade on Olymp Trade.
        
        Args:
            action: 'buy' or 'sell'
            url: Platform URL
            warm_session: Optional tuple (proc, playwright, browser, page) from warm_up_browser()
                         If None, will cold-start the browser.
        """
        import os
        import time
        import datetime
        import subprocess
        
        print(f"Executing web trade on {url} - Action: {action.upper()}")
        
        # ===== SESSION SETUP =====
        cold_start = warm_session is None
        
        if cold_start:
            print("[Cold Start] No warm session provided. Opening browser now...")
            proc, p, browser, page = self.warm_up_browser(url)
        else:
            proc, p, browser, page = warm_session
            print("[Warm Session] Using pre-warmed browser. Skipping Cloudflare wait.")
        
        try:
            # Check for Demo account
            try:
                if "demo" not in page.title().lower() and not page.locator("text=Demo account").is_visible():
                    print("WARNING: Could not verify Demo account. Please be careful.")
            except:
                print("WARNING: Could not read page title. Proceeding.")
            
            # ===== STEP 1: SET DURATION AND AMOUNT USING JS HEURISTICS =====
            print(f"[Duration/Amount] Setting duration to {duration} and amount to {amount}...")
            
            # Inject duration and amount logic utilizing heuristic JS
            js_injection = f"""
            (() => {{
                function findReactInputAndSet(keywords, valueToSet) {{
                    const inputs = document.querySelectorAll('input');
                    for (const inp of inputs) {{
                        let parentText = (inp.parentElement && inp.parentElement.textContent || "").toLowerCase();
                        let placeholder = (inp.placeholder || "").toLowerCase();
                        // Search nearby text like previous or next sibling
                        let prevText = (inp.previousElementSibling && inp.previousElementSibling.textContent || "").toLowerCase();
                        let nextText = (inp.nextElementSibling && inp.nextElementSibling.textContent || "").toLowerCase();
                        
                        if (keywords.some(kw => parentText.includes(kw) || placeholder.includes(kw) || prevText.includes(kw) || nextText.includes(kw))) {{
                            inp.focus();
                            // clear existing
                            inp.value = '';
                            inp.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            // set new
                            inp.value = valueToSet;
                            inp.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            inp.dispatchEvent(new Event('change', {{ bubbles: true }}));
                            try {{
                                // React 16+ value setter override bypass
                                let nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
                                nativeInputValueSetter.call(inp, valueToSet);
                                inp.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            }} catch(e) {{}}
                            inp.blur();
                            return true;
                        }}
                    }}
                    return false;
                }}
                
                let amountSet = findReactInputAndSet(['amount', 'invest', '$', 'summ', 'مبلغ', 'استثمار'], '{amount}');
                let durationDigit = '{duration}'.replace(/[^0-9]/g, '');
                let durationSet = findReactInputAndSet(['duration', 'time', 'min', 'مدت', 'المدة'], durationDigit);
                
                return {{amountSet: amountSet, durationSet: durationSet}};
            }})()
            """
            
            injection_result = page.evaluate(js_injection)
            print(f"[UI Heuristics] Injection result: {injection_result}")
            
            # Wait for visually confirming the change in the UI element before proceeding
            time.sleep(1.5)
            
            # ===== STEP 2: AGGRESSIVE RETRY LOGIC (3 Attempts) =====
            MAX_RETRIES = 3
            for attempt in range(1, MAX_RETRIES + 1):
                print(f"[Attempt {attempt}/{MAX_RETRIES}] Searching for trade buttons...")
                
                # Comprehensive JS search — Generic Selectors / Text-based / XPath
                found = page.evaluate("""() => {
                    // === Strategy 1: data-test attributes ===
                    let upBtn = document.querySelector('[data-test*="up-button"], [data-test*="up_button"]');
                    let downBtn = document.querySelector('[data-test*="down-button"], [data-test*="down_button"]');
                    if (upBtn && downBtn) return 'data-test';
                    
                    // === Strategy 2: Text content matching (Buy/Sell/Up/Down/Call/Put) ===
                    const keywords = ['up', 'down', 'buy', 'sell', 'call', 'put'];
                    const allBtns = document.querySelectorAll('button, [role="button"]');
                    let textMatches = 0;
                    for (const btn of allBtns) {
                        const txt = btn.textContent.trim().toLowerCase();
                        if (keywords.some(kw => txt === kw || txt.startsWith(kw + ' ') || txt.endsWith(' ' + kw))) {
                            textMatches++;
                        }
                    }
                    if (textMatches >= 2) return 'text-match';
                    
                    // === Strategy 3: XPath - find buttons by their aria or visual text ===
                    const xpathUp = document.evaluate(
                        "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'up') or contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'buy')]",
                        document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null
                    ).singleNodeValue;
                    const xpathDown = document.evaluate(
                        "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'down') or contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'sell')]",
                        document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null
                    ).singleNodeValue;
                    if (xpathUp || xpathDown) return 'xpath-match';
                    
                    // === Strategy 4: Color-based (green/red buttons) ===
                    for (const btn of allBtns) {
                        const style = window.getComputedStyle(btn);
                        const bg = style.backgroundColor;
                        if (bg.includes('0, 200') || bg.includes('0, 180') || bg.includes('76, 175')) return 'color-green';
                    }
                    
                    return null;
                }""")
                
                if found:
                    print(f"[Attempt {attempt}] Buttons found via strategy: {found}")
                    break
                else:
                    print(f"[Attempt {attempt}] Buttons NOT found. ", end="")
                    if attempt < MAX_RETRIES:
                        print("Refreshing page and retrying in 4s...")
                        page.reload(wait_until="domcontentloaded", timeout=15000)
                        time.sleep(4)
                    else:
                        print("All retries exhausted.")
                        raise Exception(f"Could not find trade buttons after {MAX_RETRIES} attempts")
            
            # ===== STEP 3: EXECUTE THE CLICK =====
            sig_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            print(f"[Signal] {sig_time} -> {action.upper()}")
            
            if action.lower() == "buy":
                print("[Action] Clicking UP/BUY button...")
                clicked = page.evaluate("""() => {
                    // P1: data-test
                    let btn = document.querySelector('[data-test*="up-button"], [data-test*="up_button"]');
                    if (btn) { btn.click(); return 'data-test-up'; }
                    // P2: exact text match (Up, Buy, Call)
                    for (const b of document.querySelectorAll('button, [role="button"]')) {
                        const txt = b.textContent.trim().toLowerCase();
                        if (txt === 'up' || txt === 'buy' || txt === 'call') { b.click(); return 'text-' + txt; }
                    }
                    // P3: partial text match
                    for (const b of document.querySelectorAll('button, [role="button"]')) {
                        const txt = b.textContent.trim().toLowerCase();
                        if (txt.includes('up') && !txt.includes('update') && !txt.includes('support')) { b.click(); return 'partial-up'; }
                    }
                    // P4: XPath
                    const x = document.evaluate("//button[contains(translate(., 'UP', 'up'), 'up')]", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                    if (x) { x.click(); return 'xpath-up'; }
                    return null;
                }""")
            else:
                print("[Action] Clicking DOWN/SELL button...")
                clicked = page.evaluate("""() => {
                    let btn = document.querySelector('[data-test*="down-button"], [data-test*="down_button"]');
                    if (btn) { btn.click(); return 'data-test-down'; }
                    for (const b of document.querySelectorAll('button, [role="button"]')) {
                        const txt = b.textContent.trim().toLowerCase();
                        if (txt === 'down' || txt === 'sell' || txt === 'put') { b.click(); return 'text-' + txt; }
                    }
                    for (const b of document.querySelectorAll('button, [role="button"]')) {
                        const txt = b.textContent.trim().toLowerCase();
                        if (txt.includes('down') && !txt.includes('download')) { b.click(); return 'partial-down'; }
                    }
                    const x = document.evaluate("//button[contains(translate(., 'DOWN', 'down'), 'down')]", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                    if (x) { x.click(); return 'xpath-down'; }
                    return null;
                }""")
            
            exec_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            if clicked:
                print(f"[JS Execution] Click confirmed via: {clicked} at {exec_time}")
                print(f"Trade Success on Web UI: {action.upper()}")
                
                # === SUCCESS SCREENSHOT: Capture the open trade ===
                time.sleep(1.5)  # Let the platform register the trade visually
                try:
                    page.screenshot(path="trade_executed.png")
                    print("[Screenshot] Saved trade confirmation to 'trade_executed.png'")
                except:
                    print("[Screenshot] Could not capture trade confirmation.")
                
                # === BEEP ALERT: 3 beeps = trade executed ===
                import winsound
                for _ in range(3):
                    winsound.Beep(1000, 200)  # 1000 Hz for 200ms
                    time.sleep(0.1)
                
                return True, f"Success via {clicked}"
            else:
                page.screenshot(path="error_screenshot.png")
                print("--> ALERT: JS click returned null. Screenshot saved.")
                # === WARNING BEEP: 1 long beep = error ===
                try:
                    import winsound
                    winsound.Beep(400, 800)  # Low tone, long = error
                except:
                    pass
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
            try:
                p.stop()
            except:
                pass
            proc.terminate()
