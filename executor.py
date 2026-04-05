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

    def execute_forex(self, action="buy", amount="10", multiplier="10",
                      tp_price=None, sl_price=None, 
                      model=None, scaler=None, processed_df=None,
                      url="https://olymptrade.com/platform", warm_session=None):
        """
        Executes a Forex (FX) trade on Olymp Trade.
        
        Forex UI differences from Fixed Time:
        - Buttons are "Buy" (green) / "Sell" (red) instead of Up/Down
        - Has a Multiplier field (10x, 100x)
        - Has Auto-close with Take Profit and Stop Loss
        - Trade stays open until TP/SL is hit or manually closed
        
        Args:
            action: 'buy' or 'sell'
            amount: Investment amount in dollars
            multiplier: Leverage multiplier ('10' or '100')
            tp_price: Take Profit price level (or None to skip)
            sl_price: Stop Loss price level (or None to skip)
            warm_session: Optional (proc, playwright, browser, page) from warm_up_browser()
            
        Returns:
            (success: bool, message: str, outcome: dict)
            outcome contains: {'hit': 'tp'|'sl'|'timeout'|'unknown', 'pnl_text': str}
        """
        import os
        import time
        import datetime
        import subprocess
        from config import Config
        
        print(f"\n\033[96m{'='*55}\033[0m")
        print(f"\033[96m       💹 FOREX MODE: {action.upper()} | {amount}$ | x{multiplier}\033[0m")
        print(f"\033[96m{'='*55}\033[0m")
        
        # ===== SESSION SETUP =====
        cold_start = warm_session is None
        
        if cold_start:
            print("[Cold Start] No warm session provided. Opening browser now...")
            proc, p, browser, page = self.warm_up_browser(url)
        else:
            proc, p, browser, page = warm_session
            print("[Warm Session] Using pre-warmed browser. Skipping Cloudflare wait.")
        
        outcome = {'hit': 'unknown', 'pnl_text': ''}
        
        try:
            # Safety: check for Demo
            try:
                if "demo" not in page.title().lower() and not page.locator("text=Demo account").is_visible():
                    print("WARNING: Could not verify Demo account. Please be careful.")
            except:
                print("WARNING: Could not read page title. Proceeding.")
            
            # ===== STEP 1: SET AMOUNT + MULTIPLIER =====
            print(f"[Forex Setup] Setting Amount: {amount}$ | Multiplier: x{multiplier}...")
            
            js_forex_setup = f"""
            (() => {{
                // === Helper: React-safe input setter ===
                function setReactInput(input, value) {{
                    try {{
                        let nativeSetter = Object.getOwnPropertyDescriptor(
                            window.HTMLInputElement.prototype, "value"
                        ).set;
                        nativeSetter.call(input, value);
                        input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    }} catch(e) {{
                        input.value = value;
                        input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    }}
                }}
                
                // === Helper: Find input by nearby text context ===
                function findInputByContext(keywords) {{
                    const inputs = document.querySelectorAll('input');
                    for (const inp of inputs) {{
                        let context = '';
                        // Check parent chain (up to 3 levels)
                        let el = inp;
                        for (let i = 0; i < 3 && el.parentElement; i++) {{
                            el = el.parentElement;
                            context += ' ' + (el.textContent || '').toLowerCase();
                        }}
                        context += ' ' + (inp.placeholder || '').toLowerCase();
                        context += ' ' + (inp.getAttribute('aria-label') || '').toLowerCase();
                        
                        if (keywords.some(kw => context.includes(kw))) {{
                            return inp;
                        }}
                    }}
                    return null;
                }}
                
                let results = {{ amount: false, multiplier: false }};
                
                // --- Set Amount ---
                const amountInput = findInputByContext(['amount', 'invest', 'sum', '$']);
                if (amountInput) {{
                    amountInput.focus();
                    setReactInput(amountInput, '{amount}');
                    amountInput.blur();
                    results.amount = true;
                }}
                
                // --- Set Multiplier ---
                // Strategy 1: Find multiplier dropdown/selector and click the target value
                const allElements = document.querySelectorAll('button, [role="button"], div, span, li, a');
                let multSet = false;
                
                // First, look for a multiplier container/dropdown
                for (const el of allElements) {{
                    const text = (el.textContent || '').trim().toLowerCase();
                    if (text === 'x{multiplier}' || text === '×{multiplier}' || text === '{multiplier}x') {{
                        el.click();
                        results.multiplier = true;
                        multSet = true;
                        break;
                    }}
                }}
                
                // Strategy 2: Find a multiplier input field
                if (!multSet) {{
                    const multInput = findInputByContext(['multiplier', 'leverage', 'mult', 'رافعة']);
                    if (multInput) {{
                        multInput.focus();
                        setReactInput(multInput, '{multiplier}');
                        multInput.blur();
                        results.multiplier = true;
                    }}
                }}
                
                return results;
            }})()
            """
            
            setup_result = page.evaluate(js_forex_setup)
            print(f"[Forex Setup] Result: {setup_result}")
            time.sleep(1)
            
            # ===== STEP 2: SET TP/SL + AUTO-CLOSE =====
            if tp_price is not None or sl_price is not None:
                print(f"[Forex TP/SL] Setting TP: {tp_price} | SL: {sl_price}...")
                
                tp_str = str(round(tp_price, 5)) if tp_price else ''
                sl_str = str(round(sl_price, 5)) if sl_price else ''
                
                js_tpsl_setup = f"""
                (() => {{
                    function setReactInput(input, value) {{
                        try {{
                            let nativeSetter = Object.getOwnPropertyDescriptor(
                                window.HTMLInputElement.prototype, "value"
                            ).set;
                            nativeSetter.call(input, value);
                            input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        }} catch(e) {{
                            input.value = value;
                            input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        }}
                    }}
                    
                    let results = {{ autoClose: false, tp: false, sl: false }};
                    
                    // --- Activate Auto-close toggle ---
                    const toggles = document.querySelectorAll(
                        '[class*="toggle"], [class*="switch"], [role="switch"], input[type="checkbox"]'
                    );
                    for (const toggle of toggles) {{
                        let context = '';
                        let el = toggle;
                        for (let i = 0; i < 3 && el.parentElement; i++) {{
                            el = el.parentElement;
                            context += ' ' + (el.textContent || '').toLowerCase();
                        }}
                        if (context.includes('auto') && (context.includes('close') || context.includes('إغلاق'))) {{
                            // Check if already active
                            const isActive = toggle.classList.contains('active') || 
                                           toggle.getAttribute('aria-checked') === 'true' ||
                                           toggle.checked === true;
                            if (!isActive) {{
                                toggle.click();
                                results.autoClose = 'activated';
                            }} else {{
                                results.autoClose = 'already_active';
                            }}
                            break;
                        }}
                    }}
                    
                    // Also try clicking text that says "Auto close"
                    if (!results.autoClose) {{
                        const allEls = document.querySelectorAll('div, span, label, button');
                        for (const el of allEls) {{
                            const txt = (el.textContent || '').trim().toLowerCase();
                            if ((txt.includes('auto') && txt.includes('close')) || 
                                (txt.includes('auto-close')) ||
                                (txt.includes('إغلاق تلقائي'))) {{
                                el.click();
                                results.autoClose = 'text_click';
                                break;
                            }}
                        }}
                    }}
                    
                    // Wait a moment for TP/SL fields to appear after auto-close activation
                    await new Promise(r => setTimeout(r, 500));
                    
                    // --- Set Take Profit ---
                    if ('{tp_str}') {{
                        const inputs = document.querySelectorAll('input');
                        for (const inp of inputs) {{
                            let context = '';
                            let el = inp;
                            for (let i = 0; i < 4 && el.parentElement; i++) {{
                                el = el.parentElement;
                                context += ' ' + (el.textContent || '').toLowerCase();
                            }}
                            if ((context.includes('take') && context.includes('profit')) || 
                                context.includes('tp') ||
                                context.includes('أخذ الربح')) {{
                                inp.focus();
                                setReactInput(inp, '{tp_str}');
                                inp.blur();
                                results.tp = true;
                                break;
                            }}
                        }}
                    }}
                    
                    // --- Set Stop Loss ---
                    if ('{sl_str}') {{
                        const inputs = document.querySelectorAll('input');
                        for (const inp of inputs) {{
                            let context = '';
                            let el = inp;
                            for (let i = 0; i < 4 && el.parentElement; i++) {{
                                el = el.parentElement;
                                context += ' ' + (el.textContent || '').toLowerCase();
                            }}
                            if ((context.includes('stop') && context.includes('loss')) ||
                                context.includes('sl') ||
                                context.includes('وقف الخسارة')) {{
                                inp.focus();
                                setReactInput(inp, '{sl_str}');
                                inp.blur();
                                results.sl = true;
                                break;
                            }}
                        }}
                    }}
                    
                    return results;
                }})()
                """
                
                # Use async evaluate for the await inside
                try:
                    tpsl_result = page.evaluate(js_tpsl_setup)
                except:
                    # Fallback without async
                    js_tpsl_sync = js_tpsl_setup.replace("await new Promise(r => setTimeout(r, 500));", "")
                    tpsl_result = page.evaluate(js_tpsl_sync)
                
                print(f"[Forex TP/SL] Result: {tpsl_result}")
                time.sleep(1)
            
            # ===== STEP 3: CLICK BUY/SELL BUTTON =====
            sig_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            print(f"\n[Signal] {sig_time} -> {action.upper()}")
            
            if action.lower() == "buy":
                print("[Action] Clicking BUY (Green) button...")
                clicked = page.evaluate("""() => {
                    // P1: data-test attributes for buy
                    let btn = document.querySelector(
                        '[data-test*="buy-button"], [data-test*="buy_button"], ' +
                        '[data-test*="up-button"], [data-test*="up_button"]'
                    );
                    if (btn) { btn.click(); return 'data-test-buy'; }
                    
                    // P2: Exact text match — Buy
                    for (const b of document.querySelectorAll('button, [role="button"]')) {
                        const txt = b.textContent.trim().toLowerCase();
                        if (txt === 'buy') { b.click(); return 'text-buy'; }
                    }
                    
                    // P3: Green-colored large button (Buy is typically green)
                    for (const b of document.querySelectorAll('button, [role="button"]')) {
                        const style = window.getComputedStyle(b);
                        const bg = style.backgroundColor;
                        const rect = b.getBoundingClientRect();
                        // Green button with reasonable size (not small icon buttons)
                        if (rect.width > 60 && rect.height > 30) {
                            if (bg.includes('0, 200') || bg.includes('0, 180') || 
                                bg.includes('76, 175') || bg.includes('56, 142') ||
                                bg.includes('46, 125') || bg.includes('67, 160')) {
                                b.click(); return 'color-green-buy';
                            }
                        }
                    }
                    
                    // P4: Partial text + position (Buy button is usually on the left/top)
                    for (const b of document.querySelectorAll('button, [role="button"]')) {
                        const txt = b.textContent.trim().toLowerCase();
                        if (txt.includes('buy') && !txt.includes('buying')) {
                            b.click(); return 'partial-buy';
                        }
                    }
                    
                    // P5: Fallback — Up/Call for backward compat
                    for (const b of document.querySelectorAll('button, [role="button"]')) {
                        const txt = b.textContent.trim().toLowerCase();
                        if (txt === 'up' || txt === 'call') { b.click(); return 'fallback-' + txt; }
                    }
                    
                    return null;
                }""")
            else:
                print("[Action] Clicking SELL (Red) button...")
                clicked = page.evaluate("""() => {
                    // P1: data-test
                    let btn = document.querySelector(
                        '[data-test*="sell-button"], [data-test*="sell_button"], ' +
                        '[data-test*="down-button"], [data-test*="down_button"]'
                    );
                    if (btn) { btn.click(); return 'data-test-sell'; }
                    
                    // P2: Exact text — Sell
                    for (const b of document.querySelectorAll('button, [role="button"]')) {
                        const txt = b.textContent.trim().toLowerCase();
                        if (txt === 'sell') { b.click(); return 'text-sell'; }
                    }
                    
                    // P3: Red-colored large button
                    for (const b of document.querySelectorAll('button, [role="button"]')) {
                        const style = window.getComputedStyle(b);
                        const bg = style.backgroundColor;
                        const rect = b.getBoundingClientRect();
                        if (rect.width > 60 && rect.height > 30) {
                            if (bg.includes('244, 67') || bg.includes('229, 57') ||
                                bg.includes('211, 47') || bg.includes('239, 83') ||
                                bg.includes('255, 82') || bg.includes('234, 57')) {
                                b.click(); return 'color-red-sell';
                            }
                        }
                    }
                    
                    // P4: Partial text
                    for (const b of document.querySelectorAll('button, [role="button"]')) {
                        const txt = b.textContent.trim().toLowerCase();
                        if (txt.includes('sell') && !txt.includes('selling')) {
                            b.click(); return 'partial-sell';
                        }
                    }
                    
                    // P5: Fallback
                    for (const b of document.querySelectorAll('button, [role="button"]')) {
                        const txt = b.textContent.trim().toLowerCase();
                        if (txt === 'down' || txt === 'put') { b.click(); return 'fallback-' + txt; }
                    }
                    
                    return null;
                }""")
            
            exec_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            if clicked:
                print(f"\033[92m[JS Execution] Click confirmed via: {clicked} at {exec_time}\033[0m")
                print(f"\033[92m[Forex Trade Opened] {action.upper()} | x{multiplier} | {amount}$\033[0m")
                
                # Screenshot
                time.sleep(1.5)
                try:
                    page.screenshot(path="trade_executed.png")
                    print("[Screenshot] Saved trade confirmation to 'trade_executed.png'")
                except:
                    print("[Screenshot] Could not capture trade confirmation.")
                
                # Beep alert
                try:
                    import winsound
                    for _ in range(3):
                        winsound.Beep(1000, 200)
                        time.sleep(0.1)
                except:
                    pass
                
                # ===== STEP 4: SHARK EXIT — INTELLIGENT TRADE MONITORING =====
                print(f"\n\033[96m{'='*55}\033[0m")
                print(f"\033[96m   🦈 SHARK MODE: Monitoring Trade\033[0m")
                print(f"\033[96m   Reversal Check every {Config.SHARK_POLL_INTERVAL}s...\033[0m")
                print(f"\033[96m   Max Hold: {Config.FOREX_MAX_HOLD_SECONDS}s | BB Threshold: {Config.SHARK_REVERSAL_BB_THRESHOLD}\033[0m")
                print(f"\033[96m{'='*55}\033[0m")
                
                outcome = self.monitor_and_close(
                    page=page,
                    action=action,
                    model=model,
                    scaler=scaler,
                    processed_df=processed_df
                )
                
                # Final screenshot after close
                try:
                    time.sleep(1)
                    page.screenshot(path="trade_closed.png")
                except:
                    pass
                
                return True, f"Forex {action.upper()} via {clicked}", outcome
            else:
                page.screenshot(path="error_screenshot.png")
                print("--> ALERT: JS click returned null. Screenshot saved.")
                try:
                    import winsound
                    winsound.Beep(400, 800)
                except:
                    pass
                return False, "JS click returned null - Buy/Sell button not found in DOM", outcome
                
        except Exception as e:
            error_msg = str(e).split('\n')[0]
            print(f"Failed to execute Forex trade: {error_msg}")
            try:
                page.screenshot(path="error_screenshot.png")
                print("--> ALERT: Saved debug screenshot to error_screenshot.png")
            except:
                pass
            return False, error_msg, outcome
        finally:
            time.sleep(2)
            try:
                p.stop()
            except:
                pass
            proc.terminate()

    def monitor_and_close(self, page, action, model, scaler, processed_df):
        """
        🦈 Shark Exit — Intelligent Trade Monitoring & Dynamic Close.
        
        Instead of waiting passively for TP/SL, this method actively monitors
        the trade and closes it when reversal conditions are detected.
        
        Exit Conditions:
        A) Platform TP/SL hit (detected via DOM)
        B) LSTM opposite signal with high confidence
        C) Price at Bollinger Band reversal zone while in profit
        
        Args:
            page: Playwright page object for DOM interaction
            action: 'buy' or 'sell' — the direction of the open trade
            model: Trained LSTM model (same one that generated the entry signal)
            scaler: Fitted RobustScaler for feature normalization
            processed_df: Latest processed DataFrame with features
            
        Returns:
            dict: {hit: 'tp'|'sl'|'shark_exit'|'timeout', reason: str, pnl_text: str}
        """
        import time
        import numpy as np
        from config import Config
        from features import feature_engineering_pipeline
        
        outcome = {'hit': 'unknown', 'reason': '', 'pnl_text': ''}
        trade_start = time.time()
        last_print = 0
        shark_check_count = 0
        
        while (time.time() - trade_start) < Config.FOREX_MAX_HOLD_SECONDS:
            time.sleep(Config.SHARK_POLL_INTERVAL)
            elapsed = int(time.time() - trade_start)
            shark_check_count += 1
            
            # ===== CHECK 1: Platform TP/SL (DOM-based) =====
            try:
                trade_status = page.evaluate("""() => {
                    const pnlSelectors = [
                        '[class*="profit"]', '[class*="pnl"]', '[class*="result"]',
                        '[class*="deal-result"]', '[class*="trade-result"]',
                        '[data-test*="profit"]', '[data-test*="result"]'
                    ];
                    
                    for (const sel of pnlSelectors) {
                        const el = document.querySelector(sel);
                        if (el) {
                            const text = el.textContent.trim();
                            return { found: true, text: text, selector: sel };
                        }
                    }
                    
                    const allEls = document.querySelectorAll('div, span');
                    let pnlText = '';
                    let activeTradeFound = false;
                    
                    for (const el of allEls) {
                        const text = (el.textContent || '').trim();
                        if (/^[+\\-]\\$?\\d+\\.\\d{2}$/.test(text) || /^[+\\-]\\d+\\.\\d{2}\\$$/.test(text)) {
                            pnlText = text;
                            activeTradeFound = true;
                        }
                        if (text.toLowerCase().includes('closed') || text.toLowerCase().includes('تم الإغلاق')) {
                            return { found: true, closed: true, text: text, pnl: pnlText };
                        }
                    }
                    
                    if (activeTradeFound) {
                        return { found: true, closed: false, pnl: pnlText };
                    }
                    
                    return { found: false };
                }""")
            except Exception as e:
                print(f"\033[93m  [{elapsed}s] DOM poll error: {str(e)[:50]}\033[0m")
                continue
            
            pnl_text = ''
            if trade_status and trade_status.get('found'):
                pnl_text = trade_status.get('pnl', trade_status.get('text', ''))
                
                # Check if platform already closed the trade (TP/SL hit)
                if trade_status.get('closed'):
                    outcome['pnl_text'] = pnl_text
                    if pnl_text.startswith('+') or (pnl_text and '-' not in pnl_text and '$' in pnl_text):
                        outcome['hit'] = 'tp'
                        outcome['reason'] = 'Platform Take Profit hit'
                        print(f"\033[92m[Shark] ✅ TAKE PROFIT HIT by platform! PnL: {pnl_text}\033[0m")
                    else:
                        outcome['hit'] = 'sl'
                        outcome['reason'] = 'Platform Stop Loss hit'
                        print(f"\033[91m[Shark] ❌ STOP LOSS HIT by platform! PnL: {pnl_text}\033[0m")
                    return outcome
            
            # Print status every 30 seconds
            if elapsed - last_print >= 30:
                print(f"  [{elapsed}s] 🦈 Shark Check #{shark_check_count} | PnL: {pnl_text}")
                last_print = elapsed
            
            # ===== CHECK 2: LSTM Reversal Signal =====
            try:
                # Get fresh data via the unified data loader (yfinance for Forex, scraper for OTC)
                from data_loader import fetch_data
                fresh_df = fetch_data()
                
                if fresh_df is not None and not fresh_df.empty:
                    fresh_processed = feature_engineering_pipeline(fresh_df)
                    
                    if len(fresh_processed) >= Config.SEQUENCE_LENGTH:
                        latest_features = fresh_processed.drop(['Target'], axis=1).values
                        latest_scaled = scaler.transform(latest_features)
                        X_live = np.array([latest_scaled[-Config.SEQUENCE_LENGTH:]])
                        
                        new_prob = model.predict(X_live, verbose=0)[0][0]
                        new_signal = "buy" if new_prob > 0.5 else "sell"
                        confidence = new_prob if new_signal == "buy" else (1.0 - new_prob)
                        
                        # Check for opposite signal with sufficient confidence
                        is_opposite = (action == "buy" and new_signal == "sell") or \
                                      (action == "sell" and new_signal == "buy")
                        
                        if is_opposite and confidence >= Config.SHARK_OPPOSITE_SIGNAL_CONFIDENCE:
                            print(f"\n\033[93m[Shark] 🔄 OPPOSITE SIGNAL DETECTED!\033[0m")
                            print(f"\033[93m  Entry: {action.upper()} | Now: {new_signal.upper()} ({confidence*100:.1f}% confidence)\033[0m")
                            
                            # Check if in profit before closing
                            in_profit = pnl_text.startswith('+') if pnl_text else False
                            
                            if in_profit or pnl_text == '':
                                # Close the trade! 
                                close_success = self._click_close_button(page)
                                if close_success:
                                    outcome['hit'] = 'shark_exit'
                                    outcome['reason'] = f'Opposite signal ({new_signal.upper()} @ {confidence*100:.1f}%)'
                                    outcome['pnl_text'] = pnl_text
                                    print(f"\033[92m[Shark] 🦈 TRADE CLOSED! Reason: {outcome['reason']} | PnL: {pnl_text}\033[0m")
                                    
                                    try:
                                        import winsound
                                        winsound.Beep(800, 300)
                                        time.sleep(0.1)
                                        winsound.Beep(800, 300)
                                    except:
                                        pass
                                    
                                    return outcome
                                else:
                                    print(f"\033[91m[Shark] Close button not found! Trade still open.\033[0m")
                            else:
                                print(f"\033[93m  PnL is negative ({pnl_text}). Holding — letting TP/SL decide.\033[0m")
                        
                        # ===== CHECK 3: Bollinger Band Reversal =====
                        if 'BB_position' in fresh_processed.columns:
                            bb_pos = fresh_processed['BB_position'].iloc[-1]
                            
                            reversal_risk = False
                            if action == "buy" and bb_pos >= Config.SHARK_REVERSAL_BB_THRESHOLD:
                                reversal_risk = True
                                print(f"\033[93m  [{elapsed}s] BB Position: {bb_pos:.2f} — Price near top band (BUY reversal zone)\033[0m")
                            elif action == "sell" and bb_pos <= (1.0 - Config.SHARK_REVERSAL_BB_THRESHOLD):
                                reversal_risk = True
                                print(f"\033[93m  [{elapsed}s] BB Position: {bb_pos:.2f} — Price near bottom band (SELL reversal zone)\033[0m")
                            
                            if reversal_risk:
                                in_profit = pnl_text.startswith('+') if pnl_text else False
                                if in_profit:
                                    print(f"\033[93m[Shark] 📈 PROFIT PROTECTION triggered! BB reversal zone + positive PnL\033[0m")
                                    close_success = self._click_close_button(page)
                                    if close_success:
                                        outcome['hit'] = 'shark_exit'
                                        outcome['reason'] = f'BB reversal protection (BB_pos={bb_pos:.2f})'
                                        outcome['pnl_text'] = pnl_text
                                        print(f"\033[92m[Shark] 🦈 TRADE CLOSED! Profit secured: {pnl_text}\033[0m")
                                        return outcome
                
            except Exception as e:
                # Shark analysis failed — not critical, just skip this cycle
                if shark_check_count % 6 == 0:  # Print error every ~60s
                    print(f"\033[93m  [{elapsed}s] Shark analysis error: {str(e)[:60]}\033[0m")
            
            # Timeout progress
            if elapsed >= Config.FOREX_MAX_HOLD_SECONDS * 0.9:
                remaining = Config.FOREX_MAX_HOLD_SECONDS - elapsed
                print(f"\033[93m  [{elapsed}s] ⏰ Warning: {remaining}s until auto-timeout\033[0m")
        
        # Max hold time reached
        outcome['hit'] = 'timeout'
        outcome['reason'] = f'Max hold time ({Config.FOREX_MAX_HOLD_SECONDS}s) reached'
        outcome['pnl_text'] = pnl_text
        print(f"\033[93m[Shark] ⏰ TIMEOUT. Trade may still be open. PnL at timeout: {pnl_text}\033[0m")
        
        try:
            page.screenshot(path="trade_timeout.png")
        except:
            pass
        
        return outcome

    def _click_close_button(self, page):
        """
        Clicks the Close/Sell-out button on the Olymp Trade platform DOM.
        Returns True if a button was found and clicked, False otherwise.
        """
        import time
        
        clicked = page.evaluate("""() => {
            // Strategy 1: Look for explicit Close/Sell-out buttons
            const closeKeywords = ['close', 'sell out', 'close trade', 'إغلاق', 'بيع'];
            const allBtns = document.querySelectorAll('button, [role="button"], a, div[class*="button"]');
            
            for (const btn of allBtns) {
                const txt = (btn.textContent || '').trim().toLowerCase();
                for (const kw of closeKeywords) {
                    if (txt === kw || txt.includes(kw)) {
                        const rect = btn.getBoundingClientRect();
                        if (rect.width > 20 && rect.height > 15) {
                            btn.click();
                            return 'text-' + kw;
                        }
                    }
                }
            }
            
            // Strategy 2: data-test attributes
            const closeSelectors = [
                '[data-test*="close"]', '[data-test*="sell-out"]',
                '[data-test*="close-trade"]', '[data-test*="close_trade"]'
            ];
            for (const sel of closeSelectors) {
                const el = document.querySelector(sel);
                if (el) { el.click(); return 'data-test-close'; }
            }
            
            // Strategy 3: X/close icons near active trade
            const closeIcons = document.querySelectorAll(
                '[class*="close-icon"], [class*="close_icon"], ' +
                'svg[class*="close"], [class*="deal"] [class*="close"]'
            );
            for (const icon of closeIcons) {
                icon.click();
                return 'icon-close';
            }
            
            return null;
        }""")
        
        if clicked:
            print(f"\033[92m[Close] Button clicked via: {clicked}\033[0m")
            time.sleep(1)
            
            # Sometimes there's a confirmation dialog
            try:
                page.evaluate("""() => {
                    const confirmBtns = document.querySelectorAll('button, [role="button"]');
                    for (const btn of confirmBtns) {
                        const txt = (btn.textContent || '').trim().toLowerCase();
                        if (txt === 'yes' || txt === 'confirm' || txt === 'ok' || txt === 'نعم') {
                            btn.click();
                            return true;
                        }
                    }
                    return false;
                }""")
            except:
                pass
            
            return True
        
        return False
