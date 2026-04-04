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

    def execute_web(self, action="buy", url="https://olymptrade.com", user_data_dir=None):
        """
        Executes a paper trade on a web platform using Playwright connected natively over CDP to defeat Cloudflare.
        """
        import os
        import time
        import subprocess
        
        print(f"Executing web trade on {url} - Action: {action.upper()}")
        
        # Robust data-test selectors extracted from Olymp Trade production DOM
        tab_buy = '[data-test="deal-form_direction-buy"]'
        tab_sell = '[data-test="deal-form_direction-sell"]'
        execute_button = '[data-test="cfd-desktop_deal-form_trade-button-wrapper"] button'
        
        browser_exe = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        if not os.path.exists(browser_exe):
            browser_exe = r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
            if not os.path.exists(browser_exe):
                browser_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
                
        profile_path = os.path.abspath("./cdp_profile")
        
        # Native Browser Launch
        proc = subprocess.Popen([
            browser_exe,
            "--remote-debugging-port=9223", # Diff port than test_web to avoid conflict if both run
            f"--user-data-dir={profile_path}",
            url
        ])
        
        time.sleep(4) # Let real browser load and negotiate CF
        
        with sync_playwright() as p:
            try:
                browser = p.chromium.connect_over_cdp("http://localhost:9223")
                page = browser.contexts[0].pages[0]
                
                # Check for Demo account active indicator - safety check framework
                if "demo" not in page.title().lower() and not page.locator("text=Demo account").is_visible():
                    print("WARNING: Could not verify Demo account presence on screen. Please be careful.")
                
                # Wait for the trade form wrapper to exist
                print("Waiting for trading interface to load...")
                page.wait_for_selector(execute_button, timeout=15000)
                
                print(f"[Action] Signal received: {action.upper()}")
                if action.lower() == "buy":
                    print("[Action] Setting direction to BUY (Up)...")
                    page.locator(tab_buy).click()
                    # Small delay to allow UI state to update
                    page.wait_for_timeout(500)
                    print("[Action] Clicking Execute button on Olymp Trade...")
                    page.locator(execute_button).click()
                else:
                    print("[Action] Setting direction to SELL (Down)...")
                    page.locator(tab_sell).click()
                    page.wait_for_timeout(500)
                    print("[Action] Clicking Execute button on Olymp Trade...")
                    page.locator(execute_button).click()
                    
                print(f"Trade Success on Web UI: {action.upper()}")
                return True
            except Exception as e:
                print(f"Failed to execute web trade: {e}")
                return False
            finally:
                time.sleep(3) # Let user see what happened
                proc.terminate() # Close Native Browser
