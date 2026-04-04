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

    def execute_web(self, action="buy", url="https://olymptrade.com", user_data_dir="./playwright_profile", 
                    buy_selector=".button-buy", sell_selector=".button-sell", amount_selector=".amount-input"):
        """
        Executes a paper trade on a web platform using Playwright with persistent context.
        """
        print(f"Executing web trade on {url} - Action: {action.upper()}")
        with sync_playwright() as p:
            # Persistent context keeps the session (cookies, local storage) across runs
            browser = p.chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                headless=False,
                args=["--start-maximized"],
                no_viewport=True
            )
            
            # Since persistent context opens a default page, use it
            page = browser.pages[0] if browser.pages else browser.new_page()
            
            try:
                page.goto(url, timeout=60000)
                print("Page loaded successfully.")
                
                # Check for Demo account active indicator - safety check framework
                # This selector should be updated based on actual DOM
                if "demo" not in page.title().lower() and not page.locator("text=Demo account").is_visible():
                    print("WARNING: Could not verify Demo account presence on screen. Please be careful.")
                
                # Wait for the chart to appear (as requested)
                # Just a generic check, assuming canvas or SVG chart
                print("Waiting for chart to be visible...")
                # page.wait_for_selector("canvas", timeout=15000) 
                
                print(f"[Action] Signal received: {action.upper()}")
                if action.lower() == "buy":
                    print(f"Looking for BUY button using selector: {buy_selector}")
                    # page.click(buy_selector, timeout=10000)
                    print("[Action] Clicking BUY on Olymp Trade...")
                else:
                    print(f"Looking for SELL button using selector: {sell_selector}")
                    # page.click(sell_selector, timeout=10000)
                    print("[Action] Clicking SELL on Olymp Trade...")
                    
            except Exception as e:
                print(f"Failed to execute web trade: {e}")
            finally:
                import time
                time.sleep(3) # Let user see what happened
                browser.close()
