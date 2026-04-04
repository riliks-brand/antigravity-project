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
        if action.lower() == "buy":
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
            sl = price - 100 * point
            tp = price + 100 * point
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
            sl = price + 100 * point
            tp = price - 100 * point

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
        return result.order

    def execute_web(self, action="buy", url="https://olymptrade.com", call_selector="#call-btn", put_selector="#put-btn"):
        """
        Executes a paper trade on a web platform using Playwright.
        Selectors are completely configurable.
        """
        print(f"Executing web trade on {url} - Action: {action}")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False) # Headless=False so the user can see it
            page = browser.new_page()
            
            # Navigate to the platform
            try:
                page.goto(url, timeout=30000)
                print("Page loaded successfully.")
                
                # In a real scenario, we might need to handle login or wait for DOM 
                # This is a generic clicker based on selectors
                if action.lower() == "buy":
                    print(f"Looking for CALL button using selector: {call_selector}")
                    # page.click(call_selector, timeout=10000)
                    print("CALL Clicked (Simulation)")
                else:
                    print(f"Looking for PUT button using selector: {put_selector}")
                    # page.click(put_selector, timeout=10000)
                    print("PUT Clicked (Simulation)")
            except Exception as e:
                print(f"Failed to execute web trade: {e}")
            finally:
                # browser.close() # We'll keep it open for demo purposes, or close it after 5 seconds
                import time
                time.sleep(2)
                browser.close()
