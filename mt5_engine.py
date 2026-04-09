import MetaTrader5 as mt5

def connect_to_exness():
    """
    Initializes connection and logs into the MT5 Terminal for Exness.
    Prioritizes attaching to an already open terminal.
    """
    from config import Config
    import time
    
    # 1. Force shutdown of any stale handles first
    mt5.shutdown()
    time.sleep(1)
    
    # 2. Attempt initialization with Retries
    max_retries = 3
    connected = False
    
    print(f"\033[94m[MT5] Attempting to link with the open terminal...\033[0m")
    
    for attempt in range(max_retries):
        # Try local attach first (no path)
        if mt5.initialize():
            connected = True
            break
        # If that fails, try explicit path
        if mt5.initialize(path=Config.MT5_PATH):
            connected = True
            break
            
        print(f"\033[93m[MT5] Connection attempt {attempt+1} failed. Retrying in 2s...\033[0m")
        time.sleep(2)
        
    if not connected:
        error = mt5.last_error()
        print(f"\033[91mMT5 Initialization failed. Error code: {error}\033[0m")
        if error[0] == -6:
            print("\033[93m[Tip] Terminal reports Auth Error. Please close MT5 and reopen it.\033[0m")
        else:
            print("\033[93m[Tip] Make sure MT5 is open and 'Allow Algorithmic Trading' is Checked.\033[0m")
        return False
        
    # Attempt Login
    authorized = mt5.login(
        login=Config.LOGIN,
        password=Config.PASSWORD,
        server=Config.SERVER
    )
    
    if not authorized:
        print(f"\033[91mMT5 Login failed for account {Config.LOGIN}. Error code: {mt5.last_error()}\033[0m")
        return False
        
    print(f"\033[92mConnected & Logged into Exness Account {Config.LOGIN} Successfully!\033[0m")
    return True

def get_tick_info(symbol):
    """
    Retrieves latest tick and symbol properties.
    Ensures symbol is selected in Market Watch.
    """
    # 1. Force select in Market Watch (Crucial for getting info)
    if not mt5.symbol_select(symbol, True):
        print(f"\033[93m[MT5] Symbol '{symbol}' not found in terminal symbols. Trying to find it...\033[0m")
        # Optional: scan for common suffixes like .m, m, etc if needed.
    
    info = mt5.symbol_info(symbol)
    if not info:
        print(f"\033[91mFailed to retrieve info for symbol {symbol}. Is it available in your account symbols?\033[0m")
        return None, None
        
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print(f"\033[93m[MT5] Waiting for first tick for {symbol}...\033[0m")
        return None, info
        
    return tick, info

def calculate_lot_size(symbol, risk_amount, stop_loss_points):
    """
    Calculates the exact Lot size based on the risk amount, 
    accounting for tick value and MT5 volume parameters.
    """
    tick, info = get_tick_info(symbol)
    if not info:
        return 0.0

    # EURUSD doesn't usually have issues, but for reliability:
    tick_value = info.trade_tick_value
    if tick_value is None or tick_value <= 0:
        # Fallback to standard math if tick_value is missing (rare on EURUSD)
        tick_value = 1.0  
        
    lot_size = risk_amount / (stop_loss_points * tick_value)
    
    lot_step = info.volume_step
    lot_size = round(lot_size / lot_step) * lot_step
    lot_size = max(lot_size, info.volume_min)
    lot_size = min(lot_size, info.volume_max)
    
    digits = 0
    if "." in str(lot_step):
        digits = len(str(lot_step).split('.')[1])
    return round(float(lot_size), digits)

def execute_forex_trade(action, symbol, risk_amount, sl_points, tp_points):
    """
    Builds the MT5 order payload based on risk params and executes the trade.
    """
    tick, info = get_tick_info(symbol)
    if not tick or not info:
        return False, "Failed to get symbol info or tick"

    lot = calculate_lot_size(symbol, risk_amount, sl_points)
    if lot <= 0:
        return False, f"Invalid lot calculation: {lot}"

    price = tick.ask if action == 'BUY' else tick.bid
    
    if action == 'BUY':
        sl_price = price - (sl_points * info.point)
        tp_price = price + (tp_points * info.point)
        order_type = mt5.ORDER_TYPE_BUY
    else:
        sl_price = price + (sl_points * info.point)
        tp_price = price - (tp_points * info.point)
        order_type = mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot),
        "type": order_type,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": 10,
        "magic": 121052,
        "comment": "Antigravity_Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    
    if result is None:
        return False, "MT5 order_send returned None (Internal error)"
        
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        # Fallback for FOK filling
        if result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
            print("\033[93m[Retry] Broker requires FOK. Retrying with ORDER_FILLING_FOK...\033[0m")
            request["type_filling"] = mt5.ORDER_FILLING_FOK
            result = mt5.order_send(request)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"\033[92m[MT5] Success (FOK)! Ticket: {result.order}\033[0m")
                return True, f"Success (FOK). Ticket: {result.order}"
        
        err_msg = f"Order failed: {result.comment} (code: {result.retcode})"
        print(f"\033[91m[MT5 Error] {err_msg}\033[0m")
        return False, err_msg

    print(f"\033[92m[MT5] Successfully opened Forex Trade! Ticket: {result.order}\033[0m")
    return True, f"Success. Ticket: {result.order}"
