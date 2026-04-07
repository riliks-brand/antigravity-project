import MetaTrader5 as mt5

def connect_to_exness():
    """
    Initializes connection and logs into the MT5 Terminal for Exness.
    Includes retry logic for IPC timeouts.
    """
    from config import Config
    import time
    
    # Attempt initialization with Retries
    max_retries = 3
    connected = False
    for attempt in range(max_retries):
        if mt5.initialize():
            connected = True
            break
        print(f"\033[93m[MT5] Connection attempt {attempt+1} failed. Retrying in 2s...\033[0m")
        time.sleep(2)
        
    if not connected:
        print(f"\033[91mMT5 Initialization failed after {max_retries} attempts. Error code: {mt5.last_error()}\033[0m")
        print("\033[93m[Tip] Make sure the MetaTrader 5 Terminal is open on your Windows desktop.\033[0m")
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
    """
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if not info:
        print(f"\033[91mFailed to retrieve info for symbol {symbol}\033[0m")
        return None, None
    if not info.visible:
        # Try to make it visible
        if not mt5.symbol_select(symbol, True):
            print(f"\033[91msymbol_select({symbol}) failed, error: {mt5.last_error()}\033[0m")
            return None, None
    return tick, info

def calculate_lot_size(symbol, risk_amount, stop_loss_points):
    """
    Calculates the exact Lot size based on the risk amount, 
    accounting for tick value and MT5 volume parameters.
    """
    tick, info = get_tick_info(symbol)
    if not info:
        return 0.0

    tick_value = info.trade_tick_value
    tick_size = info.trade_tick_size
    point = info.point

    # Sometimes for crypto, tick_value is 0 for whatever reason, fallback or warn.
    if tick_value == 0:
        print("\033[93m[Warning] MT5 returned 0 for trade_tick_value. Validating...\033[0m")
        tick_value = 1.0  # safe guard for potential MT5 bugs in demo crypto

    print(f"[Lot Calculation Debug] Base Tick Value: {tick_value}, Tick Size: {tick_size}, Point: {point}")
    
    # Calculate initial lot size
    # If SL is in points, the formula relies on exact tick math:
    # 1 Lot risk = SL(points) * (tick_value / (tick_size / point))
    # It simplifies properly when tick_size == point. Usually it resolves to:
    lot_size = risk_amount / (stop_loss_points * tick_value)
    
    # Round to nearest valid contract size
    lot_step = info.volume_step
    lot_size = round(lot_size / lot_step) * lot_step
    
    # Ensure it's not below minimum or above maximum
    lot_size = max(lot_size, info.volume_min)
    lot_size = min(lot_size, info.volume_max)
    
    # Rounding properly based on lot_step precision to avoid MT5 error Invalid Volume
    digits = 0
    if "." in str(lot_step):
        digits = len(str(lot_step).split('.')[1])
    lot_size = round(lot_size, digits)

    return float(lot_size)

def execute_forex_trade(action, symbol, risk_amount, sl_points, tp_points):
    """
    Builds the MT5 order payload based on risk params and executes the trade.
    """
    tick, info = get_tick_info(symbol)
    if not tick or not info:
        return False, "Failed to get symbol info"

    lot = calculate_lot_size(symbol, risk_amount, sl_points)
    if lot <= 0:
        return False, f"Invalid lot calculation: {lot}"

    point = info.point
    
    # Use Ask for BUY, Bid for SELL
    price = tick.ask if action == 'BUY' else tick.bid
    
    # Calculate Stop Loss and Take Profit
    if action == 'BUY':
        sl_price = price - (sl_points * point)
        tp_price = price + (tp_points * point)
        order_type = mt5.ORDER_TYPE_BUY
    else: # SELL
        sl_price = price + (sl_points * point)
        tp_price = price - (tp_points * point)
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
        "magic": 121052, # Matches MAGIC_NUMBER but hardcoding is fine here
        "comment": "Antigravity_Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC, # standard for many brokers, some require FOK
    }
    
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        err_msg = f"Order execution failed, retcode={result.retcode} ({result.comment})"
        print(f"\033[91m[MT5 Error] {err_msg}\033[0m")
        
        # fallback for FOK filling
        if result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
            print("\033[93m[Retry] Retrying with ORDER_FILLING_FOK...\033[0m")
            request["type_filling"] = mt5.ORDER_FILLING_FOK
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return False, f"FOK Retry failed: retcode={result.retcode}"
            else:
                return True, f"Success (FOK). Ticket: {result.order}"
        return False, err_msg

    print(f"\033[92m[MT5] Successfully opened Forex Trade! Ticket: {result.order}\033[0m")
    return True, f"Success. Ticket: {result.order}"
