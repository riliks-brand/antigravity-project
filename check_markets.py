import MetaTrader5 as mt5
import datetime

mt5.initialize()

symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'US30', 'BTCUSD']
print('='*55)
print('  MARKET STATUS CHECK')
print('='*55)

tick = mt5.symbol_info_tick('EURUSD')
if tick:
    server_time = datetime.datetime.utcfromtimestamp(tick.time)
    hour = server_time.hour
    print(f'  Server Time (UTC): {server_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'  Server Hour: {hour}')
    print()
    
    if 7 <= hour < 16:
        session = 'London'
    elif 13 <= hour < 22:
        session = 'New York'
    elif hour >= 0 and hour < 7:
        session = 'Asia'
    else:
        session = 'OFF HOURS'
    print(f'  Active Session: {session}')
    print()

print('-'*55)
print(f'  {"Symbol":<12} {"Status":<15} {"Spread":<15} {"Bid":<12}')
print('-'*55)

for sym in symbols:
    info = mt5.symbol_info(sym)
    tick = mt5.symbol_info_tick(sym)
    if info is None:
        print(f'  {sym:<12} NOT FOUND ON BROKER')
        continue
    
    if tick and info.point > 0:
        spread = (tick.ask - tick.bid) / info.point
        trade_mode = info.trade_mode
        if trade_mode == 0:
            status = 'DISABLED'
        elif trade_mode == 4:
            status = 'CLOSE ONLY'
        elif trade_mode == 2:
            status = 'OPEN'
        else:
            status = f'MODE={trade_mode}'
        print(f'  {sym:<12} {status:<15} {spread:<15.1f} {tick.bid:<12.5f}')
    else:
        print(f'  {sym:<12} NO TICK DATA')

print('-'*55)
mt5.shutdown()
