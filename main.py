from data_loader import init_mt5, fetch_data
from features import feature_engineering_pipeline
from executor import TradeExecutor

def main():
    if not init_mt5():
        print("Failed initialization.")
        return

    import pandas as pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Fetch raw data
    raw_df = fetch_data()
    if raw_df is None or raw_df.empty:
        return
        
    # Apply Feature Engineering
    processed_df = feature_engineering_pipeline(raw_df)
    
    print("\n" + "="*50)
    print("PROCESSED DATA: HEAD")
    print("="*50)
    # Including new features hour, day_of_week, and ATR
    columns_to_show = ['open', 'close', 'RSI', 'ATR', 'hour', 'day_of_week', 'Target']
    print(processed_df[[c for c in columns_to_show if c in processed_df.columns]].head(10))

    # Train LSTM
    from lstm_model import prepare_sequential_data, train_and_evaluate
    X_train, X_test, y_train, y_test, scaler = prepare_sequential_data(processed_df)
    model, history, acc = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*50)
    print("LIVE PREDICTION & EXECUTION")
    print("="*50)
    
    import numpy as np
    from config import Config
    
    latest_features = processed_df.drop(['Target'], axis=1).values
    latest_features_scaled = scaler.transform(latest_features)
    
    if len(latest_features_scaled) >= Config.SEQUENCE_LENGTH:
        X_live = latest_features_scaled[-Config.SEQUENCE_LENGTH:]
        X_live = np.array([X_live])
        
        prob = model.predict(X_live)[0][0]
        action = "buy" if prob > 0.5 else "sell"
        
        print(f"Prediction Probability: {prob:.4f} -> {action.upper()}")
        
        executor = TradeExecutor()
        trade_details = executor.execute_mt5(action=action, volume=0.01)
        if type(trade_details) is dict:
            print(f"\n[LIVE TRADE VERIFIED]")
            print(f" > Ticket ID:    {trade_details['ticket']}")
            print(f" > Entry Price:  {trade_details['entry_price']}")
            print(f" > Stop Loss:    {trade_details['sl']}")
            print(f" > Take Profit:  {trade_details['tp']}")
        else:
            print(f"Successfully opened trade in MT5. Ticket ID: {trade_details}")
    else:
        print("Not enough data to form a sequence.")

if __name__ == "__main__":
    main()
