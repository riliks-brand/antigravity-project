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
    print("DEMO EXECUTION (PAPER TRADE MT5)")
    print("="*50)
    executor = TradeExecutor()
    ticket = executor.execute_mt5(action="buy", volume=0.01)
    if ticket:
        print(f"Successfully opened trade in MT5. Ticket ID: {ticket}")
    else:
        print("Failed to open trade in MT5.")

if __name__ == "__main__":
    main()
