from data_loader import init_mt5, fetch_data
from features import feature_engineering_pipeline

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
        
    print("Raw Data sample:")
    print(raw_df[['open', 'high', 'low', 'close', 'tick_volume']].head())

    # Apply Feature Engineering
    processed_df = feature_engineering_pipeline(raw_df)
    
    print("\n" + "="*50)
    print("PROCESSED DATA: HEAD")
    print("="*50)
    columns_to_show = ['open', 'close', 'RSI', 'upper_shadow_ratio', 'lower_shadow_ratio', 'body_direction', 'Target']
    print(processed_df[columns_to_show].head(10))
    
    print("\n" + "="*50)
    print("PROCESSED DATA: TAIL")
    print("="*50)
    print(processed_df[columns_to_show].tail(10))

    # Train LSTM
    from lstm_model import prepare_sequential_data, train_and_evaluate
    X_train, X_test, y_train, y_test, scaler = prepare_sequential_data(processed_df)
    model, history, acc = train_and_evaluate(X_train, X_test, y_train, y_test)
    
if __name__ == "__main__":
    main()
