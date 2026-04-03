from data_loader import init_mt5, fetch_data
from features import feature_engineering_pipeline
import MetaTrader5 as mt5

def main():
    if not init_mt5():
        print("Failed to initialize MT5. Make sure the terminal is running.")
        return

    # Fetch raw data
    raw_df = fetch_data()
    if raw_df is None or raw_df.empty:
        mt5.shutdown()
        return
        
    print("Raw Data sample:")
    print(raw_df[['open', 'high', 'low', 'close', 'tick_volume']].head())

    # Apply Feature Engineering
    processed_df = feature_engineering_pipeline(raw_df)
    
    print("\nProcessed Data with Features:")
    print(processed_df.head())
    
    # Check the columns and target
    print("\nColumns:", processed_df.columns.tolist())
    print("Target distribution:", processed_df['Target'].value_counts())

    mt5.shutdown()

if __name__ == "__main__":
    main()
