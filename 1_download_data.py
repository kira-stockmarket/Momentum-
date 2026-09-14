import yfinance as yf
import pandas as pd
from ta import add_all_ta_features
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('data', exist_ok=True)

# Define Nifty 100 Universe (Expand to all 100 in production)
NIFTY_100 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS"
]

def build_dataset():
    all_data = []
    print("Phase 1: Downloading max historical data and building features...")
    
    for ticker in NIFTY_100:
        try:
            print(f"Processing {ticker}...")
            df = yf.download(ticker, period="max", progress=False)
            
            if df.empty or len(df) < 500:
                continue
                
            # FIX: Flatten the multi-level columns from recent yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Massive Feature Engineering using the 'ta' library
            df = add_all_ta_features(
                df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
            )
            
            # Target creation: Did the stock jump >= 20% in the next 21 trading days (1 month)?
            df['Future_Return'] = df['Close'].shift(-21) / df['Close'] - 1
            df['Target'] = (df['Future_Return'] >= 0.20).astype(int)
            df['Ticker'] = ticker
            
            # Drop the shifted future return column to prevent data leakage in training
            df.drop(columns=['Future_Return'], inplace=True)
            
            all_data.append(df.dropna())
        except Exception as e:
            print(f"Failed {ticker}: {e}")
            
    if not all_data:
        print("ERROR: No data was successfully downloaded and processed.")
        return
        
    master_df = pd.concat(all_data)
    master_df.sort_index(inplace=True) 
    
    # Save to Parquet format
    master_df.to_parquet('data/nifty100_features.parquet')
    print(f"Data saved successfully. Total rows: {len(master_df)}, Total Features: {len(master_df.columns)}")

if __name__ == "__main__":
    build_dataset()
