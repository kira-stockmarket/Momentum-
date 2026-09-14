import yfinance as yf
import pandas as pd
from ta import add_all_ta_features
import os
import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

os.makedirs('data', exist_ok=True)

# Define Nifty 100 Universe
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
                print(f"  -> Skipped {ticker}: Not enough data ({len(df)} rows)")
                continue
                
            # Flatten the multi-level columns from recent yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [str(c[0]) for c in df.columns]
            
            # Massive Feature Engineering using the 'ta' library
            df = add_all_ta_features(
                df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
            )
            
            # Target creation: Did the stock jump >= 20% in the next 21 trading days (1 month)?
            df['Future_Return'] = df['Close'].shift(-21) / df['Close'] - 1
            df['Target'] = (df['Future_Return'] >= 0.20).astype(int)
            df['Ticker'] = ticker
            
            df.drop(columns=['Future_Return'], inplace=True)
            
            # FIX: Do not use dropna() because a single faulty indicator will delete all rows.
            # Instead, we just drop the last 21 days (which have no target due to shift) 
            # and let LightGBM natively handle any remaining NaNs in the features.
            df = df.iloc[:-21]
            
            all_data.append(df)
            print(f"  -> Success: {ticker} ({len(df)} rows)")
            
        except Exception as e:
            print(f"  -> Failed {ticker}: {e}")
            traceback.print_exc() # Prints the exact line that caused the error
            
    if not all_data:
        print("CRITICAL ERROR: No data was successfully downloaded and processed.")
        sys.exit(1) # This forces the GitHub Action to fail here, stopping Phase 2.
        
    master_df = pd.concat(all_data)
    master_df.sort_index(inplace=True) 
    
    # Save to Parquet format
    master_df.to_parquet('data/nifty100_features.parquet')
    print(f"Data saved successfully. Total rows: {len(master_df)}, Total Features: {len(master_df.columns)}")

if __name__ == "__main__":
    build_dataset()
