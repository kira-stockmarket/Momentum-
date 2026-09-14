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
            # Use auto_adjust=False to prevent yfinance from doing weird price modifications
            df = yf.download(ticker, period="max", progress=False, auto_adjust=False)
            
            if df.empty or len(df) < 500:
                print(f"  -> Skipped {ticker}: Not enough data ({len(df)} rows)")
                continue
                
            # BULLETPROOFING: Flatten Multi-Index if it exists
            if isinstance(df.columns, pd.MultiIndex):
                # Take the first level (Price type) and ignore the ticker level
                df.columns = [str(c[0]).strip().capitalize() for c in df.columns]
            else:
                df.columns = [str(c).strip().capitalize() for c in df.columns]
            
            # Ensure we have the exact columns required by the 'ta' library
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [c for c in required_cols if c not in df.columns]
            
            if missing_cols:
                # Sometimes yfinance returns "Adj close", let's rename it if needed
                if 'Adj close' in df.columns and 'Close' not in df.columns:
                    df.rename(columns={'Adj close': 'Close'}, inplace=True)
                else:
                    print(f"  -> Skipped {ticker}: Missing columns {missing_cols}. Found: {df.columns.tolist()}")
                    continue

            # Keep only the required columns and force them to float type
            df = df[required_cols].astype(float)
            
            # Drop rows where 'Close' or 'Volume' is completely missing (market holidays)
            df.dropna(subset=['Close', 'Volume'], inplace=True)
            
            # Strip timezones from the index to prevent indexing bugs
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Feature Engineering using the 'ta' library
            df = add_all_ta_features(
                df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
            )
            
            # Target creation: Did the stock jump >= 20% in the next 21 trading days?
            df['Future_Return'] = df['Close'].shift(-21) / df['Close'] - 1
            df['Target'] = (df['Future_Return'] >= 0.20).astype(int)
            df['Ticker'] = ticker
            
            df.drop(columns=['Future_Return'], inplace=True)
            
            # Drop the last 21 days which have no target
            df = df.iloc[:-21]
            
            all_data.append(df)
            print(f"  -> Success: {ticker} ({len(df)} rows)")
            
        except Exception as e:
            print(f"  -> Failed {ticker}: {e}")
            # This will print the exact reason it failed in your GitHub logs
            traceback.print_exc() 
            
    if not all_data:
        print("CRITICAL ERROR: No data was successfully downloaded and processed.")
        sys.exit(1)
        
    print("Concatenating all stocks...")
    master_df = pd.concat(all_data)
    master_df.sort_index(inplace=True) 
    
    master_df.to_parquet('data/nifty100_features.parquet')
    print(f"Data saved successfully. Total rows: {len(master_df)}, Total Features: {len(master_df.columns)}")

if __name__ == "__main__":
    build_dataset()
