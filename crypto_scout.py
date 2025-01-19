from binance.client import Client
import pandas as pd
import os
import datetime
from tqdm import tqdm
import time
import pandas_ta as ta

def get_all_cryptos_by_volume():
    """
    Fetch all cryptocurrencies sorted by trading volume with respect to USDT.
    Only updates the file if it hasn't been updated within the last hour.
    
    Returns:
    - pandas.DataFrame: A sorted dataframe of all traded cryptocurrencies with columns 'symbol', 'volume', and 'last_price'.
    """
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get current date and time
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d_%H")
    
    # Create the file path
    file_path = os.path.join(current_dir, f'all_cryptos_by_volume_{date_str}.csv')
    
    # Check if file already exists and is less than an hour old
    if os.path.exists(file_path):
        print(f"File already exists and is up-to-date: {file_path}")
        return pd.read_csv(file_path)
    
    # If file doesn't exist or is older than an hour, proceed with fetching new data
    client = Client()
    
    # Fetch all USDT trading pairs
    tickers = client.get_ticker()
    usdt_pairs = [ticker for ticker in tickers if ticker['symbol'].endswith('USDT')]
    
    # Create a dataframe with relevant information
    df = pd.DataFrame(usdt_pairs, columns=['symbol', 'volume', 'lastPrice'])
    df['volume'] = df['volume'].astype(float)
    df['lastPrice'] = df['lastPrice'].astype(float)
    
    # Sort by volume (descending to get highest volume first)
    df_sorted = df.sort_values('volume', ascending=False).reset_index(drop=True)
    
    # Rename columns for clarity
    df_sorted.columns = ['symbol', 'volume', 'last_price']
    
    # Save the DataFrame to CSV
    df_sorted.to_csv(file_path, index=False)
    
    print(f"All cryptocurrencies sorted by volume data saved as: {file_path}")
    
    return df_sorted