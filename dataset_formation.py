import time
import pandas as pd
import pandas_ta as ta
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta


def calculate_pivot_points(high, low, close):
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, s1, r2, s2, r3, s3

def apply_pivot_points(df):
    # Check if 'timestamp' column exists, if not, try to create it
    if 'timestamp' not in df.columns:
        if 'Open time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Open time'], unit='ms')
        elif 'open_time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['open time'], unit='ms')
        else:
            print("Error: Neither 'timestamp', 'Open time', nor 'open_time' column found in the DataFrame")
            print("Available columns:", df.columns)
            return df  # Return the original DataFrame if we can't process it
        
    # Ensure the DataFrame has a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.set_index('timestamp', inplace=True)

    # Ensure 'High', 'Low', and 'Close' columns exist and are numeric
    required_columns = ['High', 'Low', 'Close']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: '{col}' column not found in the DataFrame")
            return df
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate pivot points for each day
    df['Date'] = df.index.date
    pivot_points = df.groupby('Date').agg({
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).apply(lambda x: pd.Series(calculate_pivot_points(x['High'], x['Low'], x['Close'])), axis=1)

    pivot_points.columns = ['Pivot', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3']

    # Merge the pivot points back to the original DataFrame
    df = df.merge(pivot_points, left_on='Date', right_index=True, how='left')

    # Forward fill the pivot points to ensure each row has a value
    pivot_columns = ['Pivot', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3']
    df[pivot_columns] = df[pivot_columns].ffill()

    # Drop the temporary 'Date' column
    df.drop('Date', axis=1, inplace=True)
    
    return df
    

def fetch_and_process_crypto_data(current_dir, symbol="BNBUSDT", interval=Client.KLINE_INTERVAL_4HOUR,  target_observations=50000):
    """
    Fetch and process cryptocurrency data from Binance.

    Parameters:
    - api_key (str): Binance API key
    - api_secret (str): Binance API secret
    - symbol (str): Trading symbol (default: "BNBUSDT")
    - interval (str): Kline interval (default: 4 hours)
    - target_observations (int): Approximate number of observations to fetch (default: 50000)

    Returns:
    - str: Filename of the saved CSV file
    """
    # Step 1: Initialize Binance Client
    client = Client('API_KEY', 'API_SECRET')

    # Step 2: Set Parameters for Fetching Data
    limit = target_observations  # Maximum number of data points per request
    klines = []

    # Fetch the first batch of klines
    klines.extend(client.get_klines(symbol=symbol, interval=interval, limit=limit))

    # Continue fetching data in batches until we have approximately target_observations
    while len(klines) < target_observations:
        # Get the last closing time from the most recent kline fetched
        last_close_time = klines[-1][6]  # Close time of the last fetched kline

        # Fetch the next batch of klines starting from the last close time
        new_klines = client.get_klines(symbol=symbol, interval=interval, limit=limit, startTime=last_close_time)

        # If no new klines are returned, break the loop
        if not new_klines:
            break

        # Append the new data to the existing list
        klines.extend(new_klines)

        # Pause briefly to respect Binance API rate limits
        time.sleep(0.5)  # Sleep for 500ms

    # Convert the klines data to a DataFrame
    data = pd.DataFrame(klines, columns=[
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
        'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
        'Taker buy quote asset volume', 'Ignore'])

    # Convert the data types
    data['Open time'] = pd.to_datetime(data['Open time'], unit='ms')
    data['Close time'] = pd.to_datetime(data['Close time'], unit='ms')
    data[['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume', 
          'Number of trades', 'Taker buy base asset volume', 
          'Taker buy quote asset volume']] = data[['Open', 'High', 'Low', 'Close', 
                                                   'Volume', 'Quote asset volume', 
                                                   'Number of trades', 'Taker buy base asset volume', 
                                                   'Taker buy quote asset volume']].apply(pd.to_numeric)

    # Step 3: Calculate Moving Averages and MACD
    data['SMA_5'] = ta.sma(data['Close'], length=5)
    data['SMA_20'] = ta.sma(data['Close'], length=20)
    data['SMA_50'] = ta.sma(data['Close'], length=50)
    data['SMA_100'] = ta.sma(data['Close'], length=100)

    data['EMA_5'] = ta.ema(data['Close'], length=5)
    data['EMA_20'] = ta.ema(data['Close'], length=20)
    data['EMA_50'] = ta.ema(data['Close'], length=50)
    data['EMA_100'] = ta.ema(data['Close'], length=100)
    data['EMA_200'] = ta.ema(data['Close'], length=200)
    
    # Calculate Stochastic Oscillator
    stoch = ta.stoch(data['High'], data['Low'], data['Close'], k=14, d=3, smooth_k=3)
    data['STOCH_K'] = stoch['STOCHk_14_3_3']
    data['STOCH_D'] = stoch['STOCHd_14_3_3']

    # Calculate MACD
    macd = ta.macd(data['Close'], fast=12, slow=26, signal=9)
    data['MACD_12_26_9'] = macd['MACD_12_26_9']
    data['MACDs_12_26_9'] = macd['MACDs_12_26_9']
    data['MACDh_12_26_9'] = macd['MACDh_12_26_9']

    # Calculate RSI
    data['RSI_14'] = ta.rsi(data['Close'], length=14)

    # Apply pivot point calculation to the data
    data = apply_pivot_points(data)


    # # Calculate distance from current price to pivot levels
    # for level in ['Pivot', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3']:
    #     data[f'{level}_Distance'] = (data[level] - data['Close']) / data['Close']

    # data = pd.concat([data, macd], axis=1)

    # Step 4: Remove Empty and "Empty-Like" Columns
    data = data[201:len(data)]

    filename = symbol + ".csv"
    
    filename = str(current_dir) + '\\' + filename

    data = data.dropna(axis=1 , how='all')
    # Step 6: Store Data in CSV
    
    # try:
    #     # Drop unnecessary columns
    #     columns_to_drop = ['MACD_12_26_9.1', 'MACDh_12_26_9.1', 'MACDs_12_26_9.1']
    #     data = data.drop(columns=columns_to_drop, errors='ignore')
    # except:
    #     pass
    
    # # Print remaining columns for verification
    # print("Remaining columns after deletion:")
    # print(data.columns.tolist())
    
    data.to_csv(filename, index=False)
    print(f"Crypto data saved as: {filename}")

    return filename