import time
import pandas as pd
from binance.client import Client
from datetime import datetime
import requests
from pathlib import Path
from tqdm import tqdm

from rsi_macd_appraoch import rsi_macd_buy
from stochastic_oscillator_ma_approach import stochastic_ma_buy
from crypto_scout import get_all_cryptos_by_volume
from dataset_formation import fetch_and_process_crypto_data
from pivot_candlestick_stratergy import identify_candlestick_patterns, identify_strong_buy_sell_points

api_key = "61e28e26a9db6e61b9afd8d7"  # Replace with your actual API key
datapoints = 1000
check_days = 2
check_days = 2*6

def convert_inr_to_usdt(amount_inr, api_key):
    """
    Convert Indian Rupees (INR) to USDT (Tether) using the ExchangeRate-API.

    Parameters:
    - amount_inr (float): The amount in Indian Rupees to be converted.
    - api_key (str): Your API key for the ExchangeRate-API.

    Returns:
    - float: The equivalent amount in USDT.
    """
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/INR"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception("Error fetching exchange rate data")
    
    data = response.json()
    exchange_rate = data['conversion_rates']['USD']
    amount_usdt = amount_inr * exchange_rate
    
    return amount_usdt

def get_user_datetime():
    """
    Get user input for date and time.
    """
    from dateutil import parser
    while True:
        date_string = input("Enter date and time for prediction (e.g., '2023-05-17 14:30:00'): ")
        try:
            user_datetime = parser.parse(date_string)
            return user_datetime
        except ValueError:
            print("Unable to parse the date string. Please try again.")
            
def get_script_directory():
    return Path(__file__).parent.absolute()

def analyze_cryptos():
    check_days = 2*6
    current_dir = get_script_directory()
    print(f"Script is running in directory: {current_dir}")

    all_cryptos = get_all_cryptos_by_volume()
    all_symbols = all_cryptos['symbol'].tolist()
    strong_buy_signals = []

    for symbol in tqdm(all_symbols, desc="Processing cryptocurrencies"):
        if symbol.endswith('USDT'):
            print(f"Processing {symbol}...")
        
            try:    
                filename = fetch_and_process_crypto_data(current_dir, symbol, Client.KLINE_INTERVAL_1HOUR, datapoints)
                data = pd.read_csv(filename)
                
                patterns = identify_candlestick_patterns(data)
                buy_points, sell_points = identify_strong_buy_sell_points(data, patterns)
                
                data = stochastic_ma_buy(data)
                data = rsi_macd_buy(data)
                
                rsi_macd_signal = data['MACD_RSI']
                stoch_signal = data['Stoch_MA_Signal']
                
                check_days = 3
                pivot_signal = buy_points.iloc[-check_days:].any()
                stoch_signal = (stoch_signal.iloc[-check_days:] == 'Strong Buy').any()
                rsi_macd_signal = (rsi_macd_signal.iloc[-check_days:] == 'Strong Buy').any() or (rsi_macd_signal.iloc[-check_days:] == 'Buy').any()
                
                if pivot_signal or stoch_signal or rsi_macd_signal:
                    strong_buy_signals.append(symbol)
                    
                    print("-----------------------------------")
                    
                    print(f"Strong buy signal detected for {symbol}")
                    if pivot_signal:
                        print("  - Pivot Candlestick: Strong Buy")
                    if stoch_signal:
                        print("  - Stochastic Oscillator: Strong Buy")
                    if rsi_macd_signal:
                        print("  - RSI and MACD: Strong Buy")
                
                print("-----------------------------------")
                
                output_filename = f"{symbol}.csv"
                output_path = current_dir / output_filename
                data.to_csv(output_path, index=False)
                print(f"Dataset saved as: {output_path}")
                
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
            
            time.sleep(1)  # To avoid hitting API rate limits

    print("\nCryptocurrencies with strong buy signals in the last 5 candles:")
    for symbol in strong_buy_signals:
        print(symbol)
        
    strong_buys_filename = "strong_buys.csv"
    strong_buys_path = current_dir / strong_buys_filename
    
    strong_buys_df = pd.DataFrame({"symbol": strong_buy_signals})
    
    strong_buys_df.to_csv(strong_buys_path, index=False)
    print(f"\nStrong buy signals saved to: {strong_buys_path}")

    print("\nAnalysis complete.")

# Main execution
if __name__ == "__main__":
    analyze_cryptos()