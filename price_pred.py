import time
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client
from datetime import datetime, timedelta

import requests
import time
import os
# Add this near the top of your file, with your other imports
from pathlib import Path

from transformer import *
from rsi_macd_appraoch import *
from stochastic_oscillator_ma_approach import *
from dataset_formation import *
from crypto_scout import *
from web_data_harvester import *
from pivot_candlestick_stratergy import *

api_key = "61e28e26a9db6e61b9afd8d7"  # Replace with your actual API key
datapoints = 2000
symbol = "ETH"
symbol += "USDT"
buy_price_inr = 130
RUN_INFERRENCE = True
SCRAPING_ENABLED = False

def convert_inr_to_usdt(amount_inr, api_key):
    """
    Convert Indian Rupees (INR) to USDT (Tether) using the ExchangeRate-API.

    Parameters:
    - amount_inr (float): The amount in Indian Rupees to be converted.
    - api_key (str): Your API key for the ExchangeRate-API.

    Returns:
    - float: The equivalent amount in USDT.
    """
    # Define the API endpoint
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/INR"
    
    # Make the API request
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code != 200:
        raise Exception("Error fetching exchange rate data")
    
    # Parse the JSON response
    data = response.json()
    
    # Get the exchange rate for USDT
    exchange_rate = data['conversion_rates']['USD']
    
    # Convert the amount from INR to USDT
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
            
# Add this function to get the current working directory
def get_script_directory():
    return Path(__file__).parent.absolute()


# Main execution
if __name__ == "__main__":
    # # Example usage of the functions
    
    current_dir = get_script_directory()
    
    print(f"Script is running in directory: {current_dir}")
    filename = fetch_and_process_crypto_data(current_dir,symbol, Client.KLINE_INTERVAL_4HOUR, datapoints)
    
    get_top_200_by_volume()
    # buy_price_usdt = convert_inr_to_usdt(buy_price_inr,api_key)
    buy_price_usdt = 1
    
    data = pd.read_csv(filename)
    data = rsi_macd_buy(data)
    data = rsi_macd_sell(data)
    data = stochastic_ma_buy(data)
    data = stochastic_ma_sell(data)
    patterns = identify_candlestick_patterns(data)
    
    
    # Delete unnecessary columns
    columns_to_delete = ['MACD_12_26_9.1',	'MACDh_12_26_9.1',	'MACDs_12_26_9.1']
    data = data.drop(columns=columns_to_delete, errors='ignore')
    
    # Print the remaining columns
    print("Remaining columns after deletion:")
    print(data.columns.tolist())

    # Save the dataset with the symbol name in the current directory
    output_filename = f"{symbol}.csv"
    output_path = current_dir / output_filename
    data.to_csv(output_path, index=False)
    print(f"Dataset saved as: {output_path}")


    # Get user input for buy price and investment time
    investment_time = datetime.datetime(2024, 9, 3, 2, 53)  # Year, Month, Day, Hour, Minute

    # Plot interactive candlestick chart
    plot_interactive_candlestick(data, buy_price_usdt)
    plot_stochastic_ma(data, buy_price_usdt)
    plot_candlestick_patterns_with_signals(data, patterns)
    
    # if SCRAPING_ENABLED:
        
    #     print("Scraping cryptocurrency data...")
    #     df = scrape_cryptocurrencies()
        
    #     # Measure scraping time
    #     print("Measuring scraping time...")
    #     scraping_time = measure_scraping_time(scrape_coinmarketcap)
    #     print(f"Scraping time: {scraping_time:.2f} seconds")
        
    #     # Save data to CSV
    #     print("Saving data to CSV...")
    #     csv_path = save_to_csv(df)
        
    #     if csv_path:
    #         print(f"Data saved successfully to: {csv_path}")
    #     else:
    #         print("Failed to save data to CSV.")

    #     print("Web data harvesting complete.")

    if RUN_INFERRENCE:
        # Train and evaluate the model
        model, X_val, y_val = train_transformer_model(data, symbol=symbol)
        evaluate_model(model, X_val, y_val)

        # Make predictions
        predictions_df, std_dev, profitable_time, time_to_profit = predict_output_transformer(model, data, buy_price_usdt, investment_time)
        print(f'Standard Deviation of Prediction Errors: {std_dev}')
        print(f'Time to profit: {time_to_profit}')
    
    




