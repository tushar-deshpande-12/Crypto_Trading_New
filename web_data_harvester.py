import requests
from bs4 import BeautifulSoup
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time

def scrape_coinmarketcap(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    crypto_data = []
    rows = soup.select('tbody tr')
    
    for row in rows:
        name = row.select_one('.cmc-table__column-name')
        price = row.select_one('.cmc-table__cell--sort-by__price')
        market_cap = row.select_one('.cmc-table__cell--sort-by__market-cap')
        
        if name and price and market_cap:
            crypto_data.append({
                'Name': name.get_text(strip=True),
                'Price': price.get_text(strip=True),
                'Market Cap': market_cap.get_text(strip=True)
            })
    
    return crypto_data

def scrape_coingecko(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    crypto_data = []
    rows = soup.select('tbody tr')
    
    for row in rows:
        name = row.select_one('.coin-name')
        price = row.select_one('.td-price')
        market_cap = row.select_one('.td-market_cap')
        
        if name and price and market_cap:
            crypto_data.append({
                'Name': name.get_text(strip=True),
                'Price': price.get_text(strip=True),
                'Market Cap': market_cap.get_text(strip=True)
            })
    
    return crypto_data

def scrape_cryptocurrencies():
    urls = [
        'https://coinmarketcap.com/',
        'https://www.coingecko.com/en'
    ]
    
    with ThreadPoolExecutor(max_workers=len(urls)) as executor:
        results = list(executor.map(lambda url: scrape_coinmarketcap(url) if 'coinmarketcap' in url else scrape_coingecko(url), urls))
    
    all_data = [item for sublist in results for item in sublist]
    df = pd.DataFrame(all_data)
    
    # Remove duplicates based on the 'Name' column
    df = df.drop_duplicates(subset='Name', keep='first')
    
    # Clean and convert Market Cap to float, handling potential errors
    def clean_market_cap(value):
        try:
            # Remove currency symbols, commas, and convert to float
            cleaned = re.sub(r'[^\d.]', '', value)
            return float(cleaned)
        except ValueError:
            # Return NaN for values that can't be converted
            return float('nan')

    df['Market Cap'] = df['Market Cap'].apply(clean_market_cap)
    
    # Remove rows with NaN Market Cap and sort
    df = df.dropna(subset=['Market Cap']).sort_values('Market Cap', ascending=False)
    df = df.sort_values('Market Cap', ascending=False)
    
    return df


import time

def measure_scraping_time(scraping_function):
    start_time = time.time()
    scraping_function()
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    return elapsed_time

def save_to_csv(df, filename='cryptocurrency_web_data.csv'):
    """
    Save the scraped cryptocurrency data to a CSV file.
    
    Args:
    df (pandas.DataFrame): The DataFrame containing the scraped data.
    filename (str): The name of the file to save the data to. Defaults to 'cryptocurrency_data.csv'.
    
    Returns:
    str: The path to the saved CSV file.
    """
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create the full file path
        file_path = os.path.join(current_dir, filename)
        
        # Save the DataFrame to CSV
        df.to_csv(file_path, index=False)
        
        print(f"Data successfully saved to {file_path}")
        return file_path
    except Exception as e:
        print(f"An error occurred while saving the data: {str(e)}")
        return None




