# fetcher.py - COMPLETE REWRITE WITH FEATURE ENGINEERING
import requests
import logging
import numpy as np
import pandas as pd
import ta
from typing import List, Dict, Optional
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CryptoDataFetcher:
    BASE_URL = "https://api.binance.com/api/v3"
    REQUEST_WEIGHT_LIMIT = 1200
    WEIGHT_PER_KLINES_REQUEST = 1
    
    def __init__(self, interval: str = "1h"):
        self.interval = interval
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "CryptoDataFetcher/2.0"})
    
    def fetch_ohlcv(self, symbol: str, limit: int = 1000, start_time: Optional[int] = None, end_time: Optional[int] = None) -> Optional[List[Dict]]:
        """Fetch raw OHLCV from Binance."""
        symbol = symbol.upper().strip()
        if not symbol.endswith("USDT"):
            symbol += "USDT"
        
        params = {
            "symbol": symbol,
            "interval": self.interval,
            "limit": min(limit, 1000)
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        try:
            logger.info(f"Fetching {self.interval} data for {symbol}, limit={params['limit']}")
            response = self.session.get(f"{self.BASE_URL}/klines", params=params, timeout=10)
            response.raise_for_status()
            raw_data = response.json()
            formatted_data = self.format_klines(raw_data, symbol)
            logger.info(f"Successfully fetched {len(formatted_data)} candlesticks for {symbol}")
            return formatted_data
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None
    
    def fetch_latest_ohlcv(self, symbol: str, hours_back: int = 24) -> Optional[List[Dict]]:
        """Fetch recent data."""
        end_time = int(time.time() * 1000)
        start_time = end_time - (hours_back * 60 * 60 * 1000)
        return self.fetch_ohlcv(symbol, hours_back, start_time, end_time)
    
    def fetch_max_historical_data(self, symbol: str, max_candles: int = 10000, progress_callback=None) -> Optional[List[Dict]]:
        """Fetch max data + compute ML features & directional labels."""
        all_data = []
        current_end_time = int(time.time() * 1000)
        batch_size = 1000
        batches_needed = (max_candles // batch_size) + 1
        
        for batch_num in range(batches_needed):
            remaining = max_candles - len(all_data)
            current_limit = min(batch_size, remaining)
            if current_limit <= 0:
                break
            
            batch_data = self.fetch_ohlcv(symbol, current_limit, end_time=current_end_time)
            if not batch_data or len(batch_data) == 0:
                break
            
            all_data = batch_data + all_data  # Prepend (backwards in time)
            
            if progress_callback:
                progress_callback(len(all_data), max_candles)
            
            current_end_time = batch_data[0]['timestamp'] - 1
            if batch_num < batches_needed - 1:
                time.sleep(0.2)
        
        if len(all_data) < 50:
            logger.warning(f"Insufficient data for {symbol}: {len(all_data)} candles")
            return all_data
        
        logger.info(f"Computing features for {len(all_data)} candles of {symbol}")
        return self.compute_features(all_data)
    
    def format_klines(self, raw_klines: List, symbol: str) -> List[Dict]:
        """Format Binance klines to structured dicts."""
        formatted = []
        for kline in raw_klines:
            try:
                formatted.append({
                    'symbol': symbol,
                    'timestamp': int(kline[0]),
                    'datetime': datetime.fromtimestamp(kline[0] / 1000).isoformat(),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'close_time': int(kline[6]),
                    'quote_volume': float(kline[7]),
                    'trades': int(kline[8]),
                    'taker_buy_volume': float(kline[9]),
                    'taker_buy_quote_volume': float(kline[10])
                })
            except (IndexError, ValueError) as e:
                logger.warning(f"Skipping malformed kline: {e}")
                continue
        return formatted
    
    def compute_features(self, ohlcv_data: List[Dict]) -> List[Dict]:
        """Engineer features + directional labels for trading models."""
        df = pd.DataFrame(ohlcv_data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        # Price returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['return_1h'] = df['close'].pct_change(1)
        df['return_5h'] = df['close'].pct_change(5)
        df['return_24h'] = df['close'].pct_change(24)
        
        # Momentum indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd()
        df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
        df['stoch'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
        
        # Volatility
        df['volatility_20'] = df['log_return'].rolling(20).std()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_position'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Directional label
        threshold = 0.001
        df['direction'] = ((df['close'].shift(-1) / df['close'] - 1) > threshold).astype(int)
        df['direction'] = df['direction'].fillna(0)
        
        # Price position ratios
        df['high_low_ratio'] = df['close'] / df['low']
        df['low_high_ratio'] = df['close'] / df['high']
        
        df = df.dropna()
        result = df.reset_index().to_dict('records')
        
        logger.info(f"Features computed: {len(result)} rows")
        return result
    
    def get_available_symbols(self, quote_asset: str = "USDT") -> List[str]:
        """Fetch available trading pairs."""
        try:
            response = self.session.get(f"{self.BASE_URL}/exchangeInfo", timeout=10)
            exchange_info = response.json()
            symbols = [s['symbol'] for s in exchange_info.get('symbols', []) 
                      if s.get('quoteAsset') == quote_asset and s.get('status') == 'TRADING']
            return sorted(symbols)
        except Exception as e:
            logger.error(f"Failed to fetch symbols: {e}")
            return []
    
    def close(self):
        """Close session."""
        self.session.close()