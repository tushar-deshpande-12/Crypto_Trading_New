"""
Binance API Client Module
Handles all interactions with Binance public API endpoints
Enhanced with candlestick/kline data support
"""

import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinanceAPIClient:
    """
    Professional-grade Binance API client for market data retrieval
    Uses public endpoints - no authentication required
    """

    BASE_URL = "https://api.binance.com/api/v3"

    # Candlestick interval constants
    INTERVAL_1M = "1m"
    INTERVAL_3M = "3m"
    INTERVAL_5M = "5m"
    INTERVAL_15M = "15m"
    INTERVAL_30M = "30m"
    INTERVAL_1H = "1h"
    INTERVAL_2H = "2h"
    INTERVAL_4H = "4h"
    INTERVAL_6H = "6h"
    INTERVAL_8H = "8h"
    INTERVAL_12H = "12h"
    INTERVAL_1D = "1d"
    INTERVAL_3D = "3d"
    INTERVAL_1W = "1w"
    INTERVAL_1M_MONTH = "1M"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoTracker/2.0'
        })

    def get_exchange_info(self) -> Optional[Dict]:
        """
        Fetch all trading pairs and their configuration
        Returns: Dictionary with exchange information or None on error
        """
        try:
            response = self.session.get(f"{self.BASE_URL}/exchangeInfo", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch exchange info: {e}")
            return None

    def get_24h_ticker(self) -> Optional[List[Dict]]:
        """
        Fetch 24-hour rolling window price change statistics for all symbols

        Returns key metrics:
        - symbol: Trading pair (e.g., BTCUSDT)
        - priceChange: Absolute price change
        - priceChangePercent: Price change percentage
        - lastPrice: Most recent price
        - volume: Trading volume in base asset
        - quoteVolume: Trading volume in quote asset (usually USDT)
        - highPrice: 24h high
        - lowPrice: 24h low
        - openPrice: Price at start of 24h period
        - count: Number of trades
        """
        try:
            response = self.session.get(f"{self.BASE_URL}/ticker/24hr", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch 24h ticker data: {e}")
            return None

    def get_price_ticker(self) -> Optional[List[Dict]]:
        """
        Fetch latest price for all symbols
        Lightweight alternative to 24h ticker
        """
        try:
            response = self.session.get(f"{self.BASE_URL}/ticker/price", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch price ticker: {e}")
            return None

    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 500) -> Optional[List[List]]:
        """
        Fetch candlestick/kline data for a specific symbol

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candlestick interval (e.g., "1m", "5m", "1h", "1d")
            limit: Number of candles to fetch (max 1000, default 500)

        Returns:
            List of klines, each containing:
            [
                Open time (ms),
                Open price,
                High price,
                Low price,
                Close price,
                Volume,
                Close time (ms),
                Quote asset volume,
                Number of trades,
                Taker buy base asset volume,
                Taker buy quote asset volume,
                Ignore
            ]
        """
        try:
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': min(limit, 1000)  # API max is 1000
            }
            response = self.session.get(
                f"{self.BASE_URL}/klines",
                params=params,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch klines for {symbol}: {e}")
            return None

    def get_klines_formatted(self, symbol: str, interval: str = "1h", limit: int = 500) -> Optional[List[Dict]]:
        """
        Fetch and format candlestick data for easier consumption

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candlestick interval
            limit: Number of candles to fetch

        Returns:
            List of formatted candlestick dictionaries with OHLCV data
        """
        raw_klines = self.get_klines(symbol, interval, limit)

        if not raw_klines:
            return None

        formatted_klines = []
        for kline in raw_klines:
            formatted_klines.append({
                'timestamp': datetime.fromtimestamp(kline[0] / 1000),
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5]),
                'close_time': datetime.fromtimestamp(kline[6] / 1000),
                'quote_volume': float(kline[7]),
                'trades': int(kline[8]),
                'taker_buy_volume': float(kline[9]),
                'taker_buy_quote_volume': float(kline[10])
            })

        return formatted_klines

    def get_usdt_pairs_detailed(self) -> List[Dict]:
        """
        Get detailed 24h statistics for all USDT trading pairs
        Filters for USDT pairs as they're the most liquid and relevant

        Returns: List of dictionaries with formatted market data
        """
        ticker_data = self.get_24h_ticker()

        if not ticker_data:
            return []

        # Filter for USDT pairs and format data
        usdt_pairs = []
        for ticker in ticker_data:
            symbol = ticker.get('symbol', '')

            # Focus on USDT pairs (most liquid and commonly traded)
            if symbol.endswith('USDT'):
                formatted_data = {
                    'symbol': symbol,
                    'base_asset': symbol[:-4],  # Remove 'USDT' suffix
                    'price': float(ticker.get('lastPrice', 0)),
                    'price_change_pct': float(ticker.get('priceChangePercent', 0)),
                    'price_change': float(ticker.get('priceChange', 0)),
                    'volume_24h': float(ticker.get('volume', 0)),
                    'quote_volume_24h': float(ticker.get('quoteVolume', 0)),
                    'high_24h': float(ticker.get('highPrice', 0)),
                    'low_24h': float(ticker.get('lowPrice', 0)),
                    'open_price': float(ticker.get('openPrice', 0)),
                    'trades_count': int(ticker.get('count', 0)),
                }
                usdt_pairs.append(formatted_data)

        return usdt_pairs

    def close(self):
        """Clean up resources"""
        self.session.close()
