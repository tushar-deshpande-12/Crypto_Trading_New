"""
CCXT-based Crypto Data Client

Provides unified access to multiple exchanges with support for:
- Multiple timeframes (1m to 1w)
- Maximum candle fetching
- Multiple exchanges (Binance, Coinbase, Kraken, etc.)
"""

import ccxt
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CCXTClient:
    """
    Unified crypto data client using CCXT library.

    Features:
    - Support for 100+ exchanges
    - All timeframes from 1m to 1M
    - Automatic pagination for max candles
    - Rate limiting handling
    """

    # Timeframe configurations with max candles per request
    TIMEFRAMES = {
        '1m': {'label': '1 Minute', 'minutes': 1, 'max_candles': 1000},
        '3m': {'label': '3 Minutes', 'minutes': 3, 'max_candles': 1000},
        '5m': {'label': '5 Minutes', 'minutes': 5, 'max_candles': 1000},
        '15m': {'label': '15 Minutes', 'minutes': 15, 'max_candles': 1000},
        '30m': {'label': '30 Minutes', 'minutes': 30, 'max_candles': 1000},
        '1h': {'label': '1 Hour', 'minutes': 60, 'max_candles': 1000},
        '2h': {'label': '2 Hours', 'minutes': 120, 'max_candles': 1000},
        '4h': {'label': '4 Hours', 'minutes': 240, 'max_candles': 1000},
        '6h': {'label': '6 Hours', 'minutes': 360, 'max_candles': 1000},
        '8h': {'label': '8 Hours', 'minutes': 480, 'max_candles': 1000},
        '12h': {'label': '12 Hours', 'minutes': 720, 'max_candles': 1000},
        '1d': {'label': '1 Day', 'minutes': 1440, 'max_candles': 1000},
        '3d': {'label': '3 Days', 'minutes': 4320, 'max_candles': 1000},
        '1w': {'label': '1 Week', 'minutes': 10080, 'max_candles': 1000},
        '1M': {'label': '1 Month', 'minutes': 43200, 'max_candles': 1000},
    }

    # Supported exchanges
    EXCHANGES = {
        'binance': ccxt.binance,
        'coinbase': ccxt.coinbase,
        'kraken': ccxt.kraken,
        'kucoin': ccxt.kucoin,
        'bybit': ccxt.bybit,
        'okx': ccxt.okx,
    }

    def __init__(self, exchange_id: str = 'binance', sandbox: bool = False):
        """
        Initialize CCXT client.

        Args:
            exchange_id: Exchange identifier (binance, coinbase, etc.)
            sandbox: Use sandbox/testnet mode
        """
        self.exchange_id = exchange_id.lower()

        if self.exchange_id not in self.EXCHANGES:
            raise ValueError(f"Unsupported exchange: {exchange_id}. "
                           f"Supported: {list(self.EXCHANGES.keys())}")

        # Initialize exchange
        exchange_class = self.EXCHANGES[self.exchange_id]
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
            'options': {
                'defaultType': 'spot',
            }
        })

        # Load markets
        try:
            self.exchange.load_markets()
            logger.info(f"Connected to {self.exchange_id} - {len(self.exchange.symbols)} symbols available")
        except Exception as e:
            logger.error(f"Failed to load markets: {e}")
            raise

    def get_available_timeframes(self) -> List[str]:
        """Get timeframes supported by the exchange."""
        if hasattr(self.exchange, 'timeframes') and self.exchange.timeframes:
            return list(self.exchange.timeframes.keys())
        return list(self.TIMEFRAMES.keys())

    def get_symbols(self, quote_currency: str = 'USDT') -> List[str]:
        """
        Get all trading symbols for a quote currency.

        Args:
            quote_currency: Quote currency to filter by (default: USDT)

        Returns:
            List of symbol strings
        """
        symbols = []
        for symbol in self.exchange.symbols:
            if '/' in symbol:
                base, quote = symbol.split('/')
                if quote == quote_currency:
                    symbols.append(symbol)
        return sorted(symbols)

    def get_ohlcv(self, symbol: str, timeframe: str = '1h',
                  limit: int = 500, since: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (1m, 5m, 1h, 1d, etc.)
            limit: Number of candles to fetch
            since: Start timestamp in milliseconds

        Returns:
            DataFrame with datetime index and OHLCV columns
        """
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)

            if not ohlcv:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            df.drop('timestamp', axis=1, inplace=True)

            return df

        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            return pd.DataFrame()

    def get_max_ohlcv(self, symbol: str, timeframe: str = '1h',
                      max_candles: int = 5000,
                      progress_callback=None) -> pd.DataFrame:
        """
        Fetch maximum available OHLCV data with pagination.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe
            max_candles: Maximum candles to fetch (will paginate if needed)
            progress_callback: Optional callback(current, total, message)

        Returns:
            DataFrame with all fetched OHLCV data
        """
        all_data = []
        fetched = 0
        batch_size = min(1000, max_candles)  # Most exchanges limit to 1000 per request

        # Calculate time delta per candle
        tf_info = self.TIMEFRAMES.get(timeframe, {'minutes': 60})
        candle_minutes = tf_info['minutes']

        # Start from now and go backwards
        end_time = int(datetime.now().timestamp() * 1000)

        while fetched < max_candles:
            remaining = max_candles - fetched
            fetch_limit = min(batch_size, remaining)

            # Calculate since timestamp
            since = end_time - (fetch_limit * candle_minutes * 60 * 1000)

            if progress_callback:
                progress_callback(fetched, max_candles, f"Fetching {symbol} {timeframe}...")

            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, fetch_limit)

                if not ohlcv:
                    break

                all_data = ohlcv + all_data  # Prepend older data
                fetched += len(ohlcv)

                # Update end_time to oldest candle timestamp
                end_time = ohlcv[0][0] - 1

                # Rate limiting
                time.sleep(self.exchange.rateLimit / 1000)

                # If we got fewer than requested, we've hit the limit
                if len(ohlcv) < fetch_limit:
                    break

            except Exception as e:
                logger.error(f"Error fetching batch: {e}")
                # On first fetch failure, raise the error so caller knows symbol is invalid
                if fetched == 0:
                    raise ValueError(f"Failed to fetch data for {symbol}: {str(e)}")
                break

        if not all_data:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df.drop('timestamp', axis=1, inplace=True)

        # Remove duplicates and sort
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)

        if progress_callback:
            progress_callback(len(df), max_candles, f"Fetched {len(df)} candles")

        logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
        return df

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Get current ticker data for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')

        Returns:
            Ticker dictionary with price, volume, etc.
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'price': ticker.get('last', 0),
                'bid': ticker.get('bid', 0),
                'ask': ticker.get('ask', 0),
                'high_24h': ticker.get('high', 0),
                'low_24h': ticker.get('low', 0),
                'volume_24h': ticker.get('baseVolume', 0),
                'quote_volume_24h': ticker.get('quoteVolume', 0),
                'price_change_pct': ticker.get('percentage', 0),
                'price_change': ticker.get('change', 0),
            }
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            return None

    def get_all_tickers(self, quote_currency: str = 'USDT') -> List[Dict]:
        """
        Get tickers for all symbols of a quote currency.

        Args:
            quote_currency: Quote currency to filter by

        Returns:
            List of ticker dictionaries
        """
        try:
            tickers = self.exchange.fetch_tickers()

            result = []
            for symbol, ticker in tickers.items():
                if '/' in symbol and symbol.endswith(f'/{quote_currency}'):
                    result.append({
                        'symbol': symbol,
                        'price': ticker.get('last', 0),
                        'high_24h': ticker.get('high', 0),
                        'low_24h': ticker.get('low', 0),
                        'volume_24h': ticker.get('baseVolume', 0),
                        'quote_volume': ticker.get('quoteVolume', 0),
                        'price_change_pct': ticker.get('percentage', 0),
                        'price_change': ticker.get('change', 0),
                    })

            return sorted(result, key=lambda x: x.get('quote_volume', 0), reverse=True)

        except Exception as e:
            logger.error(f"Failed to fetch tickers: {e}")
            return []

    def convert_symbol_format(self, symbol: str, to_ccxt: bool = True) -> str:
        """
        Convert between CCXT format (BTC/USDT) and exchange format (BTCUSDT).

        Args:
            symbol: Symbol to convert
            to_ccxt: If True, convert to CCXT format, else to exchange format

        Returns:
            Converted symbol string
        """
        if to_ccxt:
            # BTCUSDT -> BTC/USDT
            if '/' not in symbol and symbol.endswith('USDT'):
                return f"{symbol[:-4]}/USDT"
            return symbol
        else:
            # BTC/USDT -> BTCUSDT
            return symbol.replace('/', '')

    def close(self):
        """Clean up resources."""
        pass  # CCXT handles cleanup automatically


# Singleton instance for convenience (thread-safe)
import threading
_client_lock = threading.Lock()
_clients: Dict[str, 'CCXTClient'] = {}

def get_ccxt_client(exchange_id: str = 'binance') -> CCXTClient:
    """Get or create a CCXT client instance (cached per exchange)."""
    global _clients
    with _client_lock:
        if exchange_id not in _clients:
            _clients[exchange_id] = CCXTClient(exchange_id)
        return _clients[exchange_id]
