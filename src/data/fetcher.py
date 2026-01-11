"""
Cryptocurrency Data Fetcher
Fetches 1-hour OHLCV (Open, High, Low, Close, Volume) data from Binance API
"""

import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class CryptoDataFetcher:
    """
    Professional-grade data fetcher for cryptocurrency OHLCV data
    Focuses on 1-hour interval data for prediction models
    """

    BASE_URL = "https://api.binance.com/api/v3"

    # Binance API rate limits: 1200 requests per minute
    REQUEST_WEIGHT_LIMIT = 1200
    WEIGHT_PER_KLINES_REQUEST = 1

    def __init__(self, interval: str = "1h"):
        """
        Initialize the data fetcher

        Args:
            interval: Candlestick interval (default: "1h")
                     Valid intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        """
        self.interval = interval
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoDataFetcher/1.0'
        })

    def fetch_ohlcv(
        self,
        symbol: str,
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Optional[List[Dict]]:
        """
        Fetch OHLCV candlestick data for a specific symbol

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            limit: Number of candlesticks to fetch (max 1000, default 1000)
            start_time: Start time in milliseconds (optional)
            end_time: End time in milliseconds (optional)

        Returns:
            List of OHLCV dictionaries or None on error
            Each dict contains: timestamp, open, high, low, close, volume, quote_volume, trades
        """
        try:
            # Validate symbol format
            symbol = symbol.upper().strip()
            if not symbol.endswith('USDT'):
                logger.warning(f"Symbol {symbol} doesn't end with USDT, appending it")
                symbol = f"{symbol}USDT"

            # Build request parameters
            params = {
                'symbol': symbol,
                'interval': self.interval,
                'limit': min(limit, 1000)  # Binance max is 1000
            }

            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time

            # Execute request
            logger.info(f"Fetching {self.interval} data for {symbol}, limit={params['limit']}")
            response = self.session.get(
                f"{self.BASE_URL}/klines",
                params=params,
                timeout=10
            )
            response.raise_for_status()

            # Parse and format response
            raw_data = response.json()
            formatted_data = self._format_klines(raw_data, symbol)

            logger.info(f"Successfully fetched {len(formatted_data)} candlesticks for {symbol}")
            return formatted_data

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {symbol}: {e}", exc_info=True)
            return None

    def fetch_latest_ohlcv(self, symbol: str, hours_back: int = 24) -> Optional[List[Dict]]:
        """
        Fetch latest OHLCV data going back N hours

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            hours_back: Number of hours of historical data to fetch

        Returns:
            List of OHLCV dictionaries or None on error
        """
        # Calculate start time
        end_time = int(time.time() * 1000)
        start_time = end_time - (hours_back * 60 * 60 * 1000)

        return self.fetch_ohlcv(
            symbol=symbol,
            limit=hours_back,  # 1 candle per hour
            start_time=start_time,
            end_time=end_time
        )

    def fetch_max_historical_data(
        self,
        symbol: str,
        max_candles: int = 10000,
        progress_callback=None
    ) -> Optional[List[Dict]]:
        """
        Fetch maximum historical data by making multiple API requests
        Goes back in time as far as possible to get the longest dataset

        For 1h interval:
        - 1000 candles = ~41 days
        - 10000 candles = ~416 days (13.7 months)
        - 50000 candles = ~5.7 years

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            max_candles: Maximum number of candles to fetch (default: 10000)
            progress_callback: Optional callback function(current, total) for progress updates

        Returns:
            List of OHLCV dictionaries sorted by timestamp (oldest first) or None on error
        """
        try:
            all_data = []
            current_end_time = int(time.time() * 1000)
            batch_size = 1000  # Binance API limit per request
            batches_needed = (max_candles + batch_size - 1) // batch_size

            logger.info(f"Fetching maximum historical data for {symbol}")
            logger.info(f"Target: {max_candles} candles across {batches_needed} batches")

            for batch_num in range(batches_needed):
                # Calculate how many candles to fetch in this batch
                remaining = max_candles - len(all_data)
                current_limit = min(batch_size, remaining)

                if current_limit <= 0:
                    break

                # Fetch batch
                logger.info(f"Batch {batch_num + 1}/{batches_needed}: Fetching {current_limit} candles ending at {datetime.fromtimestamp(current_end_time/1000)}")

                batch_data = self.fetch_ohlcv(
                    symbol=symbol,
                    limit=current_limit,
                    end_time=current_end_time
                )

                if not batch_data or len(batch_data) == 0:
                    logger.warning(f"No more data available at {datetime.fromtimestamp(current_end_time/1000)}")
                    break

                # Add to collection (prepend since we're going backwards in time)
                all_data = batch_data + all_data

                # Update progress
                if progress_callback:
                    progress_callback(len(all_data), max_candles)

                # If we got less than requested, we've hit the beginning
                if len(batch_data) < current_limit:
                    logger.info(f"Reached earliest available data at {batch_data[0]['datetime']}")
                    break

                # Move end_time to just before the earliest candle we got
                current_end_time = batch_data[0]['timestamp'] - 1

                # Rate limiting
                if batch_num < batches_needed - 1:
                    time.sleep(0.2)  # 200ms between requests for safety

            logger.info(f"Fetched {len(all_data)} total candles for {symbol}")
            if all_data:
                logger.info(f"Date range: {all_data[0]['datetime']} to {all_data[-1]['datetime']}")

            return all_data if all_data else None

        except Exception as e:
            logger.error(f"Error fetching max historical data for {symbol}: {e}", exc_info=True)
            return None

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        limit: int = 1000
    ) -> Dict[str, List[Dict]]:
        """
        Fetch OHLCV data for multiple symbols
        Implements rate limiting to respect Binance API limits

        Args:
            symbols: List of trading pairs
            limit: Number of candlesticks per symbol

        Returns:
            Dictionary mapping symbol to OHLCV data list
        """
        results = {}
        total_symbols = len(symbols)

        logger.info(f"Fetching data for {total_symbols} symbols")

        for idx, symbol in enumerate(symbols, 1):
            logger.info(f"Progress: {idx}/{total_symbols} - Fetching {symbol}")

            data = self.fetch_ohlcv(symbol, limit=limit)
            if data:
                results[symbol] = data

            # Rate limiting: avoid hitting API limits
            # With 1200 req/min limit, we can do ~20 req/sec safely
            if idx < total_symbols:
                time.sleep(0.1)  # 100ms delay between requests

        logger.info(f"Completed fetching {len(results)}/{total_symbols} symbols")
        return results

    def _format_klines(self, raw_klines: List, symbol: str) -> List[Dict]:
        """
        Format raw Binance klines data into structured dictionaries

        Binance klines format:
        [
            [
                1499040000000,      // Open time
                "0.01634790",       // Open
                "0.80000000",       // High
                "0.01575800",       // Low
                "0.01577100",       // Close
                "148976.11427815",  // Volume
                1499644799999,      // Close time
                "2434.19055334",    // Quote asset volume
                308,                // Number of trades
                "1756.87402397",    // Taker buy base asset volume
                "28.46694368",      // Taker buy quote asset volume
                "17928899.62484339" // Ignore
            ]
        ]
        """
        formatted = []

        for kline in raw_klines:
            try:
                formatted_kline = {
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
                }
                formatted.append(formatted_kline)
            except (IndexError, ValueError) as e:
                logger.warning(f"Skipping malformed kline data: {e}")
                continue

        return formatted

    def get_available_symbols(self, quote_asset: str = "USDT") -> List[str]:
        """
        Get list of all available trading symbols for a quote asset

        Args:
            quote_asset: Quote asset to filter by (default: "USDT")

        Returns:
            List of trading pair symbols
        """
        try:
            response = self.session.get(
                f"{self.BASE_URL}/exchangeInfo",
                timeout=10
            )
            response.raise_for_status()

            exchange_info = response.json()
            symbols = []

            for symbol_info in exchange_info.get('symbols', []):
                if (symbol_info.get('quoteAsset') == quote_asset and
                    symbol_info.get('status') == 'TRADING'):
                    symbols.append(symbol_info['symbol'])

            logger.info(f"Found {len(symbols)} active {quote_asset} pairs")
            return sorted(symbols)

        except Exception as e:
            logger.error(f"Failed to fetch available symbols: {e}")
            return []

    def close(self):
        """Clean up resources"""
        self.session.close()
