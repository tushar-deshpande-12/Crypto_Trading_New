"""
Stock Data Fetcher - NSE India Stock Market Data
Mirrors CryptoDataFetcher interface for seamless integration.
Uses nsepython library for data retrieval.
"""

import logging
import numpy as np
import pandas as pd
import ta
from typing import List, Dict, Optional, Callable
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class StockDataFetcher:
    """
    Stock data fetcher for NSE India.
    Mirrors CryptoDataFetcher interface for compatibility.

    Note: NSE only provides daily data for free.
    Market hours: 9:15 AM - 3:30 PM IST (Mon-Fri)
    """

    def __init__(self, interval: str = "1d"):
        """
        Initialize stock data fetcher.

        Args:
            interval: Time interval (1d, 1wk, 1mo supported)
        """
        self.interval = interval
        self._session_initialized = False
        self._last_request_time = 0
        self._min_request_interval = 0.35  # NSE rate limit

    def _rate_limit(self):
        """Enforce rate limiting for NSE API"""
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _ensure_session(self):
        """Ensure nsepython session is initialized"""
        if not self._session_initialized:
            try:
                from nsepython import nse_eq
                nse_eq("RELIANCE")
                self._session_initialized = True
            except Exception as e:
                logger.warning(f"Session initialization warning: {e}")
                self._session_initialized = True

    def fetch_ohlcv(self, symbol: str, limit: int = 500, start_date: str = None, end_date: str = None) -> Optional[List[Dict]]:
        """
        Fetch raw OHLCV data for NSE stocks using yfinance.

        Args:
            symbol: NSE stock symbol (e.g., "RELIANCE", "TCS")
            limit: Number of candles to fetch
            start_date: Start date (YYYY-MM-DD format for yfinance)
            end_date: End date (YYYY-MM-DD format for yfinance)

        Returns:
            List of OHLCV dictionaries or None on error
        """
        symbol = symbol.upper().strip()

        # Calculate dates if not provided
        if not end_date:
            end_dt = datetime.now()
        else:
            try:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            except:
                end_dt = datetime.strptime(end_date, "%d-%m-%Y")

        if not start_date:
            # Calculate start date based on limit and interval
            if self.interval == "1d":
                days_back = limit
            elif self.interval == "1wk":
                days_back = limit * 7
            elif self.interval == "1mo":
                days_back = limit * 30
            else:
                days_back = limit

            start_dt = end_dt - timedelta(days=days_back)
        else:
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            except:
                start_dt = datetime.strptime(start_date, "%d-%m-%Y")

        try:
            import yfinance as yf

            # Add .NS suffix for NSE stocks
            yf_symbol = f"{symbol}.NS"

            logger.info(f"Fetching {self.interval} data for {yf_symbol} from {start_dt.date()} to {end_dt.date()}")

            # Map intervals
            interval_map = {
                "1d": "1d",
                "1wk": "1wk",
                "1mo": "1mo",
            }
            yf_interval = interval_map.get(self.interval, "1d")

            # Fetch data using yfinance
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start_dt, end=end_dt, interval=yf_interval)

            if df is None or len(df) == 0:
                logger.warning(f"No data returned for {symbol}")
                return None

            formatted_data = self.format_yfinance_data(df, symbol)
            logger.info(f"Successfully fetched {len(formatted_data)} candlesticks for {symbol}")
            return formatted_data

        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None

    def format_nse_data(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """
        Format NSE data to match crypto data structure.

        Args:
            df: DataFrame from nsepython
            symbol: Stock symbol

        Returns:
            List of formatted OHLCV dictionaries
        """
        formatted = []

        for _, row in df.iterrows():
            try:
                # Parse timestamp
                timestamp_str = str(row.get('CH_TIMESTAMP', ''))
                try:
                    dt = datetime.strptime(timestamp_str, "%Y-%m-%d")
                except:
                    dt = datetime.now()

                formatted.append({
                    'symbol': symbol,
                    'timestamp': int(dt.timestamp() * 1000),
                    'datetime': dt.isoformat(),
                    'open': float(row.get('CH_OPENING_PRICE', 0)),
                    'high': float(row.get('CH_TRADE_HIGH_PRICE', 0)),
                    'low': float(row.get('CH_TRADE_LOW_PRICE', 0)),
                    'close': float(row.get('CH_CLOSING_PRICE', 0)),
                    'volume': float(row.get('CH_TOT_TRADED_QTY', 0)),
                    'quote_volume': float(row.get('CH_TOT_TRADED_VAL', 0)),
                    'trades': int(row.get('CH_TOTAL_TRADES', 0)),
                    # Stock-specific fields
                    'vwap': float(row.get('VWAP', 0)),
                    'delivery_qty': float(row.get('COP_DELIV_QTY', 0)),
                    'delivery_pct': float(row.get('COP_DELIV_PERC', 0)),
                })
            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"Skipping malformed row: {e}")
                continue

        # Sort by timestamp (oldest first)
        formatted.sort(key=lambda x: x['timestamp'])
        return formatted

    def format_yfinance_data(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """
        Format yfinance data to match crypto data structure.

        Args:
            df: DataFrame from yfinance
            symbol: Stock symbol

        Returns:
            List of formatted OHLCV dictionaries
        """
        formatted = []

        for idx, row in df.iterrows():
            try:
                # idx is the datetime index from yfinance
                dt = idx.to_pydatetime()

                formatted.append({
                    'symbol': symbol,
                    'timestamp': int(dt.timestamp() * 1000),
                    'datetime': dt.isoformat(),
                    'open': float(row.get('Open', 0)),
                    'high': float(row.get('High', 0)),
                    'low': float(row.get('Low', 0)),
                    'close': float(row.get('Close', 0)),
                    'volume': float(row.get('Volume', 0)),
                    'quote_volume': 0,  # Not available from yfinance
                    'trades': 0,  # Not available from yfinance
                })
            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"Skipping malformed row: {e}")
                continue

        # Sort by timestamp (oldest first)
        formatted.sort(key=lambda x: x['timestamp'])
        return formatted

    def fetch_latest_ohlcv(self, symbol: str, days_back: int = 30) -> Optional[List[Dict]]:
        """
        Fetch recent data for a symbol.

        Args:
            symbol: NSE stock symbol
            days_back: Number of days to fetch

        Returns:
            List of OHLCV dictionaries
        """
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=days_back)

        return self.fetch_ohlcv(
            symbol,
            limit=days_back,
            start_date=start_dt.strftime("%d-%m-%Y"),
            end_date=end_dt.strftime("%d-%m-%Y")
        )

    def fetch_max_historical_data(self, symbol: str, max_candles: int = 2000,
                                   progress_callback: Optional[Callable] = None) -> Optional[List[Dict]]:
        """
        Fetch maximum historical data + compute ML features.
        NSE provides up to ~2 years of daily data.

        Args:
            symbol: NSE stock symbol
            max_candles: Maximum candles to fetch (days for daily data)
            progress_callback: Optional callback(current, total, message)

        Returns:
            List of feature-enriched OHLCV dictionaries
        """
        if progress_callback:
            progress_callback(0, max_candles, f"Fetching {symbol}...")

        # NSE provides daily data, calculate date range
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=max_candles)

        all_data = self.fetch_ohlcv(
            symbol,
            limit=max_candles,
            start_date=start_dt.strftime("%d-%m-%Y"),
            end_date=end_dt.strftime("%d-%m-%Y")
        )

        if not all_data:
            logger.error(f"No data fetched for {symbol}")
            return None

        if progress_callback:
            progress_callback(len(all_data), max_candles, f"Fetched {len(all_data)} candles")

        if len(all_data) < 50:
            logger.warning(f"Insufficient data for {symbol}: {len(all_data)} candles")
            return all_data

        if progress_callback:
            progress_callback(len(all_data), max_candles, "Computing features...")

        logger.info(f"Computing features for {len(all_data)} candles of {symbol}")
        return self.compute_features(all_data)

    def compute_features(self, ohlcv_data: List[Dict]) -> List[Dict]:
        """
        Engineer features + directional labels for trading models.
        Same features as CryptoDataFetcher for consistency.

        Args:
            ohlcv_data: List of OHLCV dictionaries

        Returns:
            List of feature-enriched dictionaries
        """
        df = pd.DataFrame(ohlcv_data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

        # Price returns (adjusted for daily data)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['return_20d'] = df['close'].pct_change(20)

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

        # Directional label (for daily data, use smaller threshold)
        threshold = 0.005  # 0.5% for stocks
        df['direction'] = ((df['close'].shift(-1) / df['close'] - 1) > threshold).astype(int)
        df['direction'] = df['direction'].fillna(0)

        # Price position ratios
        df['high_low_ratio'] = df['close'] / df['low']
        df['low_high_ratio'] = df['close'] / df['high']

        # Stock-specific: delivery percentage feature (if available)
        if 'delivery_pct' in df.columns:
            df['delivery_sma'] = df['delivery_pct'].rolling(20).mean()
            df['delivery_ratio'] = df['delivery_pct'] / df['delivery_sma'].replace(0, 1)

        df = df.dropna()
        result = df.reset_index().to_dict('records')

        logger.info(f"Features computed: {len(result)} rows")
        return result

    def get_available_symbols(self) -> List[str]:
        """
        Get available NSE stock symbols.

        Returns:
            List of available symbol strings
        """
        from src.gui_pyqt.utils.stock_symbols import get_all_stock_symbols
        return get_all_stock_symbols()

    def close(self):
        """Clean up resources."""
        pass  # nsepython doesn't require explicit cleanup
