"""
NSE India API Client Module
Handles all interactions with NSE India for stock market data
Uses nsepython library for data retrieval
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class NSEAPIClient:
    """
    NSE India API client for stock market data retrieval.
    Uses nsepython library - no authentication required.

    Note: NSE rate limits requests to ~3 per second.
    """

    def __init__(self):
        self._session_initialized = False
        self._last_request_time = 0
        self._min_request_interval = 0.35  # ~3 requests per second

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
                # Make a test call to initialize session
                nse_eq("RELIANCE")
                self._session_initialized = True
            except Exception as e:
                logger.warning(f"Session initialization warning: {e}")
                self._session_initialized = True  # Continue anyway

    def get_stock_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time quote for a single stock.

        Args:
            symbol: NSE stock symbol (e.g., "RELIANCE", "TCS")

        Returns:
            Dictionary with stock quote data or None on error
        """
        try:
            self._rate_limit()
            self._ensure_session()

            from nsepython import nse_eq
            data = nse_eq(symbol.upper())

            if data and 'priceInfo' in data:
                price_info = data['priceInfo']
                return {
                    'symbol': symbol.upper(),
                    'price': float(price_info.get('lastPrice', 0)),
                    'price_change': float(price_info.get('change', 0)),
                    'price_change_pct': float(price_info.get('pChange', 0)),
                    'open': float(price_info.get('open', 0)),
                    'high': float(price_info.get('intraDayHighLow', {}).get('max', 0)),
                    'low': float(price_info.get('intraDayHighLow', {}).get('min', 0)),
                    'close': float(price_info.get('close', 0)),
                    'previous_close': float(price_info.get('previousClose', 0)),
                    'volume': int(data.get('securityWiseDP', {}).get('quantityTraded', 0)),
                }
            return None
        except Exception as e:
            logger.error(f"Failed to fetch quote for {symbol}: {e}")
            return None

    def get_24h_ticker(self, symbols: List[str]) -> List[Dict]:
        """
        Get 24h ticker data for multiple symbols.

        Args:
            symbols: List of NSE stock symbols

        Returns:
            List of dictionaries with ticker data
        """
        results = []
        for symbol in symbols:
            quote = self.get_stock_quote(symbol)
            if quote:
                results.append(quote)
        return results

    def get_all_stocks_quote(self) -> List[Dict]:
        """
        Get quote data for all Nifty stocks.
        Uses pre-built market status endpoint for efficiency.

        Returns:
            List of dictionaries with stock data
        """
        try:
            self._rate_limit()
            self._ensure_session()

            from nsepython import nse_get_advances_declines

            # Get market-wide data
            data = nse_get_advances_declines()

            if data is not None and len(data) > 0:
                results = []
                for _, row in data.iterrows():
                    results.append({
                        'symbol': row.get('symbol', ''),
                        'price': float(row.get('ltp', 0)),
                        'price_change': float(row.get('netChange', 0)),
                        'price_change_pct': float(row.get('perChange', 0)),
                        'high': float(row.get('dayHigh', 0)),
                        'low': float(row.get('dayLow', 0)),
                        'open': float(row.get('open', 0)),
                        'previous_close': float(row.get('previousClose', 0)),
                        'volume': 0,  # Not available in this endpoint
                        'quote_volume_24h': 0,
                    })
                return results
            return []
        except Exception as e:
            logger.error(f"Failed to fetch all stocks: {e}")
            return []

    def get_index_stocks(self, index: str = "NIFTY") -> List[Dict]:
        """
        Get all stocks in an index with their current data.

        Args:
            index: Index name (e.g., "NIFTY", "NIFTYBANK", "NIFTYNEXT50")

        Returns:
            List of stock data dictionaries
        """
        try:
            self._rate_limit()
            self._ensure_session()

            from nsepython import nse_preopen

            # Map common index names to preopen index names
            index_map = {
                "NIFTY 50": "NIFTY",
                "NIFTY50": "NIFTY",
                "NIFTY BANK": "BANKNIFTY",
                "NIFTYBANK": "BANKNIFTY",
                "NIFTY NEXT 50": "NIFTYNEXT50",
            }
            preopen_index = index_map.get(index.upper(), index.upper())

            df = nse_preopen(preopen_index)

            if df is not None and len(df) > 0:
                results = []
                for _, row in df.iterrows():
                    results.append({
                        'symbol': row.get('symbol', ''),
                        'price': float(row.get('lastPrice', 0)),
                        'price_change': float(row.get('change', 0)),
                        'price_change_pct': float(row.get('pChange', 0)),
                        'open': float(row.get('lastPrice', 0)),  # preopen uses lastPrice
                        'high': float(row.get('yearHigh', 0)),
                        'low': float(row.get('yearLow', 0)),
                        'previous_close': float(row.get('previousClose', 0)),
                        'volume': int(row.get('finalQuantity', 0)),
                        'quote_volume_24h': float(row.get('totalTurnover', 0)),
                    })
                return results
            return []
        except Exception as e:
            logger.error(f"Failed to fetch index {index}: {e}")
            return []

    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Optional[List[Dict]]:
        """
        Get historical OHLCV data for a stock.

        Args:
            symbol: NSE stock symbol
            start_date: Start date in DD-MM-YYYY format
            end_date: End date in DD-MM-YYYY format

        Returns:
            List of OHLCV dictionaries or None on error
        """
        try:
            self._rate_limit()
            self._ensure_session()

            from nsepython import equity_history

            df = equity_history(symbol.upper(), "EQ", start_date, end_date)

            if df is not None and len(df) > 0:
                results = []
                for _, row in df.iterrows():
                    results.append({
                        'timestamp': row.get('CH_TIMESTAMP', ''),
                        'open': float(row.get('CH_OPENING_PRICE', 0)),
                        'high': float(row.get('CH_TRADE_HIGH_PRICE', 0)),
                        'low': float(row.get('CH_TRADE_LOW_PRICE', 0)),
                        'close': float(row.get('CH_CLOSING_PRICE', 0)),
                        'volume': int(row.get('CH_TOT_TRADED_QTY', 0)),
                        'value': float(row.get('CH_TOT_TRADED_VAL', 0)),
                    })
                return results
            return None
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return None

    def get_klines(self, symbol: str, interval: str = "1d", limit: int = 500) -> Optional[List[Dict]]:
        """
        Get candlestick/kline data (mimics Binance API interface).

        Args:
            symbol: NSE stock symbol
            interval: Time interval (1d, 1wk, 1mo supported)
            limit: Number of candles to fetch

        Returns:
            List of OHLCV dictionaries
        """
        try:
            # Calculate date range based on limit and interval
            end_date = datetime.now()

            if interval == "1d":
                start_date = end_date - timedelta(days=limit)
            elif interval == "1wk":
                start_date = end_date - timedelta(weeks=limit)
            elif interval == "1mo":
                start_date = end_date - timedelta(days=limit * 30)
            else:
                start_date = end_date - timedelta(days=limit)

            start_str = start_date.strftime("%d-%m-%Y")
            end_str = end_date.strftime("%d-%m-%Y")

            return self.get_historical_data(symbol, start_str, end_str)
        except Exception as e:
            logger.error(f"Failed to fetch klines for {symbol}: {e}")
            return None

    def get_inr_pairs_detailed(self) -> List[Dict]:
        """
        Get detailed data for all Indian stocks (mirrors get_usdt_pairs_detailed).
        Fetches stocks using nse_get_advances_declines for comprehensive data.

        Returns:
            List of dictionaries with formatted market data
        """
        try:
            self._rate_limit()
            self._ensure_session()

            from nsepython import nse_get_advances_declines

            df = nse_get_advances_declines()

            if df is None or len(df) == 0:
                # Fallback to index stocks
                logger.warning("nse_get_advances_declines returned no data, falling back to index stocks")
                return self._get_index_stocks_fallback()

            results = []
            for _, row in df.iterrows():
                results.append({
                    'symbol': row.get('symbol', ''),
                    'base_asset': row.get('symbol', ''),
                    'price': float(row.get('lastPrice', 0)),
                    'price_change': float(row.get('change', 0)),
                    'price_change_pct': float(row.get('pChange', 0)),
                    'open': float(row.get('open', 0)),
                    'high': float(row.get('dayHigh', 0)),
                    'low': float(row.get('dayLow', 0)),
                    'previous_close': float(row.get('previousClose', 0)),
                    'volume': int(row.get('totalTradedVolume', 0)),
                    'quote_volume_24h': float(row.get('totalTradedValue', 0)),
                    'high_52w': float(row.get('yearHigh', 0)),
                    'low_52w': float(row.get('yearLow', 0)),
                })

            logger.info(f"Loaded {len(results)} NSE stocks")
            return results

        except Exception as e:
            logger.error(f"Failed to fetch INR pairs: {e}")
            return self._get_index_stocks_fallback()

    def _get_index_stocks_fallback(self) -> List[Dict]:
        """Fallback method using index stocks"""
        all_stocks = []

        # Fetch Nifty 50
        nifty50 = self.get_index_stocks("NIFTY")
        all_stocks.extend(nifty50)

        # Deduplicate by symbol
        seen = set()
        unique_stocks = []
        for stock in all_stocks:
            if stock['symbol'] not in seen:
                seen.add(stock['symbol'])
                stock['base_asset'] = stock['symbol']
                unique_stocks.append(stock)

        return unique_stocks

    def get_market_status(self) -> Dict:
        """
        Get current market status (open/closed).

        Returns:
            Dictionary with market status info
        """
        try:
            self._rate_limit()
            from nsepython import nse_marketStatus

            status = nse_marketStatus()
            return {
                'status': status.get('marketState', {}).get('marketStatus', 'Unknown'),
                'message': status.get('marketState', {}).get('tradeDate', ''),
            }
        except Exception as e:
            logger.error(f"Failed to fetch market status: {e}")
            return {'status': 'Unknown', 'message': str(e)}

    def close(self):
        """Clean up resources"""
        pass  # nsepython doesn't require explicit cleanup
