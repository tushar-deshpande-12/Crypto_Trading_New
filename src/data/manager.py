# manager.py - COMPLETE REWRITE WITH FEATURE SUPPORT
import logging
from typing import List, Dict, Optional, Callable
from pathlib import Path
from datetime import datetime
from .fetcher import CryptoDataFetcher
from .storage import DataStorage

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, storage_dir: str = "dataset", interval: str = "1h", autocreate_dir: bool = True):
        self.fetcher = CryptoDataFetcher(interval=interval)
        self.storage = DataStorage(base_dir=storage_dir)
        self.interval = interval
        if autocreate_dir:
            Path(storage_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"DataManager initialized: {interval=}, {storage_dir=}")
    
    def fetch_and_save(self, symbol: str, max_candles: int = 10000, progress_callback: Optional[Callable] = None, 
                      metadata: Optional[Dict] = None) -> Optional[Path]:
        """Fetch, compute features, and save in one call."""
        symbol = symbol.upper().strip()
        if not symbol.endswith("USDT"):
            symbol += "USDT"
        
        def progress(current: int, total: int, message: str = ""):
            if progress_callback:
                progress_callback(current, total, message)
        
        progress(0, max_candles, f"Fetching {symbol}...")
        
        # Fetch with features
        data = self.fetcher.fetch_max_historical_data(symbol, max_candles, progress)
        if not data:
            logger.error(f"No data for {symbol}")
            return None
        
        progress(len(data), max_candles, f"Saving {len(data)} feature-enriched candles...")
        
        # Enhanced metadata
        full_metadata = {
            'symbol': symbol,
            'interval': self.interval,
            'max_candles_requested': max_candles,
            'actual_candles_fetched': len(data),
            'fetch_timestamp': datetime.now().isoformat(),
            'features_computed': True,
            'target_column': 'direction',  # Binary up/down
            'features': ['rsi', 'macd', 'volatility_20', 'bb_position', 'return_1h', 'volume_ratio']  # Key ones
        }
        if metadata:
            full_metadata.update(metadata)
        
        # Save
        dataset_path = self.storage.save_dataset(symbol, data, metadata=full_metadata)
        logger.info(f"Saved features to {dataset_path}")
        return dataset_path
    
    def fetch_and_save_multi_timeframe(self, symbol: str, timeframes: List[str] = None,
                                       max_candles: int = 50000,
                                       progress_callback: Optional[Callable] = None) -> Dict[str, Optional[Path]]:
        """Fetch data for multiple timeframes and save each separately.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            timeframes: List of timeframes to fetch (default: 5m, 15m, 30m, 1h, 4h)
            max_candles: Number of candles per timeframe (default: 50000)
            progress_callback: Callback(current, total, message)

        Returns:
            Dict mapping timeframe -> saved dataset path (or None on failure)
        """
        from src.core.config import AppConfig
        if timeframes is None:
            timeframes = list(AppConfig.MULTI_TIMEFRAMES)

        results = {}
        total_timeframes = len(timeframes)

        for tf_idx, tf in enumerate(timeframes):
            if progress_callback:
                overall_pct = int((tf_idx / total_timeframes) * 100)
                progress_callback(overall_pct, 100, f"[{tf_idx+1}/{total_timeframes}] Fetching {symbol} @ {tf}...")

            # Switch fetcher interval for this timeframe
            self.fetcher.interval = tf
            original_interval = self.interval
            self.interval = tf

            try:
                path = self.fetch_and_save(
                    symbol=symbol,
                    max_candles=max_candles,
                    progress_callback=None,  # Use our own progress above
                    metadata={'interval': tf, 'timeframe': tf}
                )
                results[tf] = path
                if path:
                    logger.info(f"Saved {tf} data for {symbol} -> {path}")
                else:
                    logger.warning(f"Failed to fetch {tf} data for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching {tf} for {symbol}: {e}")
                results[tf] = None
            finally:
                self.interval = original_interval
                self.fetcher.interval = original_interval

        if progress_callback:
            progress_callback(100, 100, f"Completed all {total_timeframes} timeframes for {symbol}")

        return results

    def fetch_and_save_multiple(self, symbols: List[str], max_candles: int = 10000,
                               progress_callback: Optional[Callable] = None) -> Dict[str, Optional[Path]]:
        """Batch fetch multiple symbols."""
        results = {}
        total_symbols = len(symbols)
        for idx, symbol in enumerate(symbols, 1):
            def symbol_progress(current, total, message):
                if progress_callback:
                    progress_callback(symbol, idx, total_symbols, message)
            
            results[symbol] = self.fetch_and_save(symbol, max_candles, symbol_progress)
            time.sleep(0.1)  # Rate limit
        return results
    
    def load_latest(self, symbol: str, format_: str = "json") -> Optional[List[Dict]]:
        """Load latest dataset (now with features)."""
        dataset_path = self.storage.get_latest_dataset(symbol)
        if not dataset_path:
            return None
        data = self.storage.load_dataset(dataset_path, format_)
        logger.info(f"Loaded {len(data)} feature rows for {symbol}")
        return data
    
    def check_data_freshness(self, symbol: str, time_threshold_hours: int = 24) -> Dict:
        """Enhanced freshness check."""
        should_fetch = self.storage.should_fetch_new_data(symbol, time_threshold_hours)
        latest = self.storage.get_latest_dataset(symbol)
        result = {
            'symbol': symbol,
            'has_data': latest is not None,
            'should_fetch_new': should_fetch,
            'latest_dataset': str(latest) if latest else None
        }
        if latest:
            # Add feature info from metadata
            metadata_path = latest / "metadata.json"
            try:
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                result.update({
                    'candle_count': metadata.get('actual_candles_fetched'),
                    'features_computed': metadata.get('features_computed', False),
                    'fetch_timestamp': metadata.get('fetch_timestamp')
                })
            except:
                pass
        return result
    
    # Passthrough methods
    def get_available_symbols(self) -> List[str]:
        return self.fetcher.get_available_symbols()
    
    def list_all_datasets(self, symbol: Optional[str] = None) -> List[Dict]:
        return self.storage.list_datasets(symbol)
    
    def close(self):
        self.fetcher.close()
