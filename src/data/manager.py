"""
Data Manager
High-level coordinator for cryptocurrency data operations
Orchestrates fetching, storage, and retrieval of OHLCV data
"""

import logging
from typing import List, Dict, Optional, Callable
from pathlib import Path
from datetime import datetime

from .fetcher import CryptoDataFetcher
from .storage import DataStorage

logger = logging.getLogger(__name__)


class DataManager:
    """
    Professional data management coordinator
    Provides high-level interface for all data operations
    """

    def __init__(
        self,
        storage_dir: str = "dataset",
        interval: str = "1h",
        auto_create_dir: bool = True
    ):
        """
        Initialize data manager

        Args:
            storage_dir: Base directory for datasets (default: "dataset")
            interval: Candlestick interval (default: "1h")
            auto_create_dir: Auto-create storage directory (default: True)
        """
        self.fetcher = CryptoDataFetcher(interval=interval)
        self.storage = DataStorage(base_dir=storage_dir)
        self.interval = interval

        if auto_create_dir:
            Path(storage_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"DataManager initialized (interval={interval}, storage={storage_dir})")

    def fetch_and_save(
        self,
        symbol: str,
        max_candles: int = 10000,
        progress_callback: Optional[Callable] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[Path]:
        """
        Fetch maximum historical data and save to disk
        One-stop operation for data acquisition and storage

        Args:
            symbol: Trading pair (e.g., "BTCUSDT" or "BTC")
            max_candles: Maximum candles to fetch (default: 10000 ~= 13.7 months for 1h)
            progress_callback: Optional callback(current, total, message) for progress updates
            metadata: Optional metadata to store with dataset

        Returns:
            Path to saved dataset directory or None on error
        """
        try:
            # Normalize symbol
            symbol = symbol.upper().strip()
            if not symbol.endswith('USDT'):
                symbol = f"{symbol}USDT"

            logger.info(f"Starting fetch and save operation for {symbol}")

            if progress_callback:
                progress_callback(0, max_candles, f"Initializing fetch for {symbol}...")

            # Fetch data
            def fetch_progress(current, total):
                if progress_callback:
                    progress_callback(
                        current,
                        total,
                        f"Fetching {symbol}: {current}/{total} candles"
                    )

            data = self.fetcher.fetch_max_historical_data(
                symbol=symbol,
                max_candles=max_candles,
                progress_callback=fetch_progress
            )

            if not data:
                logger.error(f"No data fetched for {symbol}")
                if progress_callback:
                    progress_callback(0, max_candles, f"Failed to fetch data for {symbol}")
                return None

            if progress_callback:
                progress_callback(
                    len(data),
                    max_candles,
                    f"Saving {len(data)} candles to disk..."
                )

            # Prepare metadata
            full_metadata = {
                'interval': self.interval,
                'max_candles_requested': max_candles,
                'actual_candles_fetched': len(data),
                'fetch_method': 'max_historical'
            }

            if metadata:
                full_metadata.update(metadata)

            # Save dataset
            dataset_path = self.storage.save_dataset(
                symbol=symbol,
                data=data,
                metadata=full_metadata,
                save_csv=True,
                save_json=True
            )

            logger.info(f"Successfully saved {len(data)} candles to {dataset_path}")

            if progress_callback:
                progress_callback(
                    len(data),
                    max_candles,
                    f"Completed! Saved {len(data)} candles"
                )

            return dataset_path

        except Exception as e:
            logger.error(f"Error in fetch_and_save for {symbol}: {e}", exc_info=True)
            if progress_callback:
                progress_callback(0, max_candles, f"Error: {str(e)}")
            return None

    def fetch_and_save_multiple(
        self,
        symbols: List[str],
        max_candles: int = 10000,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Optional[Path]]:
        """
        Fetch and save data for multiple symbols

        Args:
            symbols: List of trading pairs
            max_candles: Maximum candles per symbol
            progress_callback: Optional callback(symbol, current, total, message)

        Returns:
            Dictionary mapping symbol to dataset path (or None on failure)
        """
        results = {}
        total_symbols = len(symbols)

        logger.info(f"Fetching data for {total_symbols} symbols")

        for idx, symbol in enumerate(symbols, 1):
            logger.info(f"Processing {idx}/{total_symbols}: {symbol}")

            def symbol_progress(current, total, message):
                if progress_callback:
                    progress_callback(symbol, idx, total_symbols, message)

            dataset_path = self.fetch_and_save(
                symbol=symbol,
                max_candles=max_candles,
                progress_callback=symbol_progress
            )

            results[symbol] = dataset_path

        logger.info(f"Completed fetching {len(results)} symbols")
        return results

    def load_latest(
        self,
        symbol: str,
        format: str = 'json'
    ) -> Optional[List[Dict]]:
        """
        Load the most recent dataset for a symbol

        Args:
            symbol: Trading pair
            format: Data format ('json' or 'csv')

        Returns:
            List of OHLCV dictionaries or None
        """
        dataset_path = self.storage.get_latest_dataset(symbol)
        if not dataset_path:
            logger.warning(f"No dataset found for {symbol}")
            return None

        return self.storage.load_dataset(dataset_path, format=format)

    def load_dataset_by_path(
        self,
        dataset_path: str,
        format: str = 'json'
    ) -> Optional[List[Dict]]:
        """
        Load a specific dataset by path

        Args:
            dataset_path: Path to dataset directory
            format: Data format ('json' or 'csv')

        Returns:
            List of OHLCV dictionaries or None
        """
        return self.storage.load_dataset(Path(dataset_path), format=format)

    def list_all_datasets(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        List all available datasets

        Args:
            symbol: Optional symbol filter

        Returns:
            List of dataset information dictionaries
        """
        return self.storage.list_datasets(symbol=symbol)

    def check_data_freshness(
        self,
        symbol: str,
        time_threshold_hours: int = 24
    ) -> Dict:
        """
        Check if data exists and how fresh it is

        Args:
            symbol: Trading pair
            time_threshold_hours: Hours threshold for "fresh" data

        Returns:
            Dictionary with freshness information
        """
        should_fetch = self.storage.should_fetch_new_data(
            symbol=symbol,
            time_threshold_hours=time_threshold_hours
        )

        latest = self.storage.get_latest_dataset(symbol)

        result = {
            'symbol': symbol,
            'has_data': latest is not None,
            'should_fetch_new': should_fetch,
            'latest_dataset': str(latest) if latest else None
        }

        if latest:
            metadata_path = latest / "metadata.json"
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    result['fetch_timestamp'] = metadata.get('fetch_timestamp')
                    result['candle_count'] = metadata.get('candle_count')
                    result['date_range'] = metadata.get('date_range')

        return result

    def get_available_symbols(self) -> List[str]:
        """
        Get list of all available USDT trading pairs from Binance

        Returns:
            List of symbol strings
        """
        return self.fetcher.get_available_symbols(quote_asset="USDT")

    def get_storage_statistics(self) -> Dict:
        """
        Get comprehensive storage statistics

        Returns:
            Dictionary with storage stats
        """
        return self.storage.get_storage_stats()

    def cleanup_old_datasets(
        self,
        symbol: str,
        keep_latest: int = 3
    ) -> int:
        """
        Remove old datasets, keeping only the N most recent ones

        Args:
            symbol: Trading pair
            keep_latest: Number of recent datasets to keep

        Returns:
            Number of datasets deleted
        """
        datasets = self.storage.list_datasets(symbol=symbol)

        if len(datasets) <= keep_latest:
            logger.info(f"No cleanup needed for {symbol} (only {len(datasets)} datasets)")
            return 0

        # Sort by timestamp (newest first)
        datasets.sort(key=lambda x: x.get('fetch_timestamp', ''), reverse=True)

        # Delete old datasets
        deleted = 0
        for dataset in datasets[keep_latest:]:
            try:
                dataset_path = Path(dataset['path'])
                if dataset_path.exists():
                    import shutil
                    shutil.rmtree(dataset_path)
                    deleted += 1
                    logger.info(f"Deleted old dataset: {dataset_path}")
            except Exception as e:
                logger.error(f"Failed to delete {dataset['path']}: {e}")

        logger.info(f"Cleaned up {deleted} old datasets for {symbol}")
        return deleted

    def close(self):
        """Clean up resources"""
        if self.fetcher:
            self.fetcher.close()
        logger.info("DataManager closed")
