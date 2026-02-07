"""
Data Storage Module
Manages folder-based storage of cryptocurrency OHLCV data
Organizes data by cryptocurrency and timestamp to enable time-series tracking
"""

import json
import csv
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import hashlib
import numpy as np

logger = logging.getLogger(__name__)


class DataStorage:
    """
    Professional data storage manager with folder-based organization

    Directory Structure:
    dataset/
    ├── BTCUSDT/
    │   ├── 2026-01-10_14-30-00_10000candles/
    │   │   ├── metadata.json
    │   │   ├── data.csv
    │   │   └── data.json
    │   └── 2026-01-09_08-15-00_5000candles/
    │       ├── metadata.json
    │       ├── data.csv
    │       └── data.json
    └── ETHUSDT/
        └── ...
    """

    def __init__(self, base_dir: str = "dataset"):
        """
        Initialize data storage manager

        Args:
            base_dir: Base directory for all datasets (default: "dataset")
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Data storage initialized at: {self.base_dir.absolute()}")

    def save_dataset(
        self,
        symbol: str,
        data: List[Dict],
        metadata: Optional[Dict] = None,
        save_csv: bool = True,
        save_json: bool = True
    ) -> Path:
        """
        Save cryptocurrency dataset with automatic folder organization

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            data: List of OHLCV dictionaries
            metadata: Optional metadata about the dataset
            save_csv: Whether to save CSV format (default: True)
            save_json: Whether to save JSON format (default: True)

        Returns:
            Path to the created dataset directory
        """
        try:
            if not data:
                raise ValueError("Cannot save empty dataset")

            # Create symbol directory
            symbol_dir = self.base_dir / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)

            # Generate dataset directory name with timestamp, interval and candle count
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            candle_count = len(data)
            interval = metadata.get('interval', '1h') if metadata else data[0].get('interval', '1h') if data else '1h'
            dataset_name = f"{timestamp}_{interval}_{candle_count}candles"
            dataset_dir = symbol_dir / dataset_name

            # Check if dataset already exists (shouldn't happen with timestamp, but be safe)
            if dataset_dir.exists():
                logger.warning(f"Dataset directory already exists: {dataset_dir}")
                # Add a unique suffix
                suffix = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:6]
                dataset_name = f"{dataset_name}_{suffix}"
                dataset_dir = symbol_dir / dataset_name

            dataset_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Creating dataset at: {dataset_dir}")

            # Prepare metadata
            full_metadata = {
                'symbol': symbol,
                'candle_count': candle_count,
                'interval': data[0].get('interval', '1h') if data else '1h',
                'fetch_timestamp': datetime.now().isoformat(),
                'date_range': {
                    'start': data[0]['datetime'] if data else None,
                    'end': data[-1]['datetime'] if data else None,
                    'start_timestamp': data[0]['timestamp'] if data else None,
                    'end_timestamp': data[-1]['timestamp'] if data else None
                },
                'data_points': candle_count,
                'files': []
            }

            # Add user-provided metadata
            if metadata:
                full_metadata['user_metadata'] = metadata

            # Save metadata
            metadata_path = dataset_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(full_metadata, f, indent=2, default=self._json_serializer)
            full_metadata['files'].append('metadata.json')
            logger.info(f"Saved metadata to: {metadata_path}")

            # Save CSV format
            if save_csv:
                csv_path = dataset_dir / "data.csv"
                self._save_csv(csv_path, data)
                full_metadata['files'].append('data.csv')
                logger.info(f"Saved CSV to: {csv_path}")

            # Save JSON format
            if save_json:
                json_path = dataset_dir / "data.json"
                self._save_json(json_path, data)
                full_metadata['files'].append('data.json')
                logger.info(f"Saved JSON to: {json_path}")

            # Update metadata with file list
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(full_metadata, f, indent=2, default=self._json_serializer)

            logger.info(f"Dataset saved successfully: {dataset_dir}")
            return dataset_dir

        except Exception as e:
            logger.error(f"Failed to save dataset for {symbol}: {e}", exc_info=True)
            raise

    def load_dataset(
        self,
        dataset_path: Path,
        format: str = 'json'
    ) -> Optional[List[Dict]]:
        """
        Load a dataset from disk

        Args:
            dataset_path: Path to dataset directory
            format: Data format to load ('json' or 'csv', default: 'json')

        Returns:
            List of OHLCV dictionaries or None on error
        """
        try:
            dataset_path = Path(dataset_path)

            if not dataset_path.exists():
                logger.error(f"Dataset path does not exist: {dataset_path}")
                return None

            if format == 'json':
                data_file = dataset_path / "data.json"
                if not data_file.exists():
                    logger.error(f"JSON data file not found: {data_file}")
                    return None
                return self._load_json(data_file)

            elif format == 'csv':
                data_file = dataset_path / "data.csv"
                if not data_file.exists():
                    logger.error(f"CSV data file not found: {data_file}")
                    return None
                return self._load_csv(data_file)

            else:
                logger.error(f"Unsupported format: {format}")
                return None

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}", exc_info=True)
            return None

    def list_datasets(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        List all available datasets

        Args:
            symbol: Optional symbol to filter by (e.g., "BTCUSDT")

        Returns:
            List of dataset information dictionaries
        """
        datasets = []

        try:
            # Get symbols to search
            if symbol:
                symbol_dirs = [self.base_dir / symbol]
            else:
                symbol_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]

            # Scan each symbol directory
            for symbol_dir in symbol_dirs:
                if not symbol_dir.is_dir():
                    continue

                symbol_name = symbol_dir.name

                # Find all dataset directories
                for dataset_dir in symbol_dir.iterdir():
                    if not dataset_dir.is_dir():
                        continue

                    # Load metadata
                    metadata_path = dataset_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            metadata['path'] = str(dataset_dir)
                            datasets.append(metadata)
                    else:
                        # Basic info if no metadata
                        datasets.append({
                            'symbol': symbol_name,
                            'path': str(dataset_dir),
                            'name': dataset_dir.name
                        })

            # Sort by fetch timestamp (newest first)
            datasets.sort(
                key=lambda x: x.get('fetch_timestamp', ''),
                reverse=True
            )

            logger.info(f"Found {len(datasets)} datasets")
            return datasets

        except Exception as e:
            logger.error(f"Failed to list datasets: {e}", exc_info=True)
            return []

    def get_latest_dataset(self, symbol: str) -> Optional[Path]:
        """
        Get the most recent dataset for a symbol

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")

        Returns:
            Path to latest dataset directory or None
        """
        datasets = self.list_datasets(symbol)
        if datasets:
            return Path(datasets[0]['path'])
        return None

    def should_fetch_new_data(
        self,
        symbol: str,
        time_threshold_hours: int = 24
    ) -> bool:
        """
        Determine if new data should be fetched based on time difference

        Args:
            symbol: Trading pair
            time_threshold_hours: Hours threshold for considering data "old"

        Returns:
            True if new data should be fetched, False otherwise
        """
        latest = self.get_latest_dataset(symbol)

        if not latest:
            logger.info(f"No existing data for {symbol}, should fetch new data")
            return True

        # Load metadata
        metadata_path = latest / "metadata.json"
        if not metadata_path.exists():
            logger.warning(f"No metadata found for latest dataset, should fetch new data")
            return True

        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            fetch_time = datetime.fromisoformat(metadata['fetch_timestamp'])
            time_diff = datetime.now() - fetch_time
            hours_diff = time_diff.total_seconds() / 3600

            logger.info(f"Latest data for {symbol} is {hours_diff:.1f} hours old")

            return hours_diff >= time_threshold_hours

        except Exception as e:
            logger.error(f"Error checking data age: {e}")
            return True

    def _save_csv(self, path: Path, data: List[Dict]):
        """Save data in CSV format"""
        if not data:
            return

        # Get all unique keys from all records
        fieldnames = set()
        for record in data:
            fieldnames.update(record.keys())
        fieldnames = sorted(fieldnames)

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    def _save_json(self, path: Path, data: List[Dict]):
        """Save data in JSON format"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=self._json_serializer)

    def _json_serializer(self, obj):
        """Custom JSON serializer for objects not serializable by default"""
        import pandas as pd
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, 'item'):  # numpy scalar types
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def _load_json(self, path: Path) -> List[Dict]:
        """Load data from JSON format"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_csv(self, path: Path) -> List[Dict]:
        """Load data from CSV format"""
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)

    def get_storage_stats(self) -> Dict:
        """
        Get statistics about stored datasets

        Returns:
            Dictionary with storage statistics
        """
        stats = {
            'total_datasets': 0,
            'total_symbols': 0,
            'total_size_mb': 0,
            'symbols': {}
        }

        try:
            # Count symbols
            symbol_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
            stats['total_symbols'] = len(symbol_dirs)

            # Count datasets and size
            for symbol_dir in symbol_dirs:
                symbol_name = symbol_dir.name
                datasets = list(symbol_dir.iterdir())
                dataset_count = len([d for d in datasets if d.is_dir()])

                stats['symbols'][symbol_name] = dataset_count
                stats['total_datasets'] += dataset_count

            # Calculate total size
            total_size = sum(
                f.stat().st_size
                for f in self.base_dir.rglob('*')
                if f.is_file()
            )
            stats['total_size_mb'] = round(total_size / (1024 * 1024), 2)

            return stats

        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return stats
