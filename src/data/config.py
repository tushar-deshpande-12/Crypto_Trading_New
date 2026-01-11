"""
Data Pipeline Configuration
Centralized configuration for cryptocurrency data operations
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DataPipelineConfig:
    """
    Configuration parameters for data pipeline operations
    """

    # Storage Configuration
    storage_base_dir: str = "dataset"
    auto_create_directories: bool = True
    save_csv_format: bool = True
    save_json_format: bool = True

    # Fetching Configuration
    default_interval: str = "1h"  # Candlestick interval
    default_max_candles: int = 10000  # ~13.7 months for 1h interval
    quote_asset: str = "USDT"  # Primary quote asset

    # Rate Limiting
    request_delay_ms: int = 200  # Delay between API requests (milliseconds)
    max_concurrent_fetches: int = 1  # Number of concurrent symbol fetches

    # Data Freshness
    data_freshness_threshold_hours: int = 24  # Consider data "old" after this many hours
    auto_fetch_if_old: bool = False  # Automatically fetch new data if old

    # Dataset Management
    keep_latest_datasets: int = 5  # Number of recent datasets to keep per symbol
    auto_cleanup_old_datasets: bool = False  # Automatically remove old datasets

    # API Configuration
    binance_base_url: str = "https://api.binance.com/api/v3"
    api_timeout_seconds: int = 10
    max_retries: int = 3

    # Data Quality
    validate_data_integrity: bool = True  # Validate data after fetching
    remove_duplicates: bool = True  # Remove duplicate timestamps
    fill_missing_candles: bool = False  # Fill gaps in time series

    # Logging
    log_level: str = "INFO"
    log_api_requests: bool = True
    log_storage_operations: bool = True

    # Performance
    batch_size: int = 1000  # API request batch size (Binance limit)
    use_compression: bool = False  # Compress stored data (future feature)

    # GUI Configuration
    show_progress_updates: bool = True
    progress_update_interval_ms: int = 500

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'storage': {
                'base_dir': self.storage_base_dir,
                'auto_create': self.auto_create_directories,
                'formats': {
                    'csv': self.save_csv_format,
                    'json': self.save_json_format
                }
            },
            'fetching': {
                'interval': self.default_interval,
                'max_candles': self.default_max_candles,
                'quote_asset': self.quote_asset
            },
            'rate_limiting': {
                'request_delay_ms': self.request_delay_ms,
                'max_concurrent': self.max_concurrent_fetches
            },
            'data_management': {
                'freshness_threshold_hours': self.data_freshness_threshold_hours,
                'keep_latest': self.keep_latest_datasets,
                'auto_cleanup': self.auto_cleanup_old_datasets
            },
            'api': {
                'base_url': self.binance_base_url,
                'timeout': self.api_timeout_seconds,
                'retries': self.max_retries
            }
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataPipelineConfig':
        """Create configuration from dictionary"""
        config = cls()

        if 'storage' in config_dict:
            config.storage_base_dir = config_dict['storage'].get('base_dir', config.storage_base_dir)
            config.auto_create_directories = config_dict['storage'].get('auto_create', config.auto_create_directories)
            if 'formats' in config_dict['storage']:
                config.save_csv_format = config_dict['storage']['formats'].get('csv', config.save_csv_format)
                config.save_json_format = config_dict['storage']['formats'].get('json', config.save_json_format)

        if 'fetching' in config_dict:
            config.default_interval = config_dict['fetching'].get('interval', config.default_interval)
            config.default_max_candles = config_dict['fetching'].get('max_candles', config.default_max_candles)
            config.quote_asset = config_dict['fetching'].get('quote_asset', config.quote_asset)

        return config

    def __repr__(self) -> str:
        return (
            f"DataPipelineConfig(\n"
            f"  storage_dir={self.storage_base_dir},\n"
            f"  interval={self.default_interval},\n"
            f"  max_candles={self.default_max_candles},\n"
            f"  freshness_threshold={self.data_freshness_threshold_hours}h\n"
            f")"
        )


# Default configuration instance
DEFAULT_CONFIG = DataPipelineConfig()


# Preset configurations for different use cases
class ConfigPresets:
    """Predefined configurations for common scenarios"""

    @staticmethod
    def quick_test() -> DataPipelineConfig:
        """Configuration for quick testing (small datasets)"""
        config = DataPipelineConfig()
        config.default_max_candles = 100
        config.keep_latest_datasets = 2
        return config

    @staticmethod
    def production() -> DataPipelineConfig:
        """Configuration for production use (maximum data)"""
        config = DataPipelineConfig()
        config.default_max_candles = 50000  # ~5.7 years for 1h
        config.keep_latest_datasets = 10
        config.auto_cleanup_old_datasets = True
        config.validate_data_integrity = True
        return config

    @staticmethod
    def ml_training() -> DataPipelineConfig:
        """Configuration optimized for ML model training"""
        config = DataPipelineConfig()
        config.default_max_candles = 20000  # ~2.3 years for 1h
        config.save_csv_format = True
        config.save_json_format = False  # CSV is more efficient for ML
        config.remove_duplicates = True
        config.fill_missing_candles = True
        return config

    @staticmethod
    def real_time_prediction() -> DataPipelineConfig:
        """Configuration for real-time prediction (recent data only)"""
        config = DataPipelineConfig()
        config.default_max_candles = 720  # 30 days for 1h
        config.data_freshness_threshold_hours = 1  # Fetch new data every hour
        config.auto_fetch_if_old = True
        config.keep_latest_datasets = 3
        return config
