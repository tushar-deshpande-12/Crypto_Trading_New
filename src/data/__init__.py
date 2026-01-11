"""
Data Pipeline Module
Handles cryptocurrency data fetching, storage, and management

Components:
- fetcher: Fetches OHLCV data from Binance API
- storage: Manages folder-based storage with time-series organization
- manager: Coordinates data operations and provides high-level interface
- config: Configuration management for the data pipeline
"""

from .fetcher import CryptoDataFetcher
from .storage import DataStorage
from .manager import DataManager
from .config import DataPipelineConfig, ConfigPresets

__all__ = ['CryptoDataFetcher', 'DataStorage', 'DataManager', 'DataPipelineConfig', 'ConfigPresets']
