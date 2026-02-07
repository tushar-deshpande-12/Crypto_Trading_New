"""
Data Pipeline Module
Handles cryptocurrency and stock data fetching, storage, and management

Components:
- fetcher: Fetches OHLCV data from Binance API (crypto)
- stock_fetcher: Fetches OHLCV data from NSE India (stocks)
- storage: Manages folder-based storage with time-series organization
- manager: Coordinates crypto data operations and provides high-level interface
- stock_manager: Coordinates stock data operations (NSE India)
- config: Configuration management for the data pipeline
"""

from .fetcher import CryptoDataFetcher
from .stock_fetcher import StockDataFetcher
from .storage import DataStorage
from .manager import DataManager
from .stock_manager import StockDataManager
from .config import DataPipelineConfig, ConfigPresets

__all__ = [
    'CryptoDataFetcher',
    'StockDataFetcher',
    'DataStorage',
    'DataManager',
    'StockDataManager',
    'DataPipelineConfig',
    'ConfigPresets'
]
