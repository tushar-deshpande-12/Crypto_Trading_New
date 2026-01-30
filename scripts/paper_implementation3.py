"""
Paper Implementation: Blockchain-Native Asset Direction Prediction
==================================================================
Implementation of: "Blockchain-Native Asset Direction Prediction: A Confidence-Threshold
Approach to Decentralized Financial Analytics Using Multi-Scale Feature Integration"
Source: MDPI Algorithms 18(12), 758 (November 2025)

Key Features:
- Binary classification (UP/DOWN) instead of 3-class
- Post-hoc confidence thresholds for selective execution
- Multi-scale feature integration (order book + macro momentum)
- Precision-recall trade-off analysis
- Short-term prediction (10-60 minutes)

Target Results from Paper:
- Direction accuracy on executed trades: 82.68%
- Average net profit per trade: 151.11 bp
- Market coverage at high confidence: 11.99%

Author: Crypto AI Predictor
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score
from scipy.stats import spearmanr, pearsonr
import requests
import time as time_module

# =============================================================================
# QUICK CONTROL VARIABLES (EDIT THESE FIRST)
# =============================================================================

# Set to False to skip all plotting (faster training)
ENABLE_PLOTS = True

# Set to False to skip saving plots to disk
SAVE_PLOTS_TO_DISK = False

# Set to True for verbose debug output
DEBUG_MODE = False

LABEL_SMOOTHNING = 0.01
# =============================================================================
# CONFIGURATION - EDIT THESE PARAMETERS
# =============================================================================

@dataclass
class Config:
    """All configurable parameters in one place."""

    # DATA SOURCE (set ONE of these)
    CSV_FILE: Optional[str] = r"C:\crypto\ver7\dataset\BTCUSDT\2026-01-11_11-07-31_50000candles\data.csv"
    SYMBOL: str = "BTCUSDT"
    TIMEFRAME: str = "1m"  # Changed from "1h" to "1m" for short-term prediction

    # DATA PARAMETERS
    TRAIN_SIZE: float = 0.7
    VAL_SIZE: float = 0.15
    TEST_SIZE: float = 0.15
    SEQUENCE_LENGTH: int = 30  # Shorter lookback to prevent overfitting
    PREDICTION_HORIZON: int = 120  # Predict 60 periods ahead (1 hour with 1m data)

    # PREDICTION HORIZON PRESETS (in minutes)
    HORIZON_PRESETS: Dict[str, int] = field(default_factory=lambda: {
        '10min': 10,
        '30min': 30,
        '60min': 60
    })

    # MODEL PARAMETERS - Ultra-conservative for consistent val loss decrease
    HIDDEN_SIZE: int = 16  # Reverted - small model prevents overfitting
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.8  # Increased to reduce overfitting
    BIDIRECTIONAL: bool = True
    USE_ATTENTION: bool = True  # Enable attention for better pattern focus
    USE_CONVOLUTIONS: bool = False  # Reverted - caused overfitting

    # TRAINING PARAMETERS - Slow learning with faster initial phase
    BATCH_SIZE: int = 512  # Large batch for stable gradients
    LEARNING_RATE: float = 0.00005  # Slightly higher than 0.00004, but not too aggressive
    WEIGHT_DECAY: float = 0.25  # Increased to reduce overfitting
    MAX_EPOCHS = 40
    EARLY_STOPPING_PATIENCE = 50
    GRADIENT_CLIP: float = 1.0  # Standard value, 0.5 was too aggressive

    # BINARY CLASSIFICATION (Paper: 2-class instead of 3-class)
    USE_BINARY_CLASSIFICATION: bool = True  # True = UP/DOWN only, False = UP/NEUTRAL/DOWN
    DEADBAND_THRESHOLD: float = 0.0005  # 5 basis points - reverted

    # CONFIDENCE THRESHOLDS (Paper: Post-hoc confidence thresholds)
    MODERATE_CONFIDENCE: float = 0.6  # Moderate confidence threshold
    HIGH_CONFIDENCE: float = 0.65  # Sweet spot: 10.8% coverage with positive profit

    CONFIDENCE_THRESHOLD = 0.52
    CONFIDENCE_THRESHOLDS: List[float] = field(default_factory=lambda: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])

    # BACKTESTING
    INITIAL_CAPITAL: float = 10000.0
    TRADING_FEE: float = 0.001  # 0.1% (10 basis points)
    CONFIDENCE_THRESHOLD: float = 0.6  # Default threshold for backtest

    # ORDER BOOK DATA
    ORDERBOOK_DIR: str = r"C:\crypto\ver7\dataset\BTCUSDT\orderbook"
    USE_ORDERBOOK_FEATURES: bool = True
    MATCH_ORDERBOOK_TIMERANGE: bool = False  # Set True when you have 24+ hours of order book data

    # CHECKPOINT & PROGRESS
    CHECKPOINT_DIR: str = "checkpoints"
    PROGRESS_FILE: str = "results/progress.json"
    SAVE_CHECKPOINTS: bool = True

    # OUTPUT
    MODEL_SAVE_PATH: str = "models/direction_model.pt"
    RESULTS_DIR: str = "results"
    SHOW_PLOTS: bool = ENABLE_PLOTS
    SAVE_PLOTS: bool = SAVE_PLOTS_TO_DISK


# Initialize config
CONFIG = Config()


# =============================================================================
# PROGRESS TRACKING (Task 4.2)
# =============================================================================

class ProgressTracker:
    """Track implementation progress and enable restart from checkpoints."""

    def __init__(self, config: Config):
        self.config = config
        self.progress_file = Path(config.PROGRESS_FILE)
        self.progress = self._load_progress()

    def _load_progress(self) -> Dict:
        """Load progress from file or create new."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            'current_phase': 1,
            'current_task': '1.1',
            'completed_tasks': [],
            'last_checkpoint': None,
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }

    def save_progress(self):
        """Save progress to file."""
        self.progress['last_updated'] = datetime.now().isoformat()
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def mark_task_complete(self, task_id: str):
        """Mark a task as complete."""
        if task_id not in self.progress['completed_tasks']:
            self.progress['completed_tasks'].append(task_id)
        self.save_progress()

    def is_task_complete(self, task_id: str) -> bool:
        """Check if a task is complete."""
        return task_id in self.progress['completed_tasks']

    def set_checkpoint(self, checkpoint_path: str):
        """Record checkpoint location."""
        self.progress['last_checkpoint'] = checkpoint_path
        self.save_progress()


# =============================================================================
# CHECKPOINT MANAGEMENT (Task 4.1)
# =============================================================================

class CheckpointManager:
    """Save and load training checkpoints."""

    def __init__(self, config: Config):
        self.config = config
        self.checkpoint_dir = Path(config.CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer,
                        scaler: StandardScaler, history: Dict, config: Config,
                        best_val_loss: float, best_ic: float = None) -> str:
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            'history': history,
            'config': asdict(config),
            'best_val_loss': best_val_loss,
            'best_ic': best_ic
        }

        torch.save(checkpoint, checkpoint_path)

        # Save scaler separately (sklearn object)
        scaler_path = self.checkpoint_dir / f"scaler_epoch_{epoch}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str, model: nn.Module,
                        optimizer: torch.optim.Optimizer = None) -> Dict:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scaler
        scaler_path = checkpoint_path.replace('checkpoint_', 'scaler_').replace('.pt', '.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                checkpoint['scaler'] = pickle.load(f)

        return checkpoint

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the most recent checkpoint path."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if not checkpoints:
            return None
        return str(max(checkpoints, key=lambda p: int(p.stem.split('_')[-1])))


# =============================================================================
# SELECTIVE EXECUTOR (Task 1.3: Post-Hoc Confidence Thresholds)
# =============================================================================

class SelectiveExecutor:
    """
    Implements post-hoc confidence thresholds from the paper.
    Separates prediction from execution decision.
    """

    def __init__(self, config: Config):
        self.config = config
        self.moderate_threshold = config.MODERATE_CONFIDENCE
        self.high_threshold = config.HIGH_CONFIDENCE

    def should_execute(self, confidence: float, threshold: float = None) -> bool:
        """Determine if a trade should be executed based on confidence."""
        if threshold is None:
            threshold = self.moderate_threshold
        return confidence >= threshold

    def get_confidence_level(self, confidence: float) -> str:
        """Categorize confidence level."""
        if confidence >= self.high_threshold:
            return "HIGH"
        elif confidence >= self.moderate_threshold:
            return "MODERATE"
        else:
            return "LOW"

    def filter_predictions(self, predictions: np.ndarray, probabilities: np.ndarray,
                           threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter predictions based on confidence threshold.

        Returns:
            filtered_predictions: Predictions above threshold
            filtered_probs: Corresponding probabilities
            mask: Boolean mask for which predictions passed
        """
        confidences = probabilities.max(axis=1)
        mask = confidences >= threshold

        return predictions[mask], probabilities[mask], mask

    def analyze_coverage_accuracy_tradeoff(self, predictions: np.ndarray,
                                            targets: np.ndarray,
                                            probabilities: np.ndarray) -> pd.DataFrame:
        """
        Analyze the trade-off between coverage and accuracy at different thresholds.
        This is a key contribution from the paper.
        """
        results = []
        confidences = probabilities.max(axis=1)

        for threshold in self.config.CONFIDENCE_THRESHOLDS:
            mask = confidences >= threshold
            coverage = mask.sum() / len(predictions)

            if mask.sum() > 0:
                accuracy = accuracy_score(targets[mask], predictions[mask])
                avg_confidence = confidences[mask].mean()
            else:
                accuracy = 0
                avg_confidence = 0

            results.append({
                'threshold': threshold,
                'coverage': coverage,
                'accuracy': accuracy,
                'num_predictions': mask.sum(),
                'avg_confidence': avg_confidence
            })

        return pd.DataFrame(results)


# =============================================================================
# DATA LOADING
# =============================================================================

def fetch_binance_ohlcv(symbol: str, interval: str, start_time: datetime, end_time: datetime,
                         limit_per_request: int = 1000) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance API for a specific time range.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Kline interval ('1m', '5m', '1h', etc.)
        start_time: Start datetime (UTC)
        end_time: End datetime (UTC)
        limit_per_request: Max candles per API request (Binance limit: 1000)

    Returns:
        DataFrame with OHLCV data
    """
    base_url = "https://api.binance.com/api/v3/klines"
    all_data = []

    # Convert to milliseconds
    current_start = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    # Calculate interval in milliseconds
    interval_ms = {
        '1m': 60000, '3m': 180000, '5m': 300000, '15m': 900000,
        '30m': 1800000, '1h': 3600000, '4h': 14400000, '1d': 86400000
    }.get(interval, 60000)

    print(f"Fetching {symbol} {interval} data from Binance...")
    print(f"  Range: {start_time} to {end_time}")

    request_count = 0
    while current_start < end_ms:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_ms,
            'limit': limit_per_request
        }

        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            all_data.extend(data)
            request_count += 1

            # Move to next batch
            last_close_time = data[-1][6]  # close_time is at index 6
            current_start = last_close_time + 1

            if request_count % 10 == 0:
                print(f"  Fetched {len(all_data)} candles...")

            # Rate limit: Binance allows 1200 requests/minute
            time_module.sleep(0.1)

        except requests.exceptions.RequestException as e:
            print(f"  Error fetching data: {e}")
            break

    if not all_data:
        print("  No data received from Binance")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
        'taker_buy_quote_volume', 'ignore'
    ])

    # Convert types
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                'taker_buy_volume', 'taker_buy_quote_volume']:
        df[col] = df[col].astype(float)

    df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('datetime', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']]
    df.sort_index(inplace=True)

    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]

    print(f"  Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    return df


def get_orderbook_time_range(config) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Get the time range covered by order book data.

    Returns:
        Tuple of (start_time, end_time) or (None, None) if no order book data
    """
    orderbook_dir = Path(config.ORDERBOOK_DIR)

    if not orderbook_dir.exists():
        return None, None

    # Look for feature files
    csv_files = list(orderbook_dir.glob("*_features.csv"))
    parquet_files = list(orderbook_dir.glob("*_features.parquet"))

    if not csv_files and not parquet_files:
        return None, None

    all_times = []

    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if 'timestamp' in df.columns:
                if pd.api.types.is_numeric_dtype(df['timestamp']):
                    times = pd.to_datetime(df['timestamp'], unit='ms')
                else:
                    times = pd.to_datetime(df['timestamp'])
                all_times.extend([times.min(), times.max()])
        except Exception as e:
            print(f"  Error reading {f}: {e}")

    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
            if 'timestamp' in df.columns:
                if pd.api.types.is_numeric_dtype(df['timestamp']):
                    times = pd.to_datetime(df['timestamp'], unit='ms')
                else:
                    times = pd.to_datetime(df['timestamp'])
                all_times.extend([times.min(), times.max()])
        except Exception as e:
            print(f"  Error reading {f}: {e}")

    if not all_times:
        return None, None

    return min(all_times), max(all_times)


def load_data(config: Config, match_orderbook: bool = True) -> pd.DataFrame:
    """
    Load data from CSV or fetch from exchange.

    If match_orderbook=True and order book data exists, will fetch OHLCV data
    that matches the order book time range for 100% coverage.
    """

    # Check if we should match order book time range
    if match_orderbook:
        ob_start, ob_end = get_orderbook_time_range(config)
        if ob_start is not None and ob_end is not None:
            ob_duration_hours = (ob_end - ob_start).total_seconds() / 3600
            print(f"\n[Order Book Matching Mode]")
            print(f"  Order book data range: {ob_start} to {ob_end}")
            print(f"  Order book duration: {ob_duration_hours:.2f} hours")

            # Add buffer for feature calculation (need lookback periods)
            buffer_hours = 24  # Need historical data for indicators
            fetch_start = ob_start - timedelta(hours=buffer_hours)
            fetch_end = ob_end + timedelta(hours=1)  # Small buffer at end

            # Fetch matching OHLCV data from Binance
            df = fetch_binance_ohlcv(
                symbol=config.SYMBOL,
                interval=config.TIMEFRAME,
                start_time=fetch_start,
                end_time=fetch_end
            )

            # Minimum samples needed for training (need enough for train/val/test split + sequences)
            MIN_SAMPLES_REQUIRED = 1000

            if len(df) >= MIN_SAMPLES_REQUIRED:
                print(f"  Fetched OHLCV data matching order book range")
                print(f"  OHLCV range: {df.index[0]} to {df.index[-1]}")

                # Save fetched data for future use
                output_dir = Path(config.ORDERBOOK_DIR).parent / "matched_ohlcv"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"matched_{ob_start.strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(output_file)
                print(f"  Saved matched OHLCV to: {output_file}")

                return df
            elif len(df) > 0:
                ob_duration = (ob_end - ob_start).total_seconds() / 60
                print(f"  ✗ Order book data only covers {ob_duration:.1f} minutes ({len(df)} candles)")
                print(f"  ✗ Need at least {MIN_SAMPLES_REQUIRED} candles for training")
                print(f"  ✗ Run fetch_orderbook.py for at least 24 hours to collect enough data")
                print(f"  Falling back to CSV file (order book features will have 0% coverage)")
            else:
                print("  Failed to fetch matching data, falling back to CSV")

    if config.CSV_FILE and os.path.exists(config.CSV_FILE):
        print(f"Loading data from CSV: {config.CSV_FILE}")
        df = pd.read_csv(config.CSV_FILE)

        # Standardize column names
        df.columns = df.columns.str.lower()

        # Handle datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        elif 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])

        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

    else:
        print(f"Fetching live data for {config.SYMBOL}...")
        try:
            from src.api.ccxt_client import get_ccxt_client
            client = get_ccxt_client('binance')
            ccxt_symbol = config.SYMBOL.replace('USDT', '/USDT')
            df = client.get_ohlcv(ccxt_symbol, config.TIMEFRAME, limit=10000)
        except Exception as e:
            print(f"Error fetching data: {e}")
            print("Using sample data generation...")
            df = generate_sample_data()

    print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    return df


def generate_sample_data(n_samples: int = 5000) -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='1h')

    # Generate realistic price movement
    returns = np.random.normal(0.0001, 0.02, n_samples)
    price = 50000 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'open': price * (1 + np.random.uniform(-0.005, 0.005, n_samples)),
        'high': price * (1 + np.random.uniform(0, 0.02, n_samples)),
        'low': price * (1 - np.random.uniform(0, 0.02, n_samples)),
        'close': price,
        'volume': np.random.uniform(100, 1000, n_samples) * price / 50000
    }, index=dates)

    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators as features."""

    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # --- Returns (Stationary) ---
    df['return_1h'] = close.pct_change(1)
    df['return_4h'] = close.pct_change(4)
    df['return_24h'] = close.pct_change(24)
    df['log_return'] = np.log(close / close.shift(1))

    # --- RSI ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'] / 100  # Normalize to 0-1

    # --- MACD ---
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['macd'] = (ema12 - ema26) / close  # Normalized
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # --- Bollinger Bands ---
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma20

    # --- Stochastic ---
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df['stoch_k'] = (close - low14) / (high14 - low14)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # --- ATR (Volatility) ---
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean() / close  # Normalized

    # --- ADX ---
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df['adx'] = dx.rolling(14).mean() / 100  # Normalized
    df['plus_di'] = plus_di / 100
    df['minus_di'] = minus_di / 100

    # --- Volume Features ---
    df['volume_sma'] = volume.rolling(20).mean()
    df['volume_ratio'] = volume / df['volume_sma']

    # --- Realized Volatility ---
    df['realized_vol'] = df['log_return'].rolling(24).std() * np.sqrt(24)

    # --- Price Position ---
    df['high_low_ratio'] = (close - low) / (high - low + 1e-10)
    df['close_open_ratio'] = (close - df['open']) / (df['open'] + 1e-10)

    # --- Momentum ---
    df['roc_12'] = close.pct_change(12)
    df['roc_24'] = close.pct_change(24)

    # --- Time Features ---
    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow'] = df.index.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)

    # Drop intermediate columns
    df.drop(['hour', 'dow', 'volume_sma', 'bb_upper', 'bb_lower'], axis=1, inplace=True, errors='ignore')

    return df


def create_target(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Create target variables for direction prediction.

    Key principles (paper-consistent, ML-correct):
    ------------------------------------------------
    1. Direction is ONLY defined when future move exceeds deadband
    2. Deadband samples are NOT relabeled (no noise injection)
    3. Deadband samples are retained for analysis & inference
    4. Training code decides whether to exclude them
    5. Binary and 3-class modes both supported
    """

    df = df.copy()

    # ============================================================
    # 1. FUTURE RETURN COMPUTATION
    # ============================================================

    horizon = config.PREDICTION_HORIZON

    df['future_return'] = (
        df['close']
        .pct_change(horizon)
        .shift(-horizon)
    )

    # Basis points version (used in plots / diagnostics)
    df['future_return_bp'] = df['future_return'] * 10000.0

    # ============================================================
    # 2. TARGET CREATION WITH DEADBAND
    # ============================================================

    deadband = config.DEADBAND_THRESHOLD

    if config.USE_BINARY_CLASSIFICATION:
        # --------------------------------------------------------
        # Binary classification: 0 = DOWN, 1 = UP
        # --------------------------------------------------------

        # Initialize as NaN (unknown / filtered)
        df['target'] = np.nan

        # Assign ONLY if move is meaningful
        df.loc[df['future_return'] > deadband, 'target'] = 1
        df.loc[df['future_return'] < -deadband, 'target'] = 0

        # Mark deadband samples explicitly
        df['below_deadband'] = df['target'].isna()

        # Diagnostics (do NOT affect training)
        total_valid = df['future_return'].notna().sum()
        filtered = df['below_deadband'].sum()

        if total_valid > 0:
            print(
                f"Deadband filter "
                f"({deadband * 10000:.1f} bp): "
                f"{filtered} / {total_valid} samples "
                f"({filtered / total_valid * 100:.1f}%)"
            )

    else:
        # --------------------------------------------------------
        # 3-Class classification: 0 = DOWN, 1 = NEUTRAL, 2 = UP
        # --------------------------------------------------------

        df['target'] = 1  # NEUTRAL by default
        df.loc[df['future_return'] > deadband, 'target'] = 2
        df.loc[df['future_return'] < -deadband, 'target'] = 0

        df['below_deadband'] = False  # Neutral is explicit class

    # ============================================================
    # 3. AUXILIARY TARGETS (IMPORTANT – DO NOT REMOVE)
    # ============================================================

    # Always-available binary direction (used for ICE, IC, sanity checks)
    df['target_binary'] = (df['future_return'] > 0).astype(int)

    # Signed magnitude (useful for analysis, never for training)
    df['future_return_signed_bp'] = (
        np.sign(df['future_return']) * df['future_return_bp']
    )

    # ============================================================
    # 4. FINAL CLEANUP (NO DATA LOSS HERE)
    # ============================================================

    # Do NOT drop rows here
    # Training pipeline decides what to exclude
    return df


def add_macro_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add macro momentum features (Task 2.1).
    Daily and weekly trend features for multi-scale analysis.
    Automatically adapts lookback periods based on available data.
    """

    df = df.copy()
    close = df['close']
    n = len(df)

    # Adaptive lookback - use at most 20% of data for longest lookback
    # This ensures we have 80% of data with valid features
    max_lookback = max(24, n // 5)  # At least 24, at most 20% of data

    # Daily momentum - use smaller of 1440 (1 day for 1m data) or max_lookback
    daily_periods = min(1440, max_lookback)
    df['daily_return'] = close.pct_change(daily_periods)
    df['daily_momentum'] = close / close.shift(daily_periods) - 1
    df['daily_ma'] = close.rolling(daily_periods).mean()
    df['daily_ma_ratio'] = close / df['daily_ma']

    # Weekly momentum - use smaller of 10080 (1 week for 1m data) or max_lookback
    weekly_periods = min(10080, max_lookback)
    df['weekly_return'] = close.pct_change(weekly_periods)
    df['weekly_momentum'] = close / close.shift(weekly_periods) - 1
    df['weekly_ma'] = close.rolling(weekly_periods).mean()
    df['weekly_ma_ratio'] = close / df['weekly_ma']

    print(f"  Macro momentum lookback: daily={daily_periods}, weekly={weekly_periods} (data size: {n})")

    # Cross-timeframe momentum divergence
    if 'daily_momentum' in df.columns and 'weekly_momentum' in df.columns:
        df['momentum_divergence'] = df['daily_momentum'] - df['weekly_momentum']

    # Trend strength indicators
    df['trend_strength_short'] = close.pct_change(12).rolling(12).mean()
    df['trend_strength_medium'] = close.pct_change(24).rolling(24).mean()

    # Normalize momentum features
    for col in ['daily_return', 'weekly_return', 'momentum_divergence']:
        if col in df.columns:
            df[col] = df[col].clip(-0.5, 0.5)  # Clip extreme values

    return df


def load_orderbook_features(config: Config, df: pd.DataFrame) -> pd.DataFrame:
    """
    Load and merge order book features (Task 2.2).
    Order book data contributes 81.3% feature importance according to paper.
    """

    orderbook_dir = Path(config.ORDERBOOK_DIR)

    if not orderbook_dir.exists():
        print(f"Order book directory not found: {orderbook_dir}")
        print("Skipping order book features. Run fetch_orderbook.py first.")
        return df

    # Look for parquet files (features) in orderbook directory
    parquet_files = list(orderbook_dir.glob("*_features.parquet"))

    if not parquet_files:
        # Try CSV files
        csv_files = list(orderbook_dir.glob("*_features.csv"))
        if csv_files:
            ob_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        else:
            print("No order book feature files found.")
            return df
    else:
        ob_df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

    print(f"Loaded {len(ob_df)} order book records from {len(parquet_files or csv_files)} files")

    # Convert timestamp to datetime if needed
    if 'timestamp' in ob_df.columns:
        # Check if timestamp is numeric (milliseconds) or string (ISO format)
        if pd.api.types.is_numeric_dtype(ob_df['timestamp']):
            ob_df['datetime'] = pd.to_datetime(ob_df['timestamp'], unit='ms')
        else:
            # Handle ISO format strings like '2026-01-24T13:48:15.373401'
            ob_df['datetime'] = pd.to_datetime(ob_df['timestamp'])
        ob_df.set_index('datetime', inplace=True)
    elif 'datetime' in ob_df.columns:
        ob_df['datetime'] = pd.to_datetime(ob_df['datetime'])
        ob_df.set_index('datetime', inplace=True)

    # Map expected feature names to actual column names in data
    feature_mapping = {
        'bid_ask_spread_bps': 'spread_bps',
        'order_book_imbalance_5': 'imbalance_5',
        'order_book_imbalance_10': 'imbalance_10',
        'order_book_imbalance_20': 'imbalance_20',
        'volume_imbalance': 'value_imbalance',
        'depth_weighted_mid_price_diff': 'depth_weighted_mid',
        'order_book_slope_bid': 'bid_slope',
        'order_book_slope_ask': 'ask_slope',
    }

    # Print available columns for debugging
    print(f"  Order book columns available: {list(ob_df.columns)}")

    # Find which features exist in data (using actual names)
    available_actual = [actual for expected, actual in feature_mapping.items() if actual in ob_df.columns]

    if not available_actual:
        print("No matching order book features found in data.")
        return df

    # Rename columns to expected names for consistency
    rename_map = {v: k for k, v in feature_mapping.items() if v in ob_df.columns}
    ob_df = ob_df.rename(columns=rename_map)

    # Now use the expected feature names
    ob_features = list(rename_map.values())
    available_features = ob_features

    print(f"Using order book features: {available_features}")

    # Resample order book data to match OHLCV timeframe
    # Order book is typically sampled more frequently
    ob_resampled = ob_df[available_features].resample('1T').mean()  # 1-minute resample

    # Merge with OHLCV data
    df = df.join(ob_resampled, how='left')

    # Check overlap between order book and OHLCV data
    ob_start, ob_end = ob_resampled.index.min(), ob_resampled.index.max()
    df_start, df_end = df.index.min(), df.index.max()
    print(f"  Order book date range: {ob_start} to {ob_end}")
    print(f"  OHLCV date range: {df_start} to {df_end}")

    # Count how many rows have order book data
    non_null_count = df[available_features[0]].notna().sum()
    total_count = len(df)
    coverage_pct = non_null_count / total_count * 100
    print(f"  Order book coverage: {non_null_count}/{total_count} rows ({coverage_pct:.1f}%)")

    if coverage_pct >= 95:
        print(f"  ✓ EXCELLENT: Near-complete order book coverage!")
    elif coverage_pct >= 80:
        print(f"  ✓ GOOD: Strong order book coverage for training.")
    elif coverage_pct >= 50:
        print(f"  ! MODERATE: Partial order book coverage. Results may vary.")
    elif coverage_pct < 10:
        print(f"  ✗ WARNING: Low order book coverage! Model will train mostly without order book features.")
        print(f"  Consider using match_orderbook=True in load_data() to fetch matching OHLCV data.")
    else:
        print(f"  ! LOW: Limited order book coverage. Consider collecting more data.")

    # Forward fill gaps in order book data (limit to reasonable gap)
    for col in available_features:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill', limit=60)  # Forward fill up to 60 bars (1 hour)

    # Fill remaining NaN with 0 (neutral value) so model can still train
    for col in available_features:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    print(f"Merged {len(available_features)} order book features with OHLCV data")

    return df


def get_feature_columns(include_orderbook: bool = True, include_macro: bool = True,
                        minimal: bool = False) -> List[str]:
    """Get list of feature columns (stationary only)."""

    # Minimal feature set for reduced overfitting
    if minimal:
        return [
            'return_1h', 'return_4h', 'log_return',
            'rsi', 'macd', 'macd_hist',
            'bb_position', 'atr',
            'volume_ratio',
            'hour_sin', 'hour_cos'
        ]

    base_features = [
        'return_1h', 'return_4h', 'return_24h', 'log_return',
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_position', 'bb_width',
        'stoch_k', 'stoch_d',
        'atr', 'adx', 'plus_di', 'minus_di',
        'volume_ratio', 'realized_vol',
        'high_low_ratio', 'close_open_ratio',
        'roc_12', 'roc_24',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
    ]

    macro_features = [
        'daily_return', 'daily_momentum', 'daily_ma_ratio',
        'weekly_return', 'weekly_momentum', 'weekly_ma_ratio',
        'momentum_divergence',
        'trend_strength_short', 'trend_strength_medium'
    ]

    orderbook_features = [
        'bid_ask_spread_bps',
        'order_book_imbalance_5',
        'order_book_imbalance_10',
        'order_book_imbalance_20',
        'volume_imbalance',
        'depth_weighted_mid_price_diff',
        'order_book_slope_bid',
        'order_book_slope_ask'
    ]

    features = base_features.copy()

    if include_macro:
        features.extend(macro_features)

    if include_orderbook:
        features.extend(orderbook_features)

    return features


# =============================================================================
# DATA VISUALIZATION
# =============================================================================

def plot_price_data(df: pd.DataFrame, config: Config):
    """Plot enhanced price data with volume and key statistics."""

    # Use dark style for better visuals
    plt.style.use('dark_background')

    fig = plt.figure(figsize=(16, 12))

    # Create grid spec for custom layout
    gs = fig.add_gridspec(4, 2, height_ratios=[3, 1.5, 1.5, 1], width_ratios=[3, 1],
                          hspace=0.15, wspace=0.1)

    # Main price chart (candlestick-style with line)
    ax_price = fig.add_subplot(gs[0, 0])

    # Calculate colors based on price movement
    colors = ['#26a69a' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ef5350'
              for i in range(len(df))]

    # Plot price with gradient fill
    ax_price.fill_between(df.index, df['low'], df['high'], alpha=0.1, color='#64b5f6')
    ax_price.plot(df.index, df['close'], color='#64b5f6', linewidth=1.2, label='Close Price')

    # Add moving averages
    ma20 = df['close'].rolling(20).mean()
    ma50 = df['close'].rolling(50).mean()
    ax_price.plot(df.index, ma20, color='#ffd54f', linewidth=1, alpha=0.8, label='MA20')
    ax_price.plot(df.index, ma50, color='#ff8a65', linewidth=1, alpha=0.8, label='MA50')

    # Mark high and low points
    high_idx = df['high'].idxmax()
    low_idx = df['low'].idxmin()
    ax_price.scatter([high_idx], [df['high'].max()], color='#26a69a', s=100, marker='^', zorder=5)
    ax_price.scatter([low_idx], [df['low'].min()], color='#ef5350', s=100, marker='v', zorder=5)
    ax_price.annotate(f'High: ${df["high"].max():,.2f}', xy=(high_idx, df['high'].max()),
                     xytext=(10, 10), textcoords='offset points', fontsize=9, color='#26a69a',
                     fontweight='bold')
    ax_price.annotate(f'Low: ${df["low"].min():,.2f}', xy=(low_idx, df['low'].min()),
                     xytext=(10, -15), textcoords='offset points', fontsize=9, color='#ef5350',
                     fontweight='bold')

    # Current price annotation
    current_price = df['close'].iloc[-1]
    ax_price.axhline(y=current_price, color='#64b5f6', linestyle='--', alpha=0.5, linewidth=1)
    ax_price.annotate(f'Current: ${current_price:,.2f}', xy=(df.index[-1], current_price),
                     xytext=(5, 0), textcoords='offset points', fontsize=10, color='#64b5f6',
                     fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#1e1e1e',
                                                  edgecolor='#64b5f6', alpha=0.9))

    ax_price.set_ylabel('Price (USD)', fontsize=11, fontweight='bold')
    ax_price.set_title(f'{config.SYMBOL} Price History', fontsize=14, fontweight='bold', pad=10)
    ax_price.legend(loc='upper left', fontsize=9)
    ax_price.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax_price.set_facecolor('#1a1a2e')

    # Price statistics panel
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_stats.axis('off')
    ax_stats.set_facecolor('#1a1a2e')

    # Calculate statistics
    total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    daily_returns = df['close'].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(24 * 365) * 100  # Annualized
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(24 * 365) if daily_returns.std() > 0 else 0

    stats_text = f"""
    ╔══════════════════════════╗
    ║    PRICE STATISTICS      ║
    ╠══════════════════════════╣
    ║  Current: ${current_price:>12,.2f}  ║
    ║  High:    ${df['high'].max():>12,.2f}  ║
    ║  Low:     ${df['low'].min():>12,.2f}  ║
    ╠══════════════════════════╣
    ║  Period Return: {total_return:>+7.2f}%  ║
    ║  Volatility:    {volatility:>7.1f}%  ║
    ║  Sharpe Ratio:  {sharpe:>7.2f}   ║
    ╠══════════════════════════╣
    ║  Candles: {len(df):>14,}  ║
    ║  From: {str(df.index[0])[:10]:>14}  ║
    ║  To:   {str(df.index[-1])[:10]:>14}  ║
    ╚══════════════════════════╝
    """
    ax_stats.text(0.1, 0.95, stats_text, transform=ax_stats.transAxes, fontsize=10,
                  verticalalignment='top', fontfamily='monospace', color='#e0e0e0',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#2d2d44', edgecolor='#64b5f6', alpha=0.9))

    # Volume chart with color coding
    ax_vol = fig.add_subplot(gs[1, 0], sharex=ax_price)

    # Calculate volume MA for comparison
    vol_ma = df['volume'].rolling(20).mean()

    # Color volume bars based on price direction and relative volume
    vol_colors = []
    for i in range(len(df)):
        if df['close'].iloc[i] >= df['open'].iloc[i]:
            base_color = '#26a69a'  # Green for up
        else:
            base_color = '#ef5350'  # Red for down

        # Adjust alpha based on relative volume
        if i >= 20 and vol_ma.iloc[i] > 0:
            vol_ratio = df['volume'].iloc[i] / vol_ma.iloc[i]
            alpha = min(0.9, max(0.3, 0.4 + vol_ratio * 0.2))
        else:
            alpha = 0.6
        vol_colors.append(base_color)

    ax_vol.bar(df.index, df['volume'], color=vol_colors, alpha=0.7, width=0.03)
    ax_vol.plot(df.index, vol_ma, color='#ffd54f', linewidth=1.2, label='Vol MA20')
    ax_vol.set_ylabel('Volume', fontsize=10, fontweight='bold')
    ax_vol.legend(loc='upper left', fontsize=8)
    ax_vol.grid(True, alpha=0.2)
    ax_vol.set_facecolor('#1a1a2e')

    # Volume statistics
    ax_vol_stats = fig.add_subplot(gs[1, 1])
    ax_vol_stats.axis('off')
    avg_vol = df['volume'].mean()
    current_vol = df['volume'].iloc[-1]
    vol_change = ((current_vol / avg_vol) - 1) * 100

    vol_stats = f"""
    Volume Stats:
    ─────────────
    Current: {current_vol:,.0f}
    Average: {avg_vol:,.0f}
    vs Avg: {vol_change:+.1f}%
    """
    ax_vol_stats.text(0.1, 0.9, vol_stats, transform=ax_vol_stats.transAxes, fontsize=9,
                      verticalalignment='top', fontfamily='monospace', color='#e0e0e0')

    # Returns distribution
    ax_ret = fig.add_subplot(gs[2, 0])
    returns = df['close'].pct_change().dropna() * 100  # As percentage

    # Create histogram with KDE-like appearance
    n, bins, patches = ax_ret.hist(returns, bins=80, alpha=0.7, edgecolor='none', density=True)

    # Color bins based on positive/negative
    for i, patch in enumerate(patches):
        if bins[i] < 0:
            patch.set_facecolor('#ef5350')
        else:
            patch.set_facecolor('#26a69a')

    ax_ret.axvline(returns.mean(), color='#ffd54f', linestyle='--', linewidth=2,
                   label=f'Mean: {returns.mean():.3f}%')
    ax_ret.axvline(0, color='white', linestyle='-', linewidth=1, alpha=0.5)
    ax_ret.set_xlabel('Hourly Returns (%)', fontsize=10)
    ax_ret.set_ylabel('Density', fontsize=10)
    ax_ret.set_title('Returns Distribution', fontsize=11, fontweight='bold')
    ax_ret.legend(loc='upper right', fontsize=9)
    ax_ret.grid(True, alpha=0.2)
    ax_ret.set_facecolor('#1a1a2e')

    # Returns statistics
    ax_ret_stats = fig.add_subplot(gs[2, 1])
    ax_ret_stats.axis('off')
    skew = returns.skew()
    kurt = returns.kurtosis()

    ret_stats = f"""
    Returns Stats:
    ─────────────
    Mean:   {returns.mean():+.4f}%
    Std:    {returns.std():.4f}%
    Skew:   {skew:+.3f}
    Kurt:   {kurt:.3f}
    Min:    {returns.min():.3f}%
    Max:    {returns.max():.3f}%
    """
    ax_ret_stats.text(0.1, 0.9, ret_stats, transform=ax_ret_stats.transAxes, fontsize=9,
                      verticalalignment='top', fontfamily='monospace', color='#e0e0e0')

    # Time info bar at bottom
    ax_time = fig.add_subplot(gs[3, :])
    ax_time.axis('off')
    time_info = f"Data Range: {df.index[0]} → {df.index[-1]} | Timeframe: {config.TIMEFRAME} | Total Candles: {len(df):,}"
    ax_time.text(0.5, 0.5, time_info, transform=ax_time.transAxes, fontsize=10,
                 ha='center', va='center', color='#b0b0b0')

    plt.suptitle('', fontsize=1)  # Remove default title spacing

    if config.SAVE_PLOTS:
        Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{config.RESULTS_DIR}/price_data.png', dpi=150, facecolor='#0d0d1a',
                    edgecolor='none', bbox_inches='tight')
    if config.SHOW_PLOTS:
        plt.show()
    plt.close()

    # Reset to default style
    plt.style.use('default')


def plot_features(df: pd.DataFrame, config: Config):
    """Plot feature distributions and correlations."""

    feature_cols = [c for c in get_feature_columns() if c in df.columns]
    feature_df = df[feature_cols].dropna()

    # Feature distributions
    n_features = len(feature_cols)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        axes[i].hist(feature_df[col].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(labelsize=7)

    # Hide empty subplots
    for i in range(len(feature_cols), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Feature Distributions', fontsize=14)
    plt.tight_layout()

    if config.SAVE_PLOTS:
        plt.savefig(f'{config.RESULTS_DIR}/feature_distributions.png', dpi=150)
    if config.SHOW_PLOTS:
        plt.show()
    plt.close()

    # Correlation matrix
    plt.figure(figsize=(12, 10))
    corr = feature_df.corr()
    im = plt.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(len(feature_cols)), feature_cols, rotation=90, fontsize=7)
    plt.yticks(range(len(feature_cols)), feature_cols, fontsize=7)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()

    if config.SAVE_PLOTS:
        plt.savefig(f'{config.RESULTS_DIR}/feature_correlations.png', dpi=150)
    if config.SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_target_distribution(df: pd.DataFrame, config: Config):
    """Plot target variable distribution (handles both binary and 3-class)."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Target class distribution
    target_counts = df['target'].value_counts().sort_index()

    if config.USE_BINARY_CLASSIFICATION:
        labels = ['DOWN', 'UP']
        colors = ['#ef5350', '#26a69a']
        num_classes = 2
    else:
        labels = ['DOWN', 'NEUTRAL', 'UP']
        colors = ['#ef5350', 'gray', '#26a69a']
        num_classes = 3

    counts = [target_counts.get(i, 0) for i in range(num_classes)]
    axes[0].bar(labels, counts, color=colors, alpha=0.7)
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Target Class Distribution ({"Binary" if config.USE_BINARY_CLASSIFICATION else "3-Class"})')

    for i, v in enumerate(counts):
        if v > 0:
            axes[0].text(i, v + 50, f'{v}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=9)

    # Future returns distribution with deadband
    returns = df['future_return'].dropna()
    axes[1].hist(returns, bins=100, alpha=0.7, edgecolor='black')

    # Show deadband threshold
    deadband = config.DEADBAND_THRESHOLD
    axes[1].axvline(deadband, color='#26a69a', linestyle='--', linewidth=2,
                   label=f'Deadband: +{deadband*10000:.0f}bp')
    axes[1].axvline(-deadband, color='#ef5350', linestyle='--', linewidth=2,
                   label=f'Deadband: -{deadband*10000:.0f}bp')
    axes[1].axvspan(-deadband, deadband, alpha=0.2, color='gray', label='Filtered zone')

    axes[1].set_xlabel(f'{config.PREDICTION_HORIZON}-period Future Returns')
    axes[1].set_ylabel('Frequency')
    axes[1].legend(fontsize=8)
    axes[1].set_title('Future Returns Distribution with Deadband')

    # Add deadband filter statistics
    if 'below_deadband' in df.columns:
        filtered_pct = df['below_deadband'].sum() / len(df) * 100
        axes[1].text(0.02, 0.98, f'Filtered by deadband: {filtered_pct:.1f}%',
                    transform=axes[1].transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if config.SAVE_PLOTS:
        plt.savefig(f'{config.RESULTS_DIR}/target_distribution.png', dpi=150)
    if config.SHOW_PLOTS:
        plt.show()
    plt.close()


# =============================================================================
# DATASET & MODEL
# =============================================================================

class DirectionDataset(Dataset):
    """PyTorch Dataset for direction prediction."""

    def __init__(self, data: np.ndarray, targets: np.ndarray, seq_length: int):
        self.data = torch.FloatTensor(data)
        self.targets = torch.LongTensor(targets)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.targets[idx + self.seq_length - 1]
        return x, y


class AttentionLayer(nn.Module):
    """Attention mechanism for LSTM."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        attention_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(lstm_output * attention_weights, dim=1)  # (batch, hidden_size)
        return context, attention_weights


class MultiScaleConvBlock(nn.Module):
    """Multi-scale temporal convolution block for capturing patterns at different scales."""

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int] = [3, 5, 7]):
        super().__init__()
        n_scales = len(kernel_sizes)
        # Ensure channels divide evenly by distributing remainder to first convs
        base_channels = out_channels // n_scales
        remainder = out_channels % n_scales

        self.convs = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            # Add 1 extra channel to first 'remainder' convolutions
            ch = base_channels + (1 if i < remainder else 0)
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_channels, ch, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(ch),
                nn.GELU()
            ))

        self.out_channels = out_channels

    def forward(self, x):
        # x: (batch, channels, seq_len)
        outputs = [conv(x) for conv in self.convs]
        return torch.cat(outputs, dim=1)  # (batch, out_channels, seq_len)


class DirectionLSTM(nn.Module):
    """
    Multi-Scale Feature Integration Model for Direction Prediction.

    Architecture based on paper: "Blockchain-Native Asset Direction Prediction"
    Key components:
    1. Multi-scale temporal convolutions (captures patterns at 3, 5, 7 timestep scales)
    2. Bidirectional LSTM with residual connections
    3. Multi-head self-attention for temporal weighting
    4. Feature gating mechanism
    """

    def __init__(self, config: Config, input_size: int):
        super().__init__()
        self.config = config
        self.input_size = input_size

        # Number of output classes based on config
        self.num_classes = 2 if config.USE_BINARY_CLASSIFICATION else 3

        # === Multi-Scale Feature Extraction (Paper Section 3.2) ===
        self.input_norm = nn.LayerNorm(input_size)
        self.use_convolutions = config.USE_CONVOLUTIONS

        if self.use_convolutions:
            # Project to higher dimension for conv processing
            conv_hidden = config.HIDDEN_SIZE
            self.input_proj = nn.Linear(input_size, conv_hidden)

            # Multi-scale convolution blocks
            self.multi_scale_conv1 = MultiScaleConvBlock(conv_hidden, conv_hidden, kernel_sizes=[3, 5, 7])
            self.multi_scale_conv2 = MultiScaleConvBlock(conv_hidden, conv_hidden, kernel_sizes=[3, 5, 7])

            # Residual projection
            self.conv_residual = nn.Conv1d(conv_hidden, conv_hidden, kernel_size=1)
            lstm_input_size = conv_hidden
        else:
            # Direct LSTM without convolutions
            lstm_input_size = input_size

        # === Temporal Sequence Modeling ===
        # Stacked bidirectional LSTM with layer normalization
        self.lstm_layers = nn.ModuleList()
        self.lstm_norms = nn.ModuleList()

        for i in range(config.NUM_LAYERS):
            layer_input = lstm_input_size if i == 0 else config.HIDDEN_SIZE * 2
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input,
                    hidden_size=config.HIDDEN_SIZE,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=config.BIDIRECTIONAL
                )
            )
            self.lstm_norms.append(nn.LayerNorm(config.HIDDEN_SIZE * 2))

        lstm_output_size = config.HIDDEN_SIZE * (2 if config.BIDIRECTIONAL else 1)

        # === Multi-Head Self-Attention (Paper Section 3.3) ===
        self.use_attention = config.USE_ATTENTION
        if self.use_attention:
            self.self_attention = nn.MultiheadAttention(
                embed_dim=lstm_output_size,
                num_heads=4,
                dropout=config.DROPOUT,
                batch_first=True
            )
            self.attn_norm = nn.LayerNorm(lstm_output_size)

        # === Feature Gating (Paper Section 3.4) ===
        # Input is concatenated last_hidden + mean_hidden = 2 * lstm_output_size
        pooled_size = lstm_output_size * 2
        self.feature_gate = nn.Sequential(
            nn.Linear(pooled_size, pooled_size),
            nn.Sigmoid()
        )

        # === Classification Head (simplified to reduce overfitting) ===
        self.classifier = nn.Sequential(
            nn.Linear(pooled_size, config.HIDDEN_SIZE),
            nn.LayerNorm(config.HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_SIZE, self.num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize LSTM weights
        for lstm in self.lstm_layers:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
                    # Set forget gate bias to 1 for better gradient flow
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)

        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # x: (batch, seq_len, features)
        batch_size, seq_len, _ = x.shape

        # Input normalization
        x = self.input_norm(x)

        if self.use_convolutions:
            # Project and apply multi-scale convolutions
            x = self.input_proj(x)  # (batch, seq_len, hidden)

            # Multi-scale convolution (transpose for conv1d)
            x_conv = x.transpose(1, 2)  # (batch, hidden, seq_len)
            residual = self.conv_residual(x_conv)

            x_conv = self.multi_scale_conv1(x_conv)
            x_conv = self.multi_scale_conv2(x_conv)
            x_conv = x_conv + residual  # Residual connection

            x = x_conv.transpose(1, 2)  # (batch, seq_len, hidden)

        # Stacked LSTM with residual connections
        for i, (lstm, norm) in enumerate(zip(self.lstm_layers, self.lstm_norms)):
            lstm_out, _ = lstm(x)
            lstm_out = norm(lstm_out)

            # Residual connection (after first layer when dimensions match)
            if i > 0:
                lstm_out = lstm_out + x

            x = lstm_out

        # Multi-head self-attention with residual
        if self.use_attention:
            attn_out, _ = self.self_attention(x, x, x)
            x = self.attn_norm(x + attn_out)

        # Global pooling: concatenate last timestep with mean pooling
        last_hidden = x[:, -1, :]  # (batch, hidden*2)
        mean_hidden = x.mean(dim=1)  # (batch, hidden*2)
        context = torch.cat([last_hidden, mean_hidden], dim=1)  # (batch, hidden*4)

        # Feature gating
        gate = self.feature_gate(context)
        context = context * gate

        # Classification
        logits = self.classifier(context)
        return logits

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs


# =============================================================================
# TRAINING
# =============================================================================
def prepare_data(
    df: pd.DataFrame,
    config: Config
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, int]:
    """
    Prepare data for training, validation, and testing.

    Returns:
        train_loader, val_loader, test_loader, scaler, num_features

    IMPORTANT BEHAVIOR:
    -------------------
    - Deadband samples (target is NaN) are EXCLUDED from training/val/test
    - Deadband samples are NOT modified here
    - Temporal ordering is preserved (NO shuffling before split)
    - Uses ONLY existing classes and functions from provided code
    """

    # ============================================================
    # 1. SELECT FEATURES (ONLY THOSE THAT EXIST)
    # ============================================================

    feature_cols = [c for c in get_feature_columns() if c in df.columns]

    if len(feature_cols) == 0:
        raise ValueError("No valid feature columns found in dataframe")

    # ============================================================
    # 1.5 REMOVE HIGHLY CORRELATED FEATURES
    # ============================================================

    CORRELATION_THRESHOLD = 0.85  # Remove features with correlation > 0.85

    # Calculate correlation matrix on a sample (for speed)
    sample_df = df[feature_cols].dropna().head(10000)
    if len(sample_df) > 100:
        corr_matrix = sample_df.corr().abs()

        # Find pairs of highly correlated features
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to drop (keep first feature of each correlated pair)
        features_to_drop = set()
        for column in upper_triangle.columns:
            correlated = upper_triangle.index[upper_triangle[column] > CORRELATION_THRESHOLD].tolist()
            for feat in correlated:
                if feat not in features_to_drop and column not in features_to_drop:
                    features_to_drop.add(feat)

        if features_to_drop:
            print(f"Removing {len(features_to_drop)} highly correlated features (>{CORRELATION_THRESHOLD}):")
            print(f"  Dropped: {sorted(features_to_drop)}")
            feature_cols = [f for f in feature_cols if f not in features_to_drop]
        else:
            print(f"No highly correlated features found (threshold={CORRELATION_THRESHOLD})")

    print(f"Features after correlation filter: {len(feature_cols)}")

    # ============================================================
    # 2. DROP ROWS WITH NaN/Inf IN FEATURES OR TARGET
    # ============================================================

    # Select columns we need
    df_subset = df[feature_cols + ['target']].copy()

    # Replace inf with NaN for proper handling
    df_subset = df_subset.replace([np.inf, -np.inf], np.nan)

    # Debug: show NaN info before dropping
    print(f"Data shape before NaN filter: {df_subset.shape}")
    nan_per_col = df_subset.isna().sum()
    cols_with_nan = nan_per_col[nan_per_col > 0]
    if len(cols_with_nan) > 0:
        print(f"  Columns with NaN (top 5): {dict(cols_with_nan.nlargest(5))}")

    # Drop rows where ANY column (features OR target) is NaN
    df_clean = df_subset.dropna()

    print(f"Clean data shape (after NaN/Inf filter): {df_clean.shape}")
    print(f"Number of features: {len(feature_cols)}")

    # ============================================================
    # 3. TEMPORAL TRAIN / VAL / TEST SPLIT
    # ============================================================

    n = len(df_clean)
    min_required = config.SEQUENCE_LENGTH + 10
    if n < min_required:
        print(f"\n*** DEBUG: Data too small ***")
        print(f"  Total rows in df: {len(df)}")
        print(f"  Rows after NaN filter: {n}")
        print(f"  Minimum required: {min_required}")
        print(f"  SEQUENCE_LENGTH: {config.SEQUENCE_LENGTH}")

        # Show NaN counts per column
        nan_counts = df[feature_cols + ['target']].isna().sum()
        high_nan = nan_counts[nan_counts > len(df) * 0.5]
        if len(high_nan) > 0:
            print(f"  Columns with >50% NaN: {dict(high_nan)}")

        raise ValueError(f"Not enough samples: {n} < {min_required}. Need more data or reduce SEQUENCE_LENGTH.")

    train_end = int(n * config.TRAIN_SIZE)
    val_end = int(n * (config.TRAIN_SIZE + config.VAL_SIZE))

    train_df = df_clean.iloc[:train_end]
    val_df   = df_clean.iloc[train_end:val_end]
    test_df  = df_clean.iloc[val_end:]

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Test samples:  {len(test_df)}")

    # ============================================================
    # 4. FEATURE SCALING (TRAIN FIT ONLY)
    # ============================================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(train_df[feature_cols])
    X_val   = scaler.transform(val_df[feature_cols])
    X_test  = scaler.transform(test_df[feature_cols])

    # Safety: replace any remaining NaN/Inf after scaling with 0
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  NaN/Inf check - Train: {np.isnan(X_train).sum()}, Val: {np.isnan(X_val).sum()}, Test: {np.isnan(X_test).sum()}")

    y_train = train_df['target'].astype(int).values
    y_val   = val_df['target'].astype(int).values
    y_test  = test_df['target'].astype(int).values

    # ============================================================
    # 5. DATASET CREATION (USING EXISTING DirectionDataset)
    # ============================================================

    train_dataset = DirectionDataset(
        X_train,
        y_train,
        config.SEQUENCE_LENGTH
    )

    val_dataset = DirectionDataset(
        X_val,
        y_val,
        config.SEQUENCE_LENGTH
    )

    test_dataset = DirectionDataset(
        X_test,
        y_test,
        config.SEQUENCE_LENGTH
    )

    # ============================================================
    # 6. DATALOADERS (MATCH EXISTING TRAINING CODE)
    # ============================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, scaler, len(feature_cols)



def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                config: Config, device: torch.device) -> Dict:
    """Train the model with improved training techniques from the paper."""

    model = model.to(device)

    # Calculate class weights for imbalanced data
    all_targets = []
    for _, y in train_loader:
        all_targets.extend(y.numpy())
    all_targets = np.array(all_targets)
    class_counts = np.bincount(all_targets, minlength=2 if config.USE_BINARY_CLASSIFICATION else 3)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.FloatTensor(class_weights).to(device)

    # Label smoothing - balanced value
    label_smoothing = LABEL_SMOOTHNING  # Between 0.1 (overfit) and 0.2 (too smooth)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    # AdamW with very low learning rate and numerical stability
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8  # Numerical stability for gradient updates
    )

    # Cosine annealing scheduler for smooth learning rate decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.MAX_EPOCHS - 5,  # Exclude warmup epochs
        eta_min=config.LEARNING_RATE * 0.01  # Minimum LR = 1% of initial
    )

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_ic': [], 'lr': []}
    best_ic = -float('inf')  # IC can be negative, so start at -inf
    patience_counter = 0

    # Gentler warmup - smoother transition to full learning rate
    warmup_epochs = 5
    warmup_lr_factor = 0.3  # Start at 30% of LR instead of 10%

    print("\nTraining started...")
    print(f"  - Class weights: {class_weights.cpu().numpy()}")
    print(f"  - Using label smoothing: {label_smoothing}")
    print(f"  - Warmup epochs: {warmup_epochs}")
    print("-" * 60)

    for epoch in range(config.MAX_EPOCHS):
        # Learning rate warmup
        if epoch < warmup_epochs:
            warmup_factor = warmup_lr_factor + (1 - warmup_lr_factor) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.LEARNING_RATE * warmup_factor

        # Training
        model.train()
        train_loss = 0
        num_batches = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            # Skip batches with NaN (shouldn't happen after fixes, but safety check)
            if torch.isnan(X).any():
                continue

            optimizer.zero_grad()
            logits = model(X)

            # Check for NaN in logits
            if torch.isnan(logits).any():
                continue

            loss = criterion(logits, y)

            # Skip if loss is NaN
            if torch.isnan(loss):
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / max(num_batches, 1)

        # Step scheduler after warmup (if scheduler exists)
        if scheduler is not None and epoch >= warmup_epochs:
            scheduler.step()

        # Validation (use unweighted loss for fair comparison)
        model.eval()
        val_loss = 0
        all_preds, all_targets, all_probs = [], [], []
        val_criterion = nn.CrossEntropyLoss()  # No class weights for validation

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = val_criterion(logits, y)
                val_loss += loss.item()

                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_targets, all_preds)

        # Calculate IC for early stopping
        all_probs_arr = np.array(all_probs)
        all_preds_arr = np.array(all_preds)
        all_targets_arr = np.array(all_targets)
        ic_metrics = calculate_ic(all_preds_arr, all_targets_arr, all_probs_arr)
        val_ic = ic_metrics['ic_spearman']

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_ic'].append(val_ic)
        history['lr'].append(current_lr)

        # Early stopping based on IC (higher is better)
        if val_ic > best_ic:
            best_ic = val_ic
            patience_counter = 0
            # Save best model
            Path(config.MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | IC: {val_ic:.4f} | LR: {current_lr:.2e}")

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1} (best IC: {best_ic:.4f})")
            break

    # Load best model
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))

    return history


def plot_training_history(history: Dict, config: Config):
    """Plot training history."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history['val_acc'], label='Validation Accuracy', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # IC (Information Coefficient)
    axes[2].plot(history['val_ic'], label='Validation IC', color='purple')
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IC')
    axes[2].set_title('Validation IC (Early Stop Metric)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if config.SAVE_PLOTS:
        plt.savefig(f'{config.RESULTS_DIR}/training_history.png', dpi=150)
    if config.SHOW_PLOTS:
        plt.show()
    plt.close()


# =============================================================================
# EVALUATION & PREDICTION
# =============================================================================

def calculate_ic(predictions: np.ndarray, targets: np.ndarray, probs: np.ndarray = None) -> Dict:
    """
    Calculate Information Coefficient (IC) and related metrics.
    Supports both binary (0=DOWN, 1=UP) and 3-class (0=DOWN, 1=NEUTRAL, 2=UP) classification.

    IC is the Spearman rank correlation between predictions and actual outcomes.
    It measures the predictive power of the model.

    Args:
        predictions: Model predictions (class labels)
        targets: Actual class labels
        probs: Prediction probabilities (optional, for probability-based IC)

    Returns:
        Dict with IC metrics
    """
    # Standard IC: Spearman correlation between predictions and targets
    ic_spearman, ic_pvalue = spearmanr(predictions, targets)

    # Pearson IC
    ic_pearson, pearson_pvalue = pearsonr(predictions, targets)

    # Determine if binary or 3-class based on number of probability columns
    is_binary = probs is not None and probs.shape[1] == 2

    # Probability-based IC (if probs provided)
    if probs is not None:
        confidence_scores = np.zeros(len(predictions))

        if is_binary:
            # Binary: 0=DOWN, 1=UP
            for i in range(len(predictions)):
                if predictions[i] == 1:  # UP
                    confidence_scores[i] = probs[i, 1]  # Prob of UP
                else:  # DOWN
                    confidence_scores[i] = -probs[i, 0]  # Negative prob of DOWN

            # Create signed target: +1 for UP (1), -1 for DOWN (0)
            signed_targets = np.where(targets == 1, 1, -1)
        else:
            # 3-class: 0=DOWN, 1=NEUTRAL, 2=UP
            for i in range(len(predictions)):
                if predictions[i] == 2:  # UP
                    confidence_scores[i] = probs[i, 2]  # Prob of UP
                elif predictions[i] == 0:  # DOWN
                    confidence_scores[i] = -probs[i, 0]  # Negative prob of DOWN
                else:  # NEUTRAL
                    confidence_scores[i] = 0

            # Create signed target: +1 for UP, -1 for DOWN, 0 for NEUTRAL
            signed_targets = np.where(targets == 2, 1, np.where(targets == 0, -1, 0))

        ic_prob, prob_pvalue = spearmanr(confidence_scores, signed_targets)
    else:
        ic_prob = ic_spearman
        prob_pvalue = ic_pvalue

    # IC Rating
    abs_ic = abs(ic_spearman)
    if abs_ic > 0.10:
        ic_rating = "EXCELLENT"
        ic_description = "Strong predictive signal - Highly tradeable"
    elif abs_ic > 0.05:
        ic_rating = "GOOD"
        ic_description = "Meaningful predictive power - Tradeable"
    elif abs_ic > 0.02:
        ic_rating = "MODERATE"
        ic_description = "Some predictive signal - Use with caution"
    elif abs_ic > 0.01:
        ic_rating = "WEAK"
        ic_description = "Marginal signal - May not be profitable after costs"
    else:
        ic_rating = "NONE"
        ic_description = "No predictive power detected"

    return {
        'ic_spearman': ic_spearman,
        'ic_pearson': ic_pearson,
        'ic_probability': ic_prob,
        'ic_pvalue': ic_pvalue,
        'ic_rating': ic_rating,
        'ic_description': ic_description
    }


def evaluate_model(model: nn.Module, test_loader: DataLoader,
                   device: torch.device, config: Config) -> Dict:
    """Evaluate model on test set with ICE metrics and confidence threshold analysis."""

    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Basic Metrics
    accuracy = accuracy_score(all_targets, all_preds)
    num_classes = all_probs.shape[1]

    # Direction accuracy
    if config.USE_BINARY_CLASSIFICATION:
        # For binary: both classes are directional
        direction_acc = accuracy
        class_names = ['DOWN', 'UP']
    else:
        # For 3-class: exclude neutral predictions
        mask = (all_targets != 1) & (all_preds != 1)
        if mask.sum() > 0:
            direction_acc = accuracy_score(all_targets[mask], all_preds[mask])
        else:
            direction_acc = 0
        class_names = ['DOWN', 'NEUTRAL', 'UP']

    # Calculate Information Coefficient (ICE)
    ic_metrics = calculate_ic(all_preds, all_targets, all_probs)

    # Selective execution analysis (Paper: Task 1.3)
    selective_executor = SelectiveExecutor(config)
    coverage_df = selective_executor.analyze_coverage_accuracy_tradeoff(
        all_preds, all_targets, all_probs
    )

    # Paper target metrics
    executed_accuracy_high = 0
    coverage_high = 0
    if len(coverage_df[coverage_df['threshold'] == config.HIGH_CONFIDENCE]) > 0:
        row = coverage_df[coverage_df['threshold'] == config.HIGH_CONFIDENCE].iloc[0]
        executed_accuracy_high = row['accuracy']
        coverage_high = row['coverage']

    # Print comprehensive results
    print("\n" + "=" * 70)
    print("                    TEST SET EVALUATION")
    print("=" * 70)

    # ICE Metrics Box
    print("\n┌" + "─" * 68 + "┐")
    print("│" + " " * 20 + "INFORMATION COEFFICIENT (IC)" + " " * 20 + "│")
    print("├" + "─" * 68 + "┤")
    print(f"│  Spearman IC:     {ic_metrics['ic_spearman']:>10.4f}    P-value: {ic_metrics['ic_pvalue']:.2e}" + " " * 18 + "│")
    print(f"│  Pearson IC:      {ic_metrics['ic_pearson']:>10.4f}" + " " * 38 + "│")
    print(f"│  Probability IC:  {ic_metrics['ic_probability']:>10.4f}" + " " * 38 + "│")
    print("├" + "─" * 68 + "┤")
    print(f"│  Rating: {ic_metrics['ic_rating']:>10}" + " " * 48 + "│")
    print(f"│  {ic_metrics['ic_description']:<66}│")
    print("└" + "─" * 68 + "┘")

    # Accuracy Metrics
    print("\n┌" + "─" * 68 + "┐")
    print("│" + " " * 22 + "ACCURACY METRICS" + " " * 30 + "│")
    print("├" + "─" * 68 + "┤")
    print(f"│  Overall Accuracy:        {accuracy:>8.4f}  ({accuracy*100:.2f}%)" + " " * 24 + "│")
    print(f"│  Direction Acc (UP/DOWN): {direction_acc:>8.4f}  ({direction_acc*100:.2f}%)" + " " * 24 + "│")
    print("└" + "─" * 68 + "┘")

    # Coverage-Accuracy Trade-off (Paper key contribution)
    print("\n┌" + "─" * 68 + "┐")
    print("│" + " " * 15 + "COVERAGE-ACCURACY TRADE-OFF (Paper)" + " " * 17 + "│")
    print("├" + "─" * 68 + "┤")
    for _, row in coverage_df.iterrows():
        threshold = row['threshold']
        acc = row['accuracy']
        cov = row['coverage']
        marker = " <-- HIGH" if threshold == config.HIGH_CONFIDENCE else " <-- MOD" if threshold == config.MODERATE_CONFIDENCE else ""
        print(f"│  Threshold {threshold:.2f}: Accuracy {acc*100:>6.2f}% | Coverage {cov*100:>6.2f}%{marker:<10}│")
    print("├" + "─" * 68 + "┤")
    print(f"│  Paper Target: 82.68% accuracy @ 11.99% coverage (high conf)        │")
    print(f"│  Your Result:  {executed_accuracy_high*100:>5.2f}% accuracy @ {coverage_high*100:>5.2f}% coverage             │")
    print("└" + "─" * 68 + "┘")

    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names))

    # Confusion matrix visualization
    cm = confusion_matrix(all_targets, all_preds)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Confusion Matrix
    im = axes[0, 0].imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=axes[0, 0])
    for i in range(num_classes):
        for j in range(num_classes):
            axes[0, 0].text(j, i, cm[i, j], ha='center', va='center', fontsize=14,
                        color='white' if cm[i, j] > cm.max()/2 else 'black')
    axes[0, 0].set_xticks(range(num_classes))
    axes[0, 0].set_yticks(range(num_classes))
    axes[0, 0].set_xticklabels(class_names)
    axes[0, 0].set_yticklabels(class_names)
    axes[0, 0].set_xlabel('Predicted', fontsize=11)
    axes[0, 0].set_ylabel('Actual', fontsize=11)
    axes[0, 0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')

    # ICE Visualization
    ic_values = [ic_metrics['ic_spearman'], ic_metrics['ic_pearson'], ic_metrics['ic_probability']]
    ic_labels = ['Spearman IC', 'Pearson IC', 'Probability IC']
    colors = ['#26a69a' if v > 0 else '#ef5350' for v in ic_values]

    bars = axes[0, 1].barh(ic_labels, ic_values, color=colors, alpha=0.8, edgecolor='white')
    axes[0, 1].axvline(x=0, color='gray', linestyle='-', linewidth=1)
    axes[0, 1].axvline(x=0.05, color='#ffd54f', linestyle='--', linewidth=1, alpha=0.7, label='Good (0.05)')
    axes[0, 1].axvline(x=0.10, color='#66bb6a', linestyle='--', linewidth=1, alpha=0.7, label='Excellent (0.10)')
    axes[0, 1].axvline(x=-0.05, color='#ffd54f', linestyle='--', linewidth=1, alpha=0.7)
    axes[0, 1].axvline(x=-0.10, color='#66bb6a', linestyle='--', linewidth=1, alpha=0.7)

    for bar, val in zip(bars, ic_values):
        axes[0, 1].text(val + 0.005 if val >= 0 else val - 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', ha='left' if val >= 0 else 'right', fontsize=10, fontweight='bold')

    axes[0, 1].set_xlim(-0.2, 0.2)
    axes[0, 1].set_xlabel('Information Coefficient Value', fontsize=11)
    axes[0, 1].set_title(f'ICE Metrics - Rating: {ic_metrics["ic_rating"]}', fontsize=12, fontweight='bold')
    axes[0, 1].legend(loc='lower right', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    # Coverage-Accuracy Trade-off Plot (Paper key visualization)
    axes[1, 0].plot(coverage_df['coverage'] * 100, coverage_df['accuracy'] * 100,
                    'b-o', linewidth=2, markersize=8, label='Model')
    axes[1, 0].axhline(y=82.68, color='g', linestyle='--', alpha=0.7, label='Paper target (82.68%)')
    axes[1, 0].axvline(x=11.99, color='g', linestyle=':', alpha=0.7, label='Paper coverage (11.99%)')

    # Mark key thresholds
    for _, row in coverage_df.iterrows():
        if row['threshold'] in [0.6, 0.8]:
            axes[1, 0].annotate(f"{row['threshold']:.1f}",
                              (row['coverage']*100, row['accuracy']*100),
                              textcoords="offset points", xytext=(5, 5), fontsize=9)

    axes[1, 0].set_xlabel('Market Coverage (%)', fontsize=11)
    axes[1, 0].set_ylabel('Accuracy on Executed Trades (%)', fontsize=11)
    axes[1, 0].set_title('Coverage-Accuracy Trade-off', fontsize=12, fontweight='bold')
    axes[1, 0].legend(loc='lower left', fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 100)
    axes[1, 0].set_ylim(40, 100)

    # Confidence Distribution
    confidences = all_probs.max(axis=1)
    axes[1, 1].hist(confidences, bins=50, alpha=0.7, color='#64b5f6', edgecolor='white')
    axes[1, 1].axvline(x=config.MODERATE_CONFIDENCE, color='#ffd54f', linestyle='--',
                       linewidth=2, label=f'Moderate ({config.MODERATE_CONFIDENCE})')
    axes[1, 1].axvline(x=config.HIGH_CONFIDENCE, color='#26a69a', linestyle='--',
                       linewidth=2, label=f'High ({config.HIGH_CONFIDENCE})')
    axes[1, 1].set_xlabel('Prediction Confidence', fontsize=11)
    axes[1, 1].set_ylabel('Count', fontsize=11)
    axes[1, 1].set_title('Confidence Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].legend(loc='upper left', fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if config.SAVE_PLOTS:
        plt.savefig(f'{config.RESULTS_DIR}/evaluation_metrics.png', dpi=150)
    if config.SHOW_PLOTS:
        plt.show()
    plt.close()

    return {
        'accuracy': accuracy,
        'direction_accuracy': direction_acc,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs,
        'ic_spearman': ic_metrics['ic_spearman'],
        'ic_pearson': ic_metrics['ic_pearson'],
        'ic_probability': ic_metrics['ic_probability'],
        'ic_rating': ic_metrics['ic_rating'],
        'ic_description': ic_metrics['ic_description'],
        'coverage_accuracy_df': coverage_df,
        'executed_accuracy_high': executed_accuracy_high,
        'coverage_high': coverage_high
    }


# =============================================================================
# BACKTESTING
# =============================================================================

def run_backtest(df: pd.DataFrame, predictions: np.ndarray,
                 probabilities: np.ndarray, config: Config,
                 eval_results: Dict = None) -> Dict:
    """Run backtest with predictions and display comprehensive ICE metrics."""

    # Align predictions with data
    start_idx = config.SEQUENCE_LENGTH + int(len(df) * (config.TRAIN_SIZE + config.VAL_SIZE))
    test_df = df.iloc[start_idx:start_idx + len(predictions)].copy()

    if len(test_df) != len(predictions):
        min_len = min(len(test_df), len(predictions))
        test_df = test_df.iloc[:min_len]
        predictions = predictions[:min_len]
        probabilities = probabilities[:min_len]

    test_df['prediction'] = predictions
    test_df['confidence'] = probabilities.max(axis=1)

    # Calculate ATR for volatility-scaled exits (use raw ATR, not normalized)
    high = test_df['high']
    low = test_df['low']
    close = test_df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    test_df['atr_raw'] = tr.rolling(14).mean()

    # Volatility-scaled exit parameters (paper-consistent)
    ATR_TAKE_PROFIT_BASE = 2.0  # Base take profit at 2x ATR
    ATR_STOP_LOSS = 1.5         # Stop loss at 1.5x ATR (fixed)

    # Confidence-scaled holding parameters
    # Higher confidence = wider take profit = longer holding
    CONF_TP_SCALE_MIN = 1.0    # At minimum confidence, TP = 1.0x base
    CONF_TP_SCALE_MAX = 2.5    # At maximum confidence, TP = 2.5x base
    EXIT_SIGNAL_CONF_THRESHOLD = 0.55  # Only exit on signal if exit confidence > this

    # Trading simulation
    capital = config.INITIAL_CAPITAL
    position = 0  # 0 = no position, 1 = long
    entry_price = 0
    entry_atr = 0
    entry_confidence = 0  # Track entry confidence for holding logic
    take_profit_price = 0
    stop_loss_price = 0

    trades = []
    portfolio_values = [capital]
    buy_hold_trades = 1  # Buy & hold is always 1 trade (buy at start)

    for i in range(1, len(test_df)):
        current_price = test_df['close'].iloc[i]
        current_high = test_df['high'].iloc[i]
        current_low = test_df['low'].iloc[i]
        pred = test_df['prediction'].iloc[i]
        conf = test_df['confidence'].iloc[i]
        current_atr = test_df['atr_raw'].iloc[i] if not pd.isna(test_df['atr_raw'].iloc[i]) else 0

        # Update portfolio value (long-only)
        if position == 1:
            portfolio_value = capital + (current_price - entry_price) * (capital / entry_price)
        else:
            portfolio_value = capital

        portfolio_values.append(portfolio_value)

        # Check volatility-scaled exits first (if in position)
        if position == 1:
            # Check take profit (using high to capture intrabar moves)
            if current_high >= take_profit_price and take_profit_price > 0:
                exit_price = take_profit_price  # Exit at target
                pnl = (exit_price - entry_price) / entry_price
                capital *= (1 + pnl - config.TRADING_FEE)
                trades.append({'type': 'take_profit', 'pnl': pnl, 'price': exit_price,
                               'time': test_df.index[i]})
                position = 0
                continue

            # Check stop loss (using low to capture intrabar moves)
            if current_low <= stop_loss_price and stop_loss_price > 0:
                exit_price = stop_loss_price  # Exit at stop
                pnl = (exit_price - entry_price) / entry_price
                capital *= (1 + pnl - config.TRADING_FEE)
                trades.append({'type': 'stop_loss', 'pnl': pnl, 'price': exit_price,
                               'time': test_df.index[i]})
                position = 0
                continue

        # Trading logic - LONG ONLY with volatility-scaled exits
        is_up_signal = (pred == 1 and config.USE_BINARY_CLASSIFICATION) or (pred == 2 and not config.USE_BINARY_CLASSIFICATION)
        is_down_signal = pred == 0

        if conf >= config.CONFIDENCE_THRESHOLD:
            if is_up_signal and position == 0 and current_atr > 0:  # UP signal and not in position
                # Scale take profit based on confidence (higher conf = wider TP = longer hold)
                # Normalize confidence to 0-1 range relative to threshold
                conf_normalized = (conf - config.CONFIDENCE_THRESHOLD) / (1.0 - config.CONFIDENCE_THRESHOLD)
                conf_normalized = max(0, min(1, conf_normalized))  # Clamp to 0-1
                tp_scale = CONF_TP_SCALE_MIN + conf_normalized * (CONF_TP_SCALE_MAX - CONF_TP_SCALE_MIN)

                # Open long with confidence-scaled take profit
                position = 1
                entry_price = current_price
                entry_atr = current_atr
                entry_confidence = conf  # Store entry confidence
                take_profit_price = entry_price + (ATR_TAKE_PROFIT_BASE * tp_scale * entry_atr)
                stop_loss_price = entry_price - (ATR_STOP_LOSS * entry_atr)
                capital *= (1 - config.TRADING_FEE)
                trades.append({'type': 'open_long', 'price': current_price, 'time': test_df.index[i],
                               'tp': take_profit_price, 'sl': stop_loss_price, 'atr': entry_atr,
                               'entry_conf': conf, 'tp_scale': tp_scale})

            elif is_down_signal and position == 1:
                # Only exit on DOWN signal if exit confidence is strong enough
                # High confidence entries should ignore weak exit signals
                if conf >= EXIT_SIGNAL_CONF_THRESHOLD:
                    # Close long on strong exit signal
                    pnl = (current_price - entry_price) / entry_price
                    capital *= (1 + pnl - config.TRADING_FEE)
                    trades.append({'type': 'close_signal', 'pnl': pnl, 'price': current_price,
                                   'time': test_df.index[i], 'exit_conf': conf})
                    position = 0
                # else: ignore weak exit signal, let TP/SL handle it

    # Close final position (only long positions in long-only mode)
    if position == 1:
        final_price = test_df['close'].iloc[-1]
        pnl = (final_price - entry_price) / entry_price
        capital *= (1 + pnl - config.TRADING_FEE)
        trades.append({'type': 'close_final', 'pnl': pnl, 'price': final_price,
                       'time': test_df.index[-1]})

    # Calculate metrics
    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    total_return = (capital - config.INITIAL_CAPITAL) / config.INITIAL_CAPITAL
    buy_hold_return = (test_df['close'].iloc[-1] - test_df['close'].iloc[0]) / test_df['close'].iloc[0]

    sharpe_ratio = np.sqrt(252 * 24) * returns.mean() / (returns.std() + 1e-10)
    max_drawdown = np.min(portfolio_values / np.maximum.accumulate(portfolio_values) - 1)

    # Calculate MAPE (using prediction confidence as proxy for price prediction)
    # Since we're doing classification, we calculate directional MAPE
    actual_directions = np.sign(test_df['close'].pct_change(config.PREDICTION_HORIZON).shift(-config.PREDICTION_HORIZON).dropna())

    # Handle binary vs 3-class prediction directions
    if config.USE_BINARY_CLASSIFICATION:
        # Binary: 0=DOWN (-1), 1=UP (+1)
        pred_directions = np.where(predictions == 1, 1, -1)
    else:
        # 3-class: 0=DOWN (-1), 1=NEUTRAL (0), 2=UP (+1)
        pred_directions = np.where(predictions == 2, 1, np.where(predictions == 0, -1, 0))

    if len(actual_directions) > len(pred_directions):
        actual_directions = actual_directions[:len(pred_directions)]
    elif len(pred_directions) > len(actual_directions):
        pred_directions = pred_directions[:len(actual_directions)]

    # Directional MAPE: percentage of wrong directions
    direction_errors = np.abs(pred_directions - actual_directions.values) / 2  # Normalize to 0-1
    mape = np.mean(direction_errors) * 100

    winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
    losing_trades = sum(1 for t in trades if t.get('pnl', 0) < 0)
    total_trades = sum(1 for t in trades if 'pnl' in t)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # Calculate average win/loss
    wins = [t['pnl'] for t in trades if t.get('pnl', 0) > 0]
    losses = [t['pnl'] for t in trades if t.get('pnl', 0) < 0]
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')

    # Get IC from eval_results if available
    ic_spearman = eval_results.get('ic_spearman', 0) if eval_results else 0
    ic_rating = eval_results.get('ic_rating', 'N/A') if eval_results else 'N/A'
    direction_accuracy = eval_results.get('direction_accuracy', 0) if eval_results else 0

    results = {
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'excess_return': total_return - buy_hold_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'mape': mape,
        'num_trades': total_trades,
        'buy_hold_trades': buy_hold_trades,
        'win_rate': win_rate,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'final_capital': capital,
        'ic_spearman': ic_spearman,
        'ic_rating': ic_rating,
        'direction_accuracy': direction_accuracy
    }

    # Print comprehensive results with ICE
    print("\n" + "═" * 70)
    print("                      BACKTEST RESULTS SUMMARY")
    print("═" * 70)

    # ICE Metrics Box (if available)
    if eval_results:
        print("\n┌" + "─" * 68 + "┐")
        print("│" + " " * 15 + "INFORMATION COEFFICIENT (ICE) METRICS" + " " * 16 + "│")
        print("├" + "─" * 68 + "┤")
        print(f"│  IC (Spearman):        {ic_spearman:>10.4f}   Rating: {ic_rating:<20}│")
        print(f"│  Direction Accuracy:   {direction_accuracy*100:>10.2f}%                                  │")
        print(f"│  MAPE:                 {mape:>10.2f}%                                  │")
        print("└" + "─" * 68 + "┘")

    # Capital & Returns
    print("\n┌" + "─" * 68 + "┐")
    print("│" + " " * 20 + "CAPITAL & RETURNS" + " " * 31 + "│")
    print("├" + "─" * 68 + "┤")
    print(f"│  Initial Capital:     ${config.INITIAL_CAPITAL:>12,.2f}" + " " * 30 + "│")
    print(f"│  Final Capital:       ${capital:>12,.2f}" + " " * 30 + "│")
    ret_sign = "+" if total_return >= 0 else ""
    print(f"│  Strategy Return:     {ret_sign}{total_return*100:>12.2f}%" + " " * 29 + "│")
    bh_sign = "+" if buy_hold_return >= 0 else ""
    print(f"│  Buy & Hold Return:   {bh_sign}{buy_hold_return*100:>12.2f}%" + " " * 29 + "│")
    ex_sign = "+" if (total_return - buy_hold_return) >= 0 else ""
    print(f"│  Excess Return:       {ex_sign}{(total_return - buy_hold_return)*100:>12.2f}%" + " " * 29 + "│")
    print("└" + "─" * 68 + "┘")

    # Risk Metrics
    print("\n┌" + "─" * 68 + "┐")
    print("│" + " " * 22 + "RISK METRICS" + " " * 34 + "│")
    print("├" + "─" * 68 + "┤")
    print(f"│  Sharpe Ratio:        {sharpe_ratio:>12.2f}" + " " * 30 + "│")
    print(f"│  Max Drawdown:        {max_drawdown*100:>12.2f}%" + " " * 29 + "│")
    print("└" + "─" * 68 + "┘")

    # Trading Statistics
    print("\n┌" + "─" * 68 + "┐")
    print("│" + " " * 20 + "TRADING STATISTICS" + " " * 30 + "│")
    print("├" + "─" * 68 + "┤")
    print(f"│  Strategy Trades:     {total_trades:>12}" + " " * 30 + "│")
    print(f"│  Buy & Hold Trades:   {buy_hold_trades:>12}" + " " * 30 + "│")
    print(f"│  Winning Trades:      {winning_trades:>12}" + " " * 30 + "│")
    print(f"│  Losing Trades:       {losing_trades:>12}" + " " * 30 + "│")
    print(f"│  Win Rate:            {win_rate*100:>12.1f}%" + " " * 29 + "│")
    print(f"│  Avg Win:             {avg_win*100:>12.3f}%" + " " * 29 + "│")
    print(f"│  Avg Loss:            {avg_loss*100:>12.3f}%" + " " * 29 + "│")
    pf_str = f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞"
    print(f"│  Profit Factor:       {pf_str:>12}" + " " * 30 + "│")
    print("└" + "─" * 68 + "┘")
    print("═" * 70)

    # Enhanced backtest visualization
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.25, wspace=0.15)

    # Portfolio value comparison
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(test_df.index[:len(portfolio_values)], portfolio_values, '#26a69a', linewidth=2,
             label=f'Strategy ({total_return*100:+.1f}%)')
    buy_hold_values = config.INITIAL_CAPITAL * (1 + test_df['close'].pct_change().fillna(0).cumsum())
    ax1.plot(test_df.index, buy_hold_values, '#64b5f6', alpha=0.7, linewidth=1.5,
             label=f'Buy & Hold ({buy_hold_return*100:+.1f}%)')
    ax1.axhline(y=config.INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')

    # Mark trades
    for trade in trades:
        if 'open' in trade['type']:
            color = '#26a69a' if 'long' in trade['type'] else '#ef5350'
            marker = '^' if 'long' in trade['type'] else 'v'
            ax1.scatter([trade['time']], [trade['price'] * config.INITIAL_CAPITAL / test_df['close'].iloc[0]],
                       color=color, marker=marker, s=50, zorder=5, alpha=0.7)

    ax1.fill_between(test_df.index[:len(portfolio_values)], config.INITIAL_CAPITAL, portfolio_values,
                    where=portfolio_values >= config.INITIAL_CAPITAL, alpha=0.2, color='#26a69a')
    ax1.fill_between(test_df.index[:len(portfolio_values)], config.INITIAL_CAPITAL, portfolio_values,
                    where=portfolio_values < config.INITIAL_CAPITAL, alpha=0.2, color='#ef5350')

    ax1.set_ylabel('Portfolio Value ($)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Backtest Results: Strategy vs Buy & Hold | Trades: {total_trades} | Sharpe: {sharpe_ratio:.2f}',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.2)
    ax1.set_facecolor('#1a1a2e')

    # Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    drawdown = (portfolio_values / np.maximum.accumulate(portfolio_values) - 1) * 100
    ax2.fill_between(test_df.index[:len(drawdown)], drawdown, 0, alpha=0.7, color='#ef5350')
    ax2.axhline(y=max_drawdown*100, color='#ff8a65', linestyle='--', linewidth=1.5,
               label=f'Max DD: {max_drawdown*100:.1f}%')
    ax2.set_ylabel('Drawdown (%)', fontsize=10)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_title('Drawdown', fontsize=11, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.2)
    ax2.set_facecolor('#1a1a2e')

    # Returns distribution
    ax3 = fig.add_subplot(gs[1, 1])
    strategy_returns = returns * 100
    n, bins, patches = ax3.hist(strategy_returns, bins=50, alpha=0.7, edgecolor='none', density=True)
    for i, patch in enumerate(patches):
        if bins[i] < 0:
            patch.set_facecolor('#ef5350')
        else:
            patch.set_facecolor('#26a69a')
    ax3.axvline(np.mean(strategy_returns), color='#ffd54f', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(strategy_returns):.3f}%')
    ax3.set_xlabel('Returns (%)', fontsize=10)
    ax3.set_ylabel('Density', fontsize=10)
    ax3.set_title('Strategy Returns Distribution', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.2)
    ax3.set_facecolor('#1a1a2e')

    # ICE & Metrics Summary
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')
    ax4.set_facecolor('#1a1a2e')

    metrics_text = f"""
    ╔════════════════════════════════════╗
    ║      ICE & PERFORMANCE METRICS     ║
    ╠════════════════════════════════════╣
    ║  Information Coefficient: {ic_spearman:>7.4f}  ║
    ║  IC Rating:              {ic_rating:>10}  ║
    ║  Direction Accuracy:     {direction_accuracy*100:>7.2f}%  ║
    ║  MAPE:                   {mape:>7.2f}%  ║
    ╠════════════════════════════════════╣
    ║  Sharpe Ratio:           {sharpe_ratio:>7.2f}   ║
    ║  Max Drawdown:           {max_drawdown*100:>7.2f}%  ║
    ║  Win Rate:               {win_rate*100:>7.1f}%  ║
    ╚════════════════════════════════════╝
    """
    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace', color='#e0e0e0',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#2d2d44', edgecolor='#64b5f6', alpha=0.9))

    # Trade statistics
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    ax5.set_facecolor('#1a1a2e')

    trade_text = f"""
    ╔════════════════════════════════════╗
    ║        TRADING STATISTICS          ║
    ╠════════════════════════════════════╣
    ║  Strategy Trades:        {total_trades:>8}  ║
    ║  Buy & Hold Trades:      {buy_hold_trades:>8}  ║
    ║  Winning:                {winning_trades:>8}  ║
    ║  Losing:                 {losing_trades:>8}  ║
    ╠════════════════════════════════════╣
    ║  Avg Win:               {avg_win*100:>7.3f}%  ║
    ║  Avg Loss:              {avg_loss*100:>7.3f}%  ║
    ║  Profit Factor:          {pf_str:>8}  ║
    ╚════════════════════════════════════╝
    """
    ax5.text(0.05, 0.95, trade_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace', color='#e0e0e0',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#2d2d44', edgecolor='#64b5f6', alpha=0.9))

    plt.tight_layout()

    if config.SAVE_PLOTS:
        plt.savefig(f'{config.RESULTS_DIR}/backtest_results.png', dpi=150, facecolor='#0d0d1a',
                    edgecolor='none', bbox_inches='tight')
    if config.SHOW_PLOTS:
        plt.show()
    plt.close()

    plt.style.use('default')

    return results


# =============================================================================
# MULTI-THRESHOLD BACKTESTING (Task 3.2)
# =============================================================================

def run_multi_threshold_backtest(df: pd.DataFrame, predictions: np.ndarray,
                                  probabilities: np.ndarray, config: Config,
                                  eval_results: Dict = None) -> Dict:
    """
    Run backtests at multiple confidence thresholds.
    This implements the paper's key insight about precision-recall trade-off.
    """

    results_by_threshold = {}
    confidences = probabilities.max(axis=1)

    print("\n" + "=" * 70)
    print("           MULTI-THRESHOLD BACKTEST ANALYSIS")
    print("=" * 70)

    for threshold in config.CONFIDENCE_THRESHOLDS:
        # Create filtered predictions
        mask = confidences >= threshold

        if mask.sum() < 10:  # Skip if too few predictions
            continue

        # Run backtest with this threshold
        config_copy = Config()
        config_copy.CONFIDENCE_THRESHOLD = threshold
        config_copy.SAVE_PLOTS = False
        config_copy.SHOW_PLOTS = False

        result = run_backtest(df, predictions, probabilities, config_copy, eval_results)

        # Calculate additional metrics
        executed_accuracy = accuracy_score(
            eval_results['targets'][mask],
            eval_results['predictions'][mask]
        ) if mask.sum() > 0 else 0

        # Calculate average net profit per trade in basis points
        if result['num_trades'] > 0:
            avg_profit_bp = result['total_return'] * 10000 / result['num_trades']
        else:
            avg_profit_bp = 0

        results_by_threshold[threshold] = {
            'threshold': threshold,
            'coverage': mask.sum() / len(predictions),
            'num_predictions': mask.sum(),
            'executed_accuracy': executed_accuracy,
            'total_return': result['total_return'],
            'sharpe_ratio': result['sharpe_ratio'],
            'num_trades': result['num_trades'],
            'win_rate': result['win_rate'],
            'max_drawdown': result['max_drawdown'],
            'avg_profit_bp': avg_profit_bp
        }

    # Create summary DataFrame
    summary_df = pd.DataFrame(results_by_threshold.values())

    # Print summary table
    print("\n┌" + "─" * 90 + "┐")
    print("│" + " " * 30 + "THRESHOLD COMPARISON" + " " * 40 + "│")
    print("├" + "─" * 90 + "┤")
    print("│  Threshold │ Coverage │ Exec Acc │ Trades │ Return   │ Sharpe │ Avg BP/Trade │")
    print("├" + "─" * 90 + "┤")

    for _, row in summary_df.iterrows():
        thresh = row['threshold']
        cov = row['coverage'] * 100
        acc = row['executed_accuracy'] * 100
        trades = row['num_trades']
        ret = row['total_return'] * 100
        sharpe = row['sharpe_ratio']
        avg_bp = row['avg_profit_bp']

        marker = " *" if thresh == config.HIGH_CONFIDENCE else ""
        print(f"│    {thresh:.2f}    │  {cov:>5.1f}%  │  {acc:>5.1f}%  │  {trades:>4}  │  {ret:>+6.2f}%  │  {sharpe:>5.2f}  │    {avg_bp:>+6.1f}   │{marker}")

    print("├" + "─" * 90 + "┤")
    print("│  * = High confidence threshold                                                          │")
    print("│  Paper target: 82.68% accuracy, 151.11 bp avg profit @ 11.99% coverage                  │")
    print("└" + "─" * 90 + "┘")

    # Visualization
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Coverage vs Accuracy
    axes[0, 0].plot(summary_df['coverage'] * 100, summary_df['executed_accuracy'] * 100,
                    'b-o', linewidth=2, markersize=10)
    axes[0, 0].axhline(y=82.68, color='g', linestyle='--', alpha=0.7, label='Paper target (82.68%)')
    for _, row in summary_df.iterrows():
        axes[0, 0].annotate(f"{row['threshold']:.2f}",
                           (row['coverage']*100, row['executed_accuracy']*100),
                           textcoords="offset points", xytext=(5, 5), fontsize=8)
    axes[0, 0].set_xlabel('Coverage (%)', fontsize=11)
    axes[0, 0].set_ylabel('Accuracy on Executed (%)', fontsize=11)
    axes[0, 0].set_title('Coverage vs Accuracy Trade-off', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Threshold vs Return
    axes[0, 1].bar(summary_df['threshold'].astype(str), summary_df['total_return'] * 100,
                   color=['#26a69a' if r > 0 else '#ef5350' for r in summary_df['total_return']], alpha=0.8)
    axes[0, 1].set_xlabel('Confidence Threshold', fontsize=11)
    axes[0, 1].set_ylabel('Total Return (%)', fontsize=11)
    axes[0, 1].set_title('Return by Threshold', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Threshold vs Avg Profit BP
    axes[1, 0].bar(summary_df['threshold'].astype(str), summary_df['avg_profit_bp'],
                   color=['#26a69a' if r > 0 else '#ef5350' for r in summary_df['avg_profit_bp']], alpha=0.8)
    axes[1, 0].axhline(y=151.11, color='g', linestyle='--', alpha=0.7, label='Paper target (151 bp)')
    axes[1, 0].set_xlabel('Confidence Threshold', fontsize=11)
    axes[1, 0].set_ylabel('Avg Profit per Trade (bp)', fontsize=11)
    axes[1, 0].set_title('Average Net Profit per Trade', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Sharpe Ratio by Threshold
    axes[1, 1].bar(summary_df['threshold'].astype(str), summary_df['sharpe_ratio'],
                   color=['#26a69a' if r > 0 else '#ef5350' for r in summary_df['sharpe_ratio']], alpha=0.8)
    axes[1, 1].set_xlabel('Confidence Threshold', fontsize=11)
    axes[1, 1].set_ylabel('Sharpe Ratio', fontsize=11)
    axes[1, 1].set_title('Risk-Adjusted Return by Threshold', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if config.SAVE_PLOTS:
        plt.savefig(f'{config.RESULTS_DIR}/multi_threshold_backtest.png', dpi=150,
                    facecolor='#0d0d1a', edgecolor='none', bbox_inches='tight')
    if config.SHOW_PLOTS:
        plt.show()
    plt.close()

    plt.style.use('default')

    return {
        'summary_df': summary_df,
        'results_by_threshold': results_by_threshold
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution pipeline with paper implementation features."""

    print("═" * 70)
    print("    PAPER IMPLEMENTATION: Confidence-Threshold Direction Prediction")
    print("═" * 70)
    print("Reference: MDPI Algorithms 18(12), 758 (November 2025)")
    print("─" * 70)

    # Set random seeds for reproducibility
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        # Keep benchmark=True for speed, don't force deterministic

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Classification mode: {'Binary (UP/DOWN)' if CONFIG.USE_BINARY_CLASSIFICATION else '3-Class'}")
    print(f"Prediction horizon: {CONFIG.PREDICTION_HORIZON} periods")
    print(f"Deadband threshold: {CONFIG.DEADBAND_THRESHOLD * 10000:.1f} basis points")

    Path(CONFIG.RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # Initialize progress tracker and checkpoint manager
    progress_tracker = ProgressTracker(CONFIG)
    checkpoint_manager = CheckpointManager(CONFIG)

    # Check for existing checkpoint to resume from
    latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
    if latest_checkpoint:
        print(f"\nFound checkpoint: {latest_checkpoint}")
        print("(Set SAVE_CHECKPOINTS=False to start fresh)")

    # Step 1: Load Data
    print("\n[1/9] Loading data...")
    df = load_data(CONFIG, match_orderbook=CONFIG.MATCH_ORDERBOOK_TIMERANGE)
    progress_tracker.mark_task_complete('1.1')

    # Step 2: Visualize raw data
    print("\n[2/9] Visualizing price data...")
    plot_price_data(df, CONFIG)

    # Step 3: Feature engineering (including macro momentum and order book)
    print("\n[3/9] Engineering features...")
    df = add_technical_indicators(df)

    # Add macro momentum features (Task 2.1)
    print("  - Adding macro momentum features...")
    df = add_macro_momentum_features(df)

    # Load order book features if available (Task 2.2)
    if CONFIG.USE_ORDERBOOK_FEATURES:
        print("  - Loading order book microstructure features...")
        df = load_orderbook_features(CONFIG, df)

    # Create targets (binary classification with deadband)
    df = create_target(df, CONFIG)
    progress_tracker.mark_task_complete('1.2')
    progress_tracker.mark_task_complete('1.4')
    progress_tracker.mark_task_complete('2.1')
    progress_tracker.mark_task_complete('2.2')

    # Step 4: Visualize features
    print("\n[4/9] Visualizing features...")
    plot_features(df, CONFIG)
    plot_target_distribution(df, CONFIG)

    # Step 5: Prepare data & train
    print("\n[5/9] Preparing data and training model...")
    train_loader, val_loader, test_loader, scaler, num_features = prepare_data(df, CONFIG)

    print(f"  - Using {num_features} features (after correlation filter)")

    model = DirectionLSTM(CONFIG, input_size=num_features)
    print(f"  - Model output: {model.num_classes} classes")

    history = train_model(model, train_loader, val_loader, CONFIG, device)
    plot_training_history(history, CONFIG)

    # Save checkpoint after training
    if CONFIG.SAVE_CHECKPOINTS:
        checkpoint_path = checkpoint_manager.save_checkpoint(
            epoch=len(history['train_loss']),
            model=model,
            optimizer=None,  # Not needed for inference
            scaler=scaler,
            history=history,
            config=CONFIG,
            best_val_loss=min(history['val_loss']),
            best_ic=max(history['val_ic'])
        )
        progress_tracker.set_checkpoint(checkpoint_path)
        print(f"  - Checkpoint saved: {checkpoint_path}")

    progress_tracker.mark_task_complete('4.1')

    # Step 6: Evaluate with coverage-accuracy analysis
    print("\n[6/9] Evaluating model with confidence threshold analysis...")
    model = model.to(device)
    eval_results = evaluate_model(model, test_loader, device, CONFIG)
    progress_tracker.mark_task_complete('1.3')
    progress_tracker.mark_task_complete('3.1')
    progress_tracker.mark_task_complete('3.3')

    # Step 7: Single threshold backtest
    print("\n[7/9] Running backtest at default threshold...")
    backtest_results = run_backtest(
        df, eval_results['predictions'],
        eval_results['probabilities'], CONFIG,
        eval_results=eval_results
    )

    # Step 8: Multi-threshold backtest analysis (Task 3.2)
    print("\n[8/9] Running multi-threshold backtest analysis...")
    multi_threshold_results = run_multi_threshold_backtest(
        df, eval_results['predictions'],
        eval_results['probabilities'], CONFIG,
        eval_results=eval_results
    )
    progress_tracker.mark_task_complete('3.2')

    # Step 9: Save progress
    print("\n[9/9] Saving progress...")
    progress_tracker.mark_task_complete('4.2')
    progress_tracker.save_progress()

    # Final Summary with ICE metrics prominently displayed
    print("\n" + "═" * 70)
    print("                    PIPELINE COMPLETE - FINAL SUMMARY")
    print("═" * 70)

    # Paper Comparison Box
    print("\n┌" + "─" * 68 + "┐")
    print("│" + " " * 18 + "COMPARISON WITH PAPER TARGETS" + " " * 21 + "│")
    print("├" + "─" * 68 + "┤")

    # Get high confidence results
    executed_acc_high = eval_results.get('executed_accuracy_high', 0)
    coverage_high = eval_results.get('coverage_high', 0)

    # Find best threshold from multi-threshold results
    best_threshold_results = None
    if 'summary_df' in multi_threshold_results:
        summary_df = multi_threshold_results['summary_df']
        if len(summary_df) > 0:
            # Find threshold with best accuracy while maintaining >10% coverage
            filtered = summary_df[summary_df['coverage'] > 0.1]
            if len(filtered) > 0:
                best_idx = filtered['executed_accuracy'].idxmax()
                best_threshold_results = filtered.loc[best_idx]

    print("│  Metric                    │   Paper   │   Yours   │   Status            │")
    print("├" + "─" * 68 + "┤")

    # Direction accuracy at high confidence
    paper_acc = 82.68
    your_acc = executed_acc_high * 100
    acc_status = "ACHIEVED" if your_acc >= paper_acc else f"{(paper_acc - your_acc):.1f}% gap"
    print(f"│  Executed Trade Accuracy   │  {paper_acc:>5.2f}%   │  {your_acc:>5.2f}%   │  {acc_status:<18} │")

    # Coverage at high confidence
    paper_cov = 11.99
    your_cov = coverage_high * 100
    cov_status = "ACHIEVED" if your_cov >= paper_cov * 0.8 else f"{(paper_cov - your_cov):.1f}% gap"
    print(f"│  Market Coverage (High)    │  {paper_cov:>5.2f}%   │  {your_cov:>5.2f}%   │  {cov_status:<18} │")

    # Average profit per trade
    paper_bp = 151.11
    if best_threshold_results is not None:
        your_bp = best_threshold_results['avg_profit_bp']
    else:
        your_bp = backtest_results.get('total_return', 0) * 10000 / max(backtest_results.get('num_trades', 1), 1)
    bp_status = "ACHIEVED" if your_bp >= paper_bp else f"{(paper_bp - your_bp):.1f}bp gap"
    print(f"│  Avg Net Profit/Trade (bp) │  {paper_bp:>5.1f}    │  {your_bp:>+6.1f}   │  {bp_status:<18} │")

    print("└" + "─" * 68 + "┘")

    # ICE Summary Box
    print("\n┌" + "─" * 68 + "┐")
    print("│" + " " * 18 + "INFORMATION COEFFICIENT (ICE) SUMMARY" + " " * 13 + "│")
    print("├" + "─" * 68 + "┤")
    ic = eval_results.get('ic_spearman', 0)
    ic_rating = eval_results.get('ic_rating', 'N/A')
    ic_desc = eval_results.get('ic_description', '')

    if ic > 0.05:
        status = "TRADEABLE"
    elif ic > 0.02:
        status = "MARGINAL"
    else:
        status = "NOT RECOMMENDED"

    print(f"│  IC Value:     {ic:>8.4f}                                           │")
    print(f"│  IC Rating:    {ic_rating:<12}                                     │")
    print(f"│  Status:       {status:<15}                                  │")
    print(f"│  {ic_desc:<66}│")
    print("└" + "─" * 68 + "┘")

    # Performance Summary
    print("\n┌" + "─" * 68 + "┐")
    print("│" + " " * 22 + "PERFORMANCE SUMMARY" + " " * 27 + "│")
    print("├" + "─" * 68 + "┤")
    dir_acc = eval_results.get('direction_accuracy', 0)
    sharpe = backtest_results.get('sharpe_ratio', 0)
    mape = backtest_results.get('mape', 0)
    total_ret = backtest_results.get('total_return', 0)
    excess_ret = backtest_results.get('excess_return', 0)

    print(f"│  Direction Accuracy:  {dir_acc*100:>8.2f}%                               │")
    print(f"│  Sharpe Ratio:        {sharpe:>8.2f}                                 │")
    print(f"│  MAPE:                {mape:>8.2f}%                               │")
    print(f"│  Strategy Return:     {total_ret*100:>+8.2f}%                               │")
    print(f"│  Excess vs Buy&Hold:  {excess_ret*100:>+8.2f}%                               │")
    print("└" + "─" * 68 + "┘")

    # Trade Statistics
    print("\n┌" + "─" * 68 + "┐")
    print("│" + " " * 22 + "TRADE STATISTICS" + " " * 30 + "│")
    print("├" + "─" * 68 + "┤")
    num_trades = backtest_results.get('num_trades', 0)
    bh_trades = backtest_results.get('buy_hold_trades', 1)
    win_rate = backtest_results.get('win_rate', 0)
    print(f"│  Strategy Trades:     {num_trades:>8}                                 │")
    print(f"│  Buy & Hold Trades:   {bh_trades:>8}                                 │")
    print(f"│  Win Rate:            {win_rate*100:>8.1f}%                               │")
    print("└" + "─" * 68 + "┘")

    # Optimal Threshold Recommendation
    if best_threshold_results is not None:
        print("\n┌" + "─" * 68 + "┐")
        print("│" + " " * 18 + "OPTIMAL THRESHOLD RECOMMENDATION" + " " * 18 + "│")
        print("├" + "─" * 68 + "┤")
        opt_thresh = best_threshold_results['threshold']
        opt_acc = best_threshold_results['executed_accuracy'] * 100
        opt_cov = best_threshold_results['coverage'] * 100
        opt_bp = best_threshold_results['avg_profit_bp']
        print(f"│  Recommended Threshold:   {opt_thresh:.2f}                                  │")
        print(f"│  Expected Accuracy:       {opt_acc:.2f}%                                │")
        print(f"│  Market Coverage:         {opt_cov:.2f}%                                │")
        print(f"│  Avg Profit/Trade:        {opt_bp:+.1f} bp                              │")
        print("└" + "─" * 68 + "┘")

    # Files saved
    print(f"\nModel saved to: {CONFIG.MODEL_SAVE_PATH}")
    print(f"Results saved to: {CONFIG.RESULTS_DIR}/")
    print(f"Progress tracked in: {CONFIG.PROGRESS_FILE}")
    print("═" * 70)

    return {
        'model': model,
        'scaler': scaler,
        'history': history,
        'eval_results': eval_results,
        'backtest_results': backtest_results,
        'multi_threshold_results': multi_threshold_results
    }


if __name__ == "__main__":
    results = main()
