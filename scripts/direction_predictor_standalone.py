"""
Standalone Direction Prediction Script
======================================
A concise, modular script for cryptocurrency direction prediction using LSTM.

Features:
- CSV data input or live data fetch
- Configurable training parameters
- Data and feature visualization
- LSTM-based direction classification
- Comprehensive backtesting

Author: Crypto AI Predictor
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import spearmanr, pearsonr

# =============================================================================
# CONFIGURATION - EDIT THESE PARAMETERS
# =============================================================================

@dataclass
class Config:
    """All configurable parameters in one place."""

    # DATA SOURCE (set ONE of these)
    CSV_FILE: Optional[str] = r"C:\crypto\ver7\dataset\BTCUSDT\2026-01-11_11-07-31_50000candles\data.csv"  # e.g., "data/BTCUSDT_1h.csv" or None for live fetch
    SYMBOL: str = "BTCUSDT"
    TIMEFRAME: str = "1h"

    # DATA PARAMETERS
    TRAIN_SIZE: float = 0.7
    VAL_SIZE: float = 0.15
    TEST_SIZE: float = 0.15
    SEQUENCE_LENGTH: int = 168  # 168 hours = 1 week lookback
    PREDICTION_HORIZON: int = 24  # Predict 24 hours ahead

    # MODEL PARAMETERS
    HIDDEN_SIZE: int = 128
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.2
    BIDIRECTIONAL: bool = True
    USE_ATTENTION: bool = True

    # TRAINING PARAMETERS
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.0001
    WEIGHT_DECAY: float = 1e-4
    MAX_EPOCHS: int = 100
    EARLY_STOPPING_PATIENCE: int = 15
    GRADIENT_CLIP: float = 1.0

    # CLASSIFICATION THRESHOLDS
    UP_THRESHOLD: float = 0.005  # 0.5% for UP signal
    DOWN_THRESHOLD: float = -0.005  # -0.5% for DOWN signal

    # BACKTESTING
    INITIAL_CAPITAL: float = 10000.0
    TRADING_FEE: float = 0.001  # 0.1%
    CONFIDENCE_THRESHOLD: float = 0.6  # Min confidence to trade

    # OUTPUT
    MODEL_SAVE_PATH: str = "models/direction_model.pt"
    RESULTS_DIR: str = "results"
    SHOW_PLOTS: bool = True
    SAVE_PLOTS: bool = True


# Initialize config
CONFIG = Config()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(config: Config) -> pd.DataFrame:
    """Load data from CSV or fetch from exchange."""

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
    minus_dm = low.diff().abs()
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
    """Create target variable for direction prediction."""

    df = df.copy()

    # Future return
    df['future_return'] = df['close'].pct_change(config.PREDICTION_HORIZON).shift(-config.PREDICTION_HORIZON)

    # Classification target: 0=DOWN, 1=NEUTRAL, 2=UP
    df['target'] = 1  # Default neutral
    df.loc[df['future_return'] > config.UP_THRESHOLD, 'target'] = 2  # UP
    df.loc[df['future_return'] < config.DOWN_THRESHOLD, 'target'] = 0  # DOWN

    # Binary target for simpler analysis
    df['target_binary'] = (df['future_return'] > 0).astype(int)

    return df


def get_feature_columns() -> List[str]:
    """Get list of feature columns (stationary only)."""
    return [
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
    """Plot target variable distribution."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Target class distribution
    target_counts = df['target'].value_counts().sort_index()
    labels = ['DOWN', 'NEUTRAL', 'UP']
    colors = ['red', 'gray', 'green']
    axes[0].bar(labels, [target_counts.get(i, 0) for i in range(3)], color=colors, alpha=0.7)
    axes[0].set_ylabel('Count')
    axes[0].set_title('Target Class Distribution')
    for i, v in enumerate([target_counts.get(i, 0) for i in range(3)]):
        axes[0].text(i, v + 50, f'{v}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=9)

    # Future returns distribution
    returns = df['future_return'].dropna()
    axes[1].hist(returns, bins=100, alpha=0.7, edgecolor='black')
    axes[1].axvline(config.UP_THRESHOLD, color='g', linestyle='--', label=f'UP threshold: {config.UP_THRESHOLD}')
    axes[1].axvline(config.DOWN_THRESHOLD, color='r', linestyle='--', label=f'DOWN threshold: {config.DOWN_THRESHOLD}')
    axes[1].set_xlabel(f'{config.PREDICTION_HORIZON}h Future Returns')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].set_title('Future Returns Distribution')

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


class DirectionLSTM(nn.Module):
    """LSTM model for direction classification."""

    def __init__(self, config: Config, input_size: int):
        super().__init__()
        self.config = config

        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0,
            batch_first=True,
            bidirectional=config.BIDIRECTIONAL
        )

        lstm_output_size = config.HIDDEN_SIZE * (2 if config.BIDIRECTIONAL else 1)

        # Attention
        self.use_attention = config.USE_ATTENTION
        if self.use_attention:
            self.attention = AttentionLayer(lstm_output_size)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, config.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_SIZE, 3)  # 3 classes: DOWN, NEUTRAL, UP
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_norm(x)

        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        if self.use_attention:
            context, _ = self.attention(lstm_out)
        else:
            context = lstm_out[:, -1, :]  # Last timestep

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

def prepare_data(df: pd.DataFrame, config: Config) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """Prepare data for training."""

    # Get features and target
    feature_cols = [c for c in get_feature_columns() if c in df.columns]
    df_clean = df[feature_cols + ['target']].dropna()

    print(f"Clean data shape: {df_clean.shape}")
    print(f"Features: {len(feature_cols)}")

    # Split data (temporal - no shuffling)
    n = len(df_clean)
    train_end = int(n * config.TRAIN_SIZE)
    val_end = int(n * (config.TRAIN_SIZE + config.VAL_SIZE))

    train_df = df_clean.iloc[:train_end]
    val_df = df_clean.iloc[train_end:val_end]
    test_df = df_clean.iloc[val_end:]

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Scale features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(train_df[feature_cols])
    X_val = scaler.transform(val_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    y_train = train_df['target'].values
    y_val = val_df['target'].values
    y_test = test_df['target'].values

    # Create datasets
    train_dataset = DirectionDataset(X_train, y_train, config.SEQUENCE_LENGTH)
    val_dataset = DirectionDataset(X_val, y_val, config.SEQUENCE_LENGTH)
    test_dataset = DirectionDataset(X_test, y_test, config.SEQUENCE_LENGTH)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, scaler


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                config: Config, device: torch.device) -> Dict:
    """Train the model."""

    model = model.to(device)

    # Class weights for imbalanced data
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0

    print("\nTraining started...")
    print("-" * 60)

    for epoch in range(config.MAX_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = criterion(logits, y)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_targets, all_preds)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            Path(config.MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))

    return history


def plot_training_history(history: Dict, config: Config):
    """Plot training history."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

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

    # Probability-based IC (if probs provided)
    # Use the probability of the predicted class as confidence
    if probs is not None:
        # Create a signed confidence score: positive for UP, negative for DOWN
        confidence_scores = np.zeros(len(predictions))
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
    """Evaluate model on test set with ICE metrics."""

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

    # Direction accuracy (UP vs DOWN only)
    mask = (all_targets != 1) & (all_preds != 1)
    if mask.sum() > 0:
        direction_acc = accuracy_score(all_targets[mask], all_preds[mask])
    else:
        direction_acc = 0

    # Calculate Information Coefficient (ICE)
    ic_metrics = calculate_ic(all_preds, all_targets, all_probs)

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

    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds,
                               target_names=['DOWN', 'NEUTRAL', 'UP']))

    # Confusion matrix visualization
    cm = confusion_matrix(all_targets, all_preds)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion Matrix
    im = axes[0].imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=axes[0])
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, cm[i, j], ha='center', va='center', fontsize=14,
                        color='white' if cm[i, j] > cm.max()/2 else 'black')
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_xticklabels(['DOWN', 'NEUTRAL', 'UP'])
    axes[0].set_yticklabels(['DOWN', 'NEUTRAL', 'UP'])
    axes[0].set_xlabel('Predicted', fontsize=11)
    axes[0].set_ylabel('Actual', fontsize=11)
    axes[0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')

    # ICE Visualization
    ic_values = [ic_metrics['ic_spearman'], ic_metrics['ic_pearson'], ic_metrics['ic_probability']]
    ic_labels = ['Spearman IC', 'Pearson IC', 'Probability IC']
    colors = ['#26a69a' if v > 0 else '#ef5350' for v in ic_values]

    bars = axes[1].barh(ic_labels, ic_values, color=colors, alpha=0.8, edgecolor='white')
    axes[1].axvline(x=0, color='gray', linestyle='-', linewidth=1)
    axes[1].axvline(x=0.05, color='#ffd54f', linestyle='--', linewidth=1, alpha=0.7, label='Good threshold (0.05)')
    axes[1].axvline(x=0.10, color='#66bb6a', linestyle='--', linewidth=1, alpha=0.7, label='Excellent threshold (0.10)')
    axes[1].axvline(x=-0.05, color='#ffd54f', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].axvline(x=-0.10, color='#66bb6a', linestyle='--', linewidth=1, alpha=0.7)

    # Add value labels on bars
    for bar, val in zip(bars, ic_values):
        axes[1].text(val + 0.005 if val >= 0 else val - 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', ha='left' if val >= 0 else 'right', fontsize=10, fontweight='bold')

    axes[1].set_xlim(-0.2, 0.2)
    axes[1].set_xlabel('Information Coefficient Value', fontsize=11)
    axes[1].set_title(f'ICE Metrics - Rating: {ic_metrics["ic_rating"]}', fontsize=12, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=8)
    axes[1].grid(True, alpha=0.3, axis='x')

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
        'ic_description': ic_metrics['ic_description']
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

    # Trading simulation
    capital = config.INITIAL_CAPITAL
    position = 0  # 0 = no position, 1 = long, -1 = short
    entry_price = 0

    trades = []
    portfolio_values = [capital]
    buy_hold_trades = 1  # Buy & hold is always 1 trade (buy at start)

    for i in range(1, len(test_df)):
        current_price = test_df['close'].iloc[i]
        pred = test_df['prediction'].iloc[i]
        conf = test_df['confidence'].iloc[i]

        # Update portfolio value
        if position == 1:
            portfolio_value = capital + (current_price - entry_price) * (capital / entry_price)
        elif position == -1:
            portfolio_value = capital + (entry_price - current_price) * (capital / entry_price)
        else:
            portfolio_value = capital

        portfolio_values.append(portfolio_value)

        # Trading logic
        if conf >= config.CONFIDENCE_THRESHOLD:
            if pred == 2 and position != 1:  # UP signal
                if position == -1:  # Close short
                    pnl = (entry_price - current_price) / entry_price
                    capital *= (1 + pnl - config.TRADING_FEE)
                    trades.append({'type': 'close_short', 'pnl': pnl, 'price': current_price,
                                   'time': test_df.index[i]})

                # Open long
                position = 1
                entry_price = current_price
                capital *= (1 - config.TRADING_FEE)
                trades.append({'type': 'open_long', 'price': current_price, 'time': test_df.index[i]})

            elif pred == 0 and position != -1:  # DOWN signal
                if position == 1:  # Close long
                    pnl = (current_price - entry_price) / entry_price
                    capital *= (1 + pnl - config.TRADING_FEE)
                    trades.append({'type': 'close_long', 'pnl': pnl, 'price': current_price,
                                   'time': test_df.index[i]})

                # Open short
                position = -1
                entry_price = current_price
                capital *= (1 - config.TRADING_FEE)
                trades.append({'type': 'open_short', 'price': current_price, 'time': test_df.index[i]})

    # Close final position
    if position != 0:
        final_price = test_df['close'].iloc[-1]
        if position == 1:
            pnl = (final_price - entry_price) / entry_price
        else:
            pnl = (entry_price - final_price) / entry_price
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
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution pipeline."""

    print("═" * 70)
    print("         DIRECTION PREDICTION - STANDALONE SCRIPT")
    print("═" * 70)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    Path(CONFIG.RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # Step 1: Load Data
    print("\n[1/7] Loading data...")
    df = load_data(CONFIG)

    # Step 2: Visualize raw data
    print("\n[2/7] Visualizing price data...")
    plot_price_data(df, CONFIG)

    # Step 3: Feature engineering
    print("\n[3/7] Engineering features...")
    df = add_technical_indicators(df)
    df = create_target(df, CONFIG)

    # Step 4: Visualize features
    print("\n[4/7] Visualizing features...")
    plot_features(df, CONFIG)
    plot_target_distribution(df, CONFIG)

    # Step 5: Prepare data & train
    print("\n[5/7] Preparing data and training model...")
    train_loader, val_loader, test_loader, scaler = prepare_data(df, CONFIG)

    feature_cols = [c for c in get_feature_columns() if c in df.columns]
    model = DirectionLSTM(CONFIG, input_size=len(feature_cols))

    history = train_model(model, train_loader, val_loader, CONFIG, device)
    plot_training_history(history, CONFIG)

    # Step 6: Evaluate
    print("\n[6/7] Evaluating model...")
    model = model.to(device)
    eval_results = evaluate_model(model, test_loader, device, CONFIG)

    # Step 7: Backtest
    print("\n[7/7] Running backtest...")
    backtest_results = run_backtest(
        df, eval_results['predictions'],
        eval_results['probabilities'], CONFIG,
        eval_results=eval_results  # Pass eval_results for ICE metrics
    )

    # Final Summary with ICE metrics prominently displayed
    print("\n" + "═" * 70)
    print("                    PIPELINE COMPLETE - FINAL SUMMARY")
    print("═" * 70)

    # ICE Summary Box
    print("\n┌" + "─" * 68 + "┐")
    print("│" + " " * 18 + "INFORMATION COEFFICIENT (ICE) SUMMARY" + " " * 13 + "│")
    print("├" + "─" * 68 + "┤")
    ic = eval_results.get('ic_spearman', 0)
    ic_rating = eval_results.get('ic_rating', 'N/A')
    ic_desc = eval_results.get('ic_description', '')

    if ic > 0.05:
        status = "✓ TRADEABLE"
    elif ic > 0.02:
        status = "◐ MARGINAL"
    else:
        status = "✗ NOT RECOMMENDED"

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

    # Files saved
    print(f"\n📁 Model saved to: {CONFIG.MODEL_SAVE_PATH}")
    print(f"📊 Results saved to: {CONFIG.RESULTS_DIR}/")
    print("═" * 70)

    return {
        'model': model,
        'scaler': scaler,
        'history': history,
        'eval_results': eval_results,
        'backtest_results': backtest_results
    }


if __name__ == "__main__":
    results = main()
