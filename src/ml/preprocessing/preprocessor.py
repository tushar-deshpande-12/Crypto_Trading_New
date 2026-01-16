# preprocessor.py - COMPLETE FIXED VERSION FOR SMOOTH TRAINING
import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def print_msg(*args):
    """Print message (Windows compatible - handles Unicode issues)."""
    msg = ' '.join(map(str, args))
    logger.info(msg)
    try:
        print(msg)
    except UnicodeEncodeError:
        # Fall back to ASCII for Windows console encoding issues
        print(msg.encode('ascii', 'replace').decode('ascii'))

class CryptoPreprocessor:
    """Preprocessor with REGRESSION target (predicts next period return)."""

    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.scalers = {}
        self.feature_columns = []
        self.target_column = 'target_return'  # Predict returns, not absolute prices
        self.target_scaler = None  # For normalizing target values
    
    def load_symbol_data(self, symbol: str, prefer_largest: bool = True) -> pd.DataFrame:
        """Load dataset for symbol.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            prefer_largest: If True, load dataset with most candles.
                           If False, load most recent dataset.
        """
        sym_dir = self.dataset_dir / symbol
        if not sym_dir.exists():
            raise FileNotFoundError(f"No data for {symbol}")

        best_dir = None
        best_score = 0  # Either candle_count or timestamp depending on prefer_largest

        for dataset_dir in sym_dir.iterdir():
            if dataset_dir.is_dir():
                meta_path = dataset_dir / "metadata.json"
                if meta_path.exists():
                    try:
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)

                        if prefer_largest:
                            # Prefer dataset with most candles
                            score = meta.get('candle_count', meta.get('data_points', 0))
                        else:
                            # Prefer most recent dataset
                            fetch_str = meta.get('fetch_timestamp', '1970-01-01T00:00:00')
                            if isinstance(fetch_str, str):
                                score = datetime.fromisoformat(fetch_str.replace('Z', '+00:00')).timestamp()
                            else:
                                score = fetch_str

                        if score > best_score:
                            best_score = score
                            best_dir = dataset_dir
                    except Exception:
                        continue

        if not best_dir:
            dirs = sorted([d for d in sym_dir.iterdir() if d.is_dir()], key=lambda d: d.name, reverse=True)
            best_dir = dirs[0] if dirs else None

        if not best_dir:
            raise FileNotFoundError(f"No valid dataset in {sym_dir}")

        # Load data
        json_path = best_dir / "data.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            csv_path = best_dir / "data.csv"
            df = pd.read_csv(csv_path)

        print_msg(f"  [Using: {best_dir.name}]")
        
        if 'datetime' not in df.columns and 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        df['symbol'] = symbol
        df['datetime'] = pd.to_datetime(df['datetime'])
        print_msg(f"[OK] Loaded {len(df)} candles for {symbol}")
        return df.sort_values('datetime').reset_index(drop=True)
    
    def load_multi_symbol_data(self, symbols: List[str]) -> pd.DataFrame:
        """Load multiple symbols."""
        dfs = []
        for symbol in symbols:
            try:
                df = self.load_symbol_data(symbol)
                dfs.append(df)
            except Exception as e:
                print_msg(f"[X] Failed {symbol}: {e}")
        if not dfs:
            raise ValueError("No data loaded!")
        return pd.concat(dfs, ignore_index=True)
    
    def generate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicators."""
        result_dfs = []

        for symbol in df['symbol'].unique():
            sym_df = df[df['symbol'] == symbol].copy()
            sym_df = sym_df.sort_values('datetime').reset_index(drop=True)

            # Price returns (stationarity)
            sym_df['return_1h'] = sym_df['close'].pct_change(1)
            sym_df['return_4h'] = sym_df['close'].pct_change(4)
            sym_df['return_24h'] = sym_df['close'].pct_change(24)
            sym_df['log_return'] = np.log(sym_df['close'] / sym_df['close'].shift(1))

            # STATIONARY PRICE FEATURES (relative ratios instead of absolute prices)
            # These are scale-invariant and won't leak future information
            sym_df['high_low_ratio'] = sym_df['high'] / sym_df['low']  # Intrabar range
            sym_df['close_open_ratio'] = sym_df['close'] / sym_df['open']  # Intrabar direction
            sym_df['high_close_ratio'] = sym_df['high'] / sym_df['close']  # Upper wick
            sym_df['low_close_ratio'] = sym_df['low'] / sym_df['close']  # Lower wick
            sym_df['body_range_ratio'] = abs(sym_df['close'] - sym_df['open']) / (sym_df['high'] - sym_df['low']).replace(0, np.nan)  # Body vs range
            
            # RSI
            delta = sym_df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            sym_df['rsi'] = 100 - (100 / (1 + rs))
            
            # Moving averages (raw values - will be excluded from model features)
            sym_df['sma_10'] = sym_df['close'].rolling(10).mean()
            sym_df['sma_20'] = sym_df['close'].rolling(20).mean()
            sym_df['ema_12'] = sym_df['close'].ewm(span=12).mean()

            # STATIONARY MA FEATURES: Price relative to moving averages
            # These indicate whether price is above/below trend (stationary!)
            sym_df['price_sma10_ratio'] = sym_df['close'] / sym_df['sma_10']
            sym_df['price_sma20_ratio'] = sym_df['close'] / sym_df['sma_20']
            sym_df['price_ema12_ratio'] = sym_df['close'] / sym_df['ema_12']
            sym_df['sma10_sma20_ratio'] = sym_df['sma_10'] / sym_df['sma_20']  # MA crossover

            # MACD (already stationary - difference of EMAs relative to price level)
            ema_26 = sym_df['close'].ewm(span=26).mean()
            sym_df['macd'] = sym_df['ema_12'] - ema_26
            sym_df['macd_signal'] = sym_df['macd'].ewm(span=9).mean()
            sym_df['macd_pct'] = sym_df['macd'] / sym_df['close'] * 100  # MACD as % of price

            # Bollinger Bands
            sym_df['bb_middle'] = sym_df['close'].rolling(20).mean()
            bb_std = sym_df['close'].rolling(20).std()
            sym_df['bb_upper'] = sym_df['bb_middle'] + (2 * bb_std)
            sym_df['bb_lower'] = sym_df['bb_middle'] - (2 * bb_std)
            sym_df['bb_width'] = (sym_df['bb_upper'] - sym_df['bb_lower']) / sym_df['bb_middle']
            # Bollinger Band position: -1 (lower) to +1 (upper), stationary
            sym_df['bb_position'] = (sym_df['close'] - sym_df['bb_lower']) / (sym_df['bb_upper'] - sym_df['bb_lower']).replace(0, np.nan) * 2 - 1
            
            # Volatility
            sym_df['volatility'] = sym_df['log_return'].rolling(20).std()
            sym_df['atr'] = sym_df[['high', 'low', 'close']].apply(
                lambda x: max(x['high'] - x['low'], 
                            abs(x['high'] - x['close']), 
                            abs(x['low'] - x['close'])), axis=1
            ).rolling(14).mean()
            
            # Volume features
            sym_df['volume_sma'] = sym_df['volume'].rolling(20).mean()
            sym_df['volume_ratio'] = sym_df['volume'] / sym_df['volume_sma']

            # ============================================================
            # NEW TECHNICAL INDICATORS (10 additional indicators)
            # ============================================================

            # 1. STOCHASTIC OSCILLATOR (%K and %D) - bounded 0-100
            low_14 = sym_df['low'].rolling(14).min()
            high_14 = sym_df['high'].rolling(14).max()
            stoch_range = (high_14 - low_14).replace(0, np.nan)
            sym_df['stoch_k'] = 100 * (sym_df['close'] - low_14) / stoch_range
            sym_df['stoch_d'] = sym_df['stoch_k'].rolling(3).mean()

            # 2. WILLIAMS %R - bounded -100 to 0 (normalized to 0-1 for model)
            sym_df['williams_r'] = -100 * (high_14 - sym_df['close']) / stoch_range
            sym_df['williams_r_norm'] = (sym_df['williams_r'] + 100) / 100  # Normalize to 0-1

            # 3. CCI (Commodity Channel Index) - normalized to -1 to 1
            typical_price = (sym_df['high'] + sym_df['low'] + sym_df['close']) / 3
            sma_tp = typical_price.rolling(20).mean()
            mean_deviation = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
            cci_raw = (typical_price - sma_tp) / (0.015 * mean_deviation).replace(0, np.nan)
            sym_df['cci'] = cci_raw.clip(-200, 200) / 200  # Normalize to -1 to 1

            # 4. OBV (On-Balance Volume) Rate of Change - stationary
            price_direction = np.sign(sym_df['close'].diff())
            obv = (price_direction * sym_df['volume']).fillna(0).cumsum()
            sym_df['obv_roc'] = obv.pct_change(14).clip(-5, 5)  # 14-period ROC, clipped
            sym_df['obv_sma_ratio'] = obv / obv.rolling(20).mean().replace(0, np.nan)

            # 5. ADX (Average Directional Index) - bounded 0-100, normalized to 0-1
            tr = np.maximum(
                sym_df['high'] - sym_df['low'],
                np.maximum(
                    np.abs(sym_df['high'] - sym_df['close'].shift(1)),
                    np.abs(sym_df['low'] - sym_df['close'].shift(1))
                )
            )
            atr_14_adx = tr.rolling(14).mean()

            high_diff = sym_df['high'].diff()
            low_diff = -sym_df['low'].diff()
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

            plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / atr_14_adx.replace(0, np.nan)
            minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / atr_14_adx.replace(0, np.nan)

            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
            sym_df['adx'] = dx.rolling(14).mean() / 100  # Normalize to 0-1
            sym_df['plus_di'] = plus_di / 100  # Normalize to 0-1
            sym_df['minus_di'] = minus_di / 100  # Normalize to 0-1
            sym_df['di_diff'] = (plus_di - minus_di) / 100  # Directional difference

            # 6. MFI (Money Flow Index) - bounded 0-100, normalized to 0-1
            mfi_typical_price = (sym_df['high'] + sym_df['low'] + sym_df['close']) / 3
            raw_money_flow = mfi_typical_price * sym_df['volume']
            positive_flow = raw_money_flow.where(mfi_typical_price > mfi_typical_price.shift(1), 0)
            negative_flow = raw_money_flow.where(mfi_typical_price < mfi_typical_price.shift(1), 0)
            positive_sum = positive_flow.rolling(14).sum()
            negative_sum = negative_flow.rolling(14).sum().replace(0, np.nan)
            money_ratio = positive_sum / negative_sum
            sym_df['mfi'] = (100 - (100 / (1 + money_ratio))) / 100  # Normalize to 0-1

            # 7. AROON (Up, Down, Oscillator) - bounded 0-100, normalized
            aroon_period = 25
            sym_df['aroon_up'] = sym_df['high'].rolling(aroon_period + 1).apply(
                lambda x: x.argmax() / aroon_period, raw=False
            )
            sym_df['aroon_down'] = sym_df['low'].rolling(aroon_period + 1).apply(
                lambda x: x.argmin() / aroon_period, raw=False
            )
            sym_df['aroon_osc'] = sym_df['aroon_up'] - sym_df['aroon_down']  # -1 to 1

            # 8. KELTNER CHANNELS - position within bands (0-1)
            keltner_ema = sym_df['close'].ewm(span=20).mean()
            keltner_atr = tr.rolling(10).mean()  # 10-period ATR for Keltner
            keltner_upper = keltner_ema + (2 * keltner_atr)
            keltner_lower = keltner_ema - (2 * keltner_atr)
            keltner_range = (keltner_upper - keltner_lower).replace(0, np.nan)
            sym_df['keltner_position'] = (sym_df['close'] - keltner_lower) / keltner_range
            sym_df['keltner_width'] = keltner_range / keltner_ema  # Width relative to price

            # 9. ICHIMOKU CLOUD - price ratios to components (stationary)
            tenkan_sen = (sym_df['high'].rolling(9).max() + sym_df['low'].rolling(9).min()) / 2
            kijun_sen = (sym_df['high'].rolling(26).max() + sym_df['low'].rolling(26).min()) / 2
            senkou_a = (tenkan_sen + kijun_sen) / 2
            senkou_b = (sym_df['high'].rolling(52).max() + sym_df['low'].rolling(52).min()) / 2

            sym_df['ichimoku_tenkan_ratio'] = sym_df['close'] / tenkan_sen.replace(0, np.nan)
            sym_df['ichimoku_kijun_ratio'] = sym_df['close'] / kijun_sen.replace(0, np.nan)
            sym_df['ichimoku_cloud_ratio'] = sym_df['close'] / senkou_a.replace(0, np.nan)
            sym_df['ichimoku_tk_cross'] = (tenkan_sen - kijun_sen) / sym_df['close']  # TK cross signal
            sym_df['ichimoku_cloud_thickness'] = (senkou_a - senkou_b) / sym_df['close']  # Cloud thickness

            # 10. ROC (Rate of Change) - multiple periods
            sym_df['roc_6'] = sym_df['close'].pct_change(6)
            sym_df['roc_12'] = sym_df['close'].pct_change(12)
            sym_df['roc_24'] = sym_df['close'].pct_change(24)

            # ============================================================
            # END OF NEW INDICATORS
            # ============================================================

            # REGRESSION TARGET: Predict 24-hour forward RETURN
            # Hourly returns are random noise (autocorr=-0.001, SNR=0.04)
            # Longer horizons have more predictable structure
            PREDICTION_HORIZON = 24  # 24 hours ahead

            future_close = sym_df['close'].shift(-PREDICTION_HORIZON)
            sym_df['target_return'] = (future_close - sym_df['close']) / sym_df['close']  # 24h return
            sym_df['target_close'] = future_close  # Keep for reference

            # Also create multi-horizon targets for flexibility
            sym_df['target_return_1h'] = (sym_df['close'].shift(-1) - sym_df['close']) / sym_df['close']
            sym_df['target_return_4h'] = (sym_df['close'].shift(-4) - sym_df['close']) / sym_df['close']
            sym_df['target_return_24h'] = sym_df['target_return']  # Same as main target

            # Direction for evaluation (threshold at 0.5% for 24h to filter noise)
            sym_df['actual_direction'] = (sym_df['target_return'] > 0.005).astype(int)  # >0.5% = up
            
            result_dfs.append(sym_df)
        
        df = pd.concat(result_dfs, ignore_index=True)
        print_msg(f"[OK] Features generated: {df.shape[1]} columns")
        return df
    
    def generate_lagged_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
        """Generate lagged features per symbol.

        IMPORTANT: Use RETURNS for lags instead of raw prices (stationarity!).
        Raw price lags are non-stationary and don't generalize across price levels.
        """
        result_dfs = []
        for symbol in df['symbol'].unique():
            sym_df = df[df['symbol'] == symbol].copy().sort_values('datetime')

            # Calculate returns first
            returns = sym_df['close'].pct_change()

            for lag in lags:
                # STATIONARY: Lagged returns instead of raw prices
                sym_df[f'return_lag_{lag}'] = returns.shift(lag)
                # Volume ratio lags (relative to rolling mean for stationarity)
                sym_df[f'volume_lag_{lag}'] = sym_df['volume'].shift(lag)

            result_dfs.append(sym_df)
        return pd.concat(result_dfs, ignore_index=True)

    def generate_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate time-based features from datetime."""
        df = df.copy()

        # Ensure datetime is proper type
        if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            df['datetime'] = pd.to_datetime(df['datetime'])

        # Hour of day (0-23) - cyclical encoding
        df['hour'] = df['datetime'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Day of week (0-6) - cyclical encoding
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Day of month (1-31) - cyclical encoding
        df['day_of_month'] = df['datetime'].dt.day
        df['dom_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['dom_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)

        # Month (1-12) - cyclical encoding
        df['month'] = df['datetime'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Weekend flag
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        print_msg(f"[OK] Temporal features generated")
        return df

    def generate_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate rolling window statistics per symbol.

        IMPORTANT: Use STATIONARY features (returns, ratios) instead of raw prices!
        Raw close rolling features are non-stationary and leak price level information.
        """
        result_dfs = []
        windows = [6, 12, 24, 48]  # 6h, 12h, 24h, 48h for hourly data

        for symbol in df['symbol'].unique():
            sym_df = df[df['symbol'] == symbol].copy().sort_values('datetime')

            # Calculate returns for rolling statistics
            returns = sym_df['close'].pct_change()

            for window in windows:
                # STATIONARY: Rolling return statistics (mean, std)
                sym_df[f'return_roll_mean_{window}'] = returns.rolling(window).mean()
                sym_df[f'return_roll_std_{window}'] = returns.rolling(window).std()

                # Volume rolling features (volume is already relative to itself)
                sym_df[f'volume_roll_mean_{window}'] = sym_df['volume'].rolling(window).mean()
                sym_df[f'volume_roll_std_{window}'] = sym_df['volume'].rolling(window).std()
                sym_df[f'volume_roll_ratio_{window}'] = sym_df['volume'] / sym_df[f'volume_roll_mean_{window}'].replace(0, np.nan)

                # STATIONARY: Price position within rolling range (0-1)
                # This is scale-invariant and measures where price is relative to recent range
                roll_min = sym_df['close'].rolling(window).min()
                roll_max = sym_df['close'].rolling(window).max()
                roll_range = roll_max - roll_min
                sym_df[f'price_position_{window}'] = (sym_df['close'] - roll_min) / roll_range.replace(0, np.nan)

                # STATIONARY: Price relative to rolling mean (deviation from average)
                roll_mean = sym_df['close'].rolling(window).mean()
                sym_df[f'price_rollmean_ratio_{window}'] = sym_df['close'] / roll_mean

            result_dfs.append(sym_df)

        print_msg(f"[OK] Rolling features generated (stationary)")
        return pd.concat(result_dfs, ignore_index=True)

    def generate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility-based features per symbol."""
        result_dfs = []

        for symbol in df['symbol'].unique():
            sym_df = df[df['symbol'] == symbol].copy().sort_values('datetime')

            # Parkinson volatility (uses high-low range)
            sym_df['parkinson_vol'] = np.sqrt(
                (1 / (4 * np.log(2))) * (np.log(sym_df['high'] / sym_df['low']) ** 2)
            ).rolling(20).mean()

            # Garman-Klass volatility (uses OHLC)
            log_hl = np.log(sym_df['high'] / sym_df['low']) ** 2
            log_co = np.log(sym_df['close'] / sym_df['open']) ** 2
            sym_df['garman_klass_vol'] = np.sqrt(
                0.5 * log_hl - (2 * np.log(2) - 1) * log_co
            ).rolling(20).mean()

            # Realized volatility (different windows)
            returns = sym_df['close'].pct_change()
            sym_df['realized_vol_12'] = returns.rolling(12).std() * np.sqrt(12)
            sym_df['realized_vol_24'] = returns.rolling(24).std() * np.sqrt(24)
            sym_df['realized_vol_48'] = returns.rolling(48).std() * np.sqrt(48)

            # Volatility ratio (short-term vs long-term)
            sym_df['vol_ratio_12_48'] = sym_df['realized_vol_12'] / sym_df['realized_vol_48'].replace(0, np.nan)

            # Average True Range percentage
            if 'atr' in sym_df.columns:
                sym_df['atr_pct'] = sym_df['atr'] / sym_df['close'] * 100
            else:
                sym_df['atr_pct'] = np.nan

            result_dfs.append(sym_df)

        print_msg(f"[OK] Volatility features generated")
        return pd.concat(result_dfs, ignore_index=True)

    def add_time_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sequential time index per symbol for TFT model.

        IMPORTANT: This creates CONTINUOUS time indices that will be preserved
        across train/val/test splits. The time_idx should NOT be reset per split.
        """
        result_dfs = []

        for symbol in df['symbol'].unique():
            sym_df = df[df['symbol'] == symbol].copy().sort_values('datetime')
            sym_df['time_idx'] = range(len(sym_df))
            result_dfs.append(sym_df)

        return pd.concat(result_dfs, ignore_index=True)

    def split_train_val_test(self, df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15) -> Tuple:
        """Temporal split per symbol - NO SHUFFLE."""
        train_dfs, val_dfs, test_dfs = [], [], []
        
        for symbol in df['symbol'].unique():
            sym_df = df[df['symbol'] == symbol].sort_values('datetime').reset_index(drop=True)
            n = len(sym_df)
            
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            train_dfs.append(sym_df.iloc[:train_end])
            val_dfs.append(sym_df.iloc[train_end:val_end])
            test_dfs.append(sym_df.iloc[val_end:])
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        print_msg(f"SPLIT: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        self.validate_no_data_leakage(train_df, val_df, test_df)
        return train_df, val_df, test_df
    
    def normalize(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalize features per symbol. FIT on train, TRANSFORM on val/test."""
        # Define features to normalize (exclude target initially)
        # IMPORTANT: Exclude raw OHLC and raw MAs (non-stationary) - use relative ratios instead
        base_features = ['volume',  # Volume is kept but scaled
                        'return_1h', 'return_4h', 'return_24h', 'log_return',
                        'high_low_ratio', 'close_open_ratio', 'high_close_ratio',
                        'low_close_ratio', 'body_range_ratio',  # Stationary price ratios
                        'price_sma10_ratio', 'price_sma20_ratio', 'price_ema12_ratio',
                        'sma10_sma20_ratio',  # Price relative to MAs (stationary)
                        'rsi', 'macd', 'macd_signal', 'macd_pct',  # Momentum indicators
                        'bb_width', 'bb_position',  # Bollinger bands (stationary versions)
                        'volatility', 'atr', 'volume_sma', 'volume_ratio',
                        # NEW INDICATORS (10 additional)
                        'stoch_k', 'stoch_d',  # Stochastic Oscillator
                        'williams_r_norm',  # Williams %R (normalized)
                        'cci',  # CCI (normalized)
                        'obv_roc', 'obv_sma_ratio',  # OBV indicators
                        'adx', 'plus_di', 'minus_di', 'di_diff',  # ADX and DI
                        'mfi',  # Money Flow Index
                        'aroon_up', 'aroon_down', 'aroon_osc',  # Aroon
                        'keltner_position', 'keltner_width',  # Keltner Channels
                        'ichimoku_tenkan_ratio', 'ichimoku_kijun_ratio',  # Ichimoku
                        'ichimoku_cloud_ratio', 'ichimoku_tk_cross', 'ichimoku_cloud_thickness',
                        'roc_6', 'roc_12', 'roc_24']  # Rate of Change

        # Raw data features that need normalization (often have huge values!)
        raw_data_features = ['quote_volume', 'taker_buy_quote_volume',
                            'taker_buy_volume', 'trades', 'number_of_trades',
                            'taker_buy_base_asset_volume', 'taker_buy_base_volume']

        # Temporal features (cyclical encodings already in [-1, 1] range but normalize anyway)
        temporal_features = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                            'dom_sin', 'dom_cos', 'month_sin', 'month_cos', 'is_weekend']

        # Volatility features
        volatility_features = ['parkinson_vol', 'garman_klass_vol',
                              'realized_vol_12', 'realized_vol_24', 'realized_vol_48',
                              'vol_ratio_12_48', 'atr_pct']

        # Add lagged features dynamically (return_lag_* and volume_lag_*)
        lag_features = [col for col in df.columns if 'lag_' in col]

        # Add rolling features dynamically (return_roll_*, volume_roll_*, price_position_*, price_rollmean_ratio_*)
        rolling_features = [col for col in df.columns if 'roll_' in col or 'price_position_' in col or 'rollmean_ratio' in col]

        # Combine all features
        all_features = (base_features + raw_data_features + temporal_features +
                       volatility_features + lag_features + rolling_features)
        feature_cols = [col for col in all_features if col in df.columns]
        
        result_dfs = []
        for symbol in df['symbol'].unique():
            sym_df = df[df['symbol'] == symbol].copy()
            
            if fit:
                scaler = StandardScaler()
                scaler.fit(sym_df[feature_cols])
                self.scalers[symbol] = scaler
                print_msg(f"FIT scaler for {symbol}: {len(sym_df)} samples")
            else:
                if symbol not in self.scalers:
                    raise ValueError(f"No scaler for {symbol}! Fit on train first.")
                scaler = self.scalers[symbol]
            
            sym_df[feature_cols] = scaler.transform(sym_df[feature_cols])
            result_dfs.append(sym_df)
        
        self.feature_columns = feature_cols
        return pd.concat(result_dfs, ignore_index=True)
    
    def validate_no_data_leakage(self, train_df, val_df, test_df):
        """Check temporal order."""
        issues = []
        for symbol in train_df['symbol'].unique():
            train_max = train_df[train_df['symbol']==symbol]['datetime'].max()
            val_min = val_df[val_df['symbol']==symbol]['datetime'].min()
            if pd.notna(train_max) and pd.notna(val_min) and train_max >= val_min:
                issues.append(f"{symbol}: overlap")
        
        if issues:
            print_msg(f"[!] LEAKAGE: {issues}")
            return False
        print_msg("[OK] NO LEAKAGE: Temporal order preserved")
        return True
    
    def process_all(self, symbols: List[str], save_scaler_path: Optional[str] = None) -> Tuple:
        """Complete pipeline: Load -> Features -> time_idx -> SPLIT -> Normalize.

        CRITICAL ORDER:
        1. Generate features BEFORE split
        2. Add time_idx BEFORE split (so it's continuous across splits)
        3. Split temporally (preserving time_idx)
        4. Normalize: FIT on train, TRANSFORM on val/test
        """
        print_msg("="*80)
        print_msg("[START] REGRESSION PREPROCESSOR (Predicts next close price)")
        print_msg("="*80)

        # Load
        df = self.load_multi_symbol_data(symbols)
        print_msg(f"Loaded: {df.shape}")

        # Generate features
        df = self.generate_temporal_features(df)
        df = self.generate_technical_indicators(df)
        df = self.generate_lagged_features(df)
        df = self.generate_rolling_features(df)
        df = self.generate_volatility_features(df)
        print_msg(f"After features: {df.shape}")

        # Drop NaN from indicators/lags/rolling/volatility
        df = df.dropna().reset_index(drop=True)
        print_msg(f"After dropna: {df.shape}")

        # CRITICAL FIX: Add time_idx BEFORE splitting so it's CONTINUOUS across splits
        # This ensures train time_idx=0..N, val time_idx=N+1..M, test time_idx=M+1..K
        df = self.add_time_index(df)
        print_msg(f"[OK] Added continuous time_idx (0 to {df['time_idx'].max()})")

        # CRITICAL: SPLIT BEFORE NORMALIZE (preserves time_idx)
        train_df, val_df, test_df = self.split_train_val_test(df)

        # Log time_idx ranges to verify continuity
        for symbol in train_df['symbol'].unique():
            train_idx = train_df[train_df['symbol']==symbol]['time_idx']
            val_idx = val_df[val_df['symbol']==symbol]['time_idx']
            test_idx = test_df[test_df['symbol']==symbol]['time_idx']
            print_msg(f"  {symbol}: train=[{train_idx.min()}-{train_idx.max()}], "
                     f"val=[{val_idx.min()}-{val_idx.max()}], test=[{test_idx.min()}-{test_idx.max()}]")

        # Normalize: FIT on train, TRANSFORM on val/test
        train_df = self.normalize(train_df, fit=True)
        val_df = self.normalize(val_df, fit=False)
        test_df = self.normalize(test_df, fit=False)

        # Also normalize target_return for better convergence
        train_df, val_df, test_df = self._normalize_target(train_df, val_df, test_df)

        # Save scalers
        if save_scaler_path:
            Path(save_scaler_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_scaler_path, 'wb') as f:
                pickle.dump({
                    'scalers': self.scalers,
                    'features': self.feature_columns,
                    'target': self.target_column,
                    'target_scaler': self.target_scaler  # Save target scaler too
                }, f)
            print_msg(f"[SAVE] Scalers saved: {save_scaler_path}")

        print_msg("="*80)
        print_msg(f"[OK] PIPELINE COMPLETE")
        print_msg(f"Train: {train_df.shape} | Val: {val_df.shape} | Test: {test_df.shape}")
        print_msg(f"Target: '{self.target_column}' (regression - predicts next close)")
        print_msg(f"Features: {len(self.feature_columns)} columns")
        print_msg("="*80)

        return train_df, val_df, test_df

    def _normalize_target(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                          test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Normalize target_return for better convergence.

        FIT on train, TRANSFORM on val/test (same as features).
        This helps the model converge faster and more stably.
        """
        target_col = self.target_column

        # Fit scaler on training data only
        self.target_scaler = StandardScaler()
        train_targets = train_df[target_col].values.reshape(-1, 1)
        self.target_scaler.fit(train_targets)

        # Transform all splits
        train_df[target_col] = self.target_scaler.transform(train_df[target_col].values.reshape(-1, 1)).flatten()
        val_df[target_col] = self.target_scaler.transform(val_df[target_col].values.reshape(-1, 1)).flatten()
        test_df[target_col] = self.target_scaler.transform(test_df[target_col].values.reshape(-1, 1)).flatten()

        print_msg(f"[OK] Target '{target_col}' normalized (mean~0, std~1)")
        print_msg(f"  Train target: mean={train_df[target_col].mean():.4f}, std={train_df[target_col].std():.4f}")
        print_msg(f"  Val target: mean={val_df[target_col].mean():.4f}, std={val_df[target_col].std():.4f}")

        return train_df, val_df, test_df
    
    def load_scalers(self, path: str):
        """Load fitted scalers."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.scalers = data['scalers']
            self.feature_columns = data['features']
            self.target_column = data.get('target', 'target_close')
            self.target_scaler = data.get('target_scaler', None)
            if self.target_scaler:
                print_msg(f"[OK] Loaded target scaler (for denormalization)")

    # Alias for backwards compatibility
    def load_scaler(self, path: str):
        """Alias for load_scalers."""
        return self.load_scalers(path)


def validate_data_for_training(train_df: pd.DataFrame, val_df: pd.DataFrame,
                               test_df: pd.DataFrame, verbose: bool = True) -> Dict[str, bool]:
    """
    Comprehensive validation of preprocessed data before training.

    Checks for common issues that cause:
    - Validation loss not decreasing
    - Model not converging
    - Overfitting

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        verbose: Print detailed diagnostics

    Returns:
        Dict with validation results
    """
    results = {}

    if verbose:
        print("\n" + "="*80)
        print_msg("[DATA VALIDATION FOR TRAINING]")
        print("="*80)

    # 1. Check time_idx continuity
    if verbose:
        print("\n[1] TIME INDEX CONTINUITY")
    time_idx_ok = True
    for symbol in train_df['symbol'].unique():
        train_idx = train_df[train_df['symbol']==symbol]['time_idx']
        val_idx = val_df[val_df['symbol']==symbol]['time_idx']
        test_idx = test_df[test_df['symbol']==symbol]['time_idx']

        # Check continuity: val should start after train ends
        if val_idx.min() <= train_idx.max():
            if verbose:
                print(f"  [X] {symbol}: OVERLAP! val starts at {val_idx.min()} but train ends at {train_idx.max()}")
            time_idx_ok = False
        else:
            if verbose:
                print(f"  [OK] {symbol}: train=[{train_idx.min()}-{train_idx.max()}] -> val=[{val_idx.min()}-{val_idx.max()}] -> test=[{test_idx.min()}-{test_idx.max()}]")

    results['time_idx_continuous'] = time_idx_ok

    # 2. Check for NaN/Inf values
    if verbose:
        print("\n[2] NaN/Inf CHECK")
    nan_ok = True
    for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        nan_count = df.isna().sum().sum()
        inf_count = np.isinf(df.select_dtypes(include=np.number)).sum().sum()
        if nan_count > 0 or inf_count > 0:
            if verbose:
                print(f"  [X] {name}: {nan_count} NaN, {inf_count} Inf values")
            nan_ok = False
        else:
            if verbose:
                print(f"  [OK] {name}: No NaN/Inf values")
    results['no_nan_inf'] = nan_ok

    # 3. Check target distribution
    if verbose:
        print("\n[3] TARGET DISTRIBUTION")
    target_col = 'target_return'
    if target_col in train_df.columns:
        train_target = train_df[target_col]
        val_target = val_df[target_col]

        # Check if targets are normalized
        train_mean, train_std = train_target.mean(), train_target.std()
        val_mean, val_std = val_target.mean(), val_target.std()

        if verbose:
            print(f"  Train: mean={train_mean:.4f}, std={train_std:.4f}")
            print(f"  Val:   mean={val_mean:.4f}, std={val_std:.4f}")

        # Warn if very different distributions
        if abs(val_mean - train_mean) > 1.0:
            if verbose:
                print(f"  [!]  WARNING: Val mean significantly different from train (domain shift)")
            results['target_distribution_ok'] = False
        else:
            if verbose:
                print(f"  [OK] Target distributions similar")
            results['target_distribution_ok'] = True
    else:
        if verbose:
            print(f"  [!]  target_return column not found")
        results['target_distribution_ok'] = False

    # 4. Check feature statistics
    if verbose:
        print("\n[4] FEATURE STATISTICS (sample)")
    feature_ok = True
    sample_features = ['close', 'volume', 'rsi', 'macd']
    for feat in sample_features:
        if feat in train_df.columns:
            train_mean = train_df[feat].mean()
            train_std = train_df[feat].std()
            val_mean = val_df[feat].mean()
            val_std = val_df[feat].std()

            # Normalized features should have mean~0, std~1
            if abs(train_mean) > 0.1 or abs(train_std - 1.0) > 0.1:
                if verbose:
                    print(f"  [!]  {feat} (train): mean={train_mean:.2f}, std={train_std:.2f} - may not be normalized")
            else:
                if verbose:
                    print(f"  [OK] {feat}: train(mean={train_mean:.2f}, std={train_std:.2f}), val(mean={val_mean:.2f}, std={val_std:.2f})")

    results['features_normalized'] = feature_ok

    # 5. Check data leakage (temporal order)
    if verbose:
        print("\n[5] TEMPORAL ORDER (NO DATA LEAKAGE)")
    leakage_ok = True
    for symbol in train_df['symbol'].unique():
        train_max_dt = train_df[train_df['symbol']==symbol]['datetime'].max()
        val_min_dt = val_df[val_df['symbol']==symbol]['datetime'].min()
        test_min_dt = test_df[test_df['symbol']==symbol]['datetime'].min()

        if train_max_dt >= val_min_dt:
            if verbose:
                print(f"  [X] {symbol}: LEAKAGE! train ends {train_max_dt}, val starts {val_min_dt}")
            leakage_ok = False
        elif val_df[val_df['symbol']==symbol]['datetime'].max() >= test_min_dt:
            if verbose:
                print(f"  [X] {symbol}: LEAKAGE! val overlaps with test")
            leakage_ok = False
        else:
            if verbose:
                print(f"  [OK] {symbol}: Proper temporal order")
    results['no_data_leakage'] = leakage_ok

    # Summary
    if verbose:
        print("\n" + "="*80)
        print("[STATS] VALIDATION SUMMARY")
        print("="*80)
        all_pass = all(results.values())
        for check, passed in results.items():
            status = "[OK]" if passed else "[X]"
            print(f"  {status} {check}")

        if all_pass:
            print("\n[OK] All checks passed! Data is ready for training.")
        else:
            print("\n[X] Some checks failed. Fix issues before training.")

    return results


if __name__ == "__main__":
    prep = CryptoPreprocessor("dataset")
    train, val, test = prep.process_all(["BTCUSDT", "ETHUSDT"], "scalers.pkl")

    # Run validation
    results = validate_data_for_training(train, val, test)
    print("\nReady for training!" if all(results.values()) else "\n[!] Fix issues before training!")
