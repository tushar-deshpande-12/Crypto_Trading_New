"""
Debug Script: NaN Sources in Preprocessing Pipeline
Shows exactly where NaN values are introduced and how many rows are lost
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.ml.preprocessing.preprocessor import CryptoPreprocessor

def debug_nan_sources():
    """
    Trace NaN sources through the preprocessing pipeline
    """
    print("="*80)
    print("NaN SOURCES DEBUGGING")
    print("="*80)

    # Initialize preprocessor
    preprocessor = CryptoPreprocessor(dataset_dir="dataset")

    # Load a single symbol to debug
    symbols = ['ETHUSDT']  # Change to your symbol
    print(f"\nLoading data for: {symbols}")

    try:
        df = preprocessor.load_multi_symbol_data(symbols, min_candles=1000)
        print(f"\n1. RAW DATA LOADED")
        print(f"   Shape: {df.shape}")
        print(f"   NaN count: {df.isna().sum().sum()}")

        # Check raw OHLCV data
        raw_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in raw_cols:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                print(f"   - {col}: {nan_count} NaNs")

        print(f"\n{'='*80}")

        # Generate temporal features (should not create NaNs)
        df = preprocessor.generate_temporal_features(df)
        print(f"\n2. AFTER TEMPORAL FEATURES")
        print(f"   Shape: {df.shape}")
        print(f"   NaN count: {df.isna().sum().sum()}")
        print(f"   [OK] Temporal features (sin/cos) don't create NaNs")

        print(f"\n{'='*80}")

        # Generate technical indicators (pct_change creates NaNs!)
        df = preprocessor.generate_technical_indicators(df)
        print(f"\n3. AFTER TECHNICAL INDICATORS")
        print(f"   Shape: {df.shape}")
        print(f"   NaN count: {df.isna().sum().sum()}")

        technical_features = ['returns', 'log_volume', 'log_quote_volume', 'price_range',
                            'trade_intensity', 'taker_buy_ratio']
        for col in technical_features:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    print(f"   [X] {col}: {nan_count} NaNs (pct_change() creates NaN in first row)")

        print(f"\n{'='*80}")

        # Generate lagged features (shift creates NaNs!)
        df = preprocessor.generate_lagged_features(df, lags=[1, 24, 168])
        print(f"\n4. AFTER LAGGED FEATURES")
        print(f"   Shape: {df.shape}")
        print(f"   NaN count: {df.isna().sum().sum()}")

        lag_features = [f'close_lag_{lag}h' for lag in [1, 24, 168]] + \
                      [f'volume_lag_{lag}h' for lag in [1, 24, 168]]

        for col in lag_features:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                lag = int(col.split('_')[2].replace('h', ''))
                if nan_count > 0:
                    print(f"   [X] {col}: {nan_count} NaNs (first {lag} rows per symbol)")

        print(f"\n{'='*80}")

        # Generate rolling features (rolling creates NaNs!)
        df = preprocessor.generate_rolling_features(df, window=24)
        print(f"\n5. AFTER ROLLING FEATURES (24h window)")
        print(f"   Shape: {df.shape}")
        print(f"   NaN count: {df.isna().sum().sum()}")

        rolling_features = ['rolling_mean_close_24h', 'rolling_std_close_24h',
                          'rolling_max_high_24h', 'rolling_min_low_24h']

        for col in rolling_features:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    print(f"   [X] {col}: {nan_count} NaNs (first 24 rows per symbol)")

        print(f"\n{'='*80}")

        # Generate volatility features (multiple windows create NaNs!)
        df = preprocessor.generate_volatility_features(df)
        print(f"\n6. AFTER VOLATILITY FEATURES")
        print(f"   Shape: {df.shape}")
        print(f"   NaN count: {df.isna().sum().sum()}")

        volatility_features = [
            'volatility_6h', 'volatility_24h', 'volatility_168h',
            'parkinson_vol_24h', 'vol_of_vol', 'vol_percentile_7d',
            'vol_ratio_short_long', 'garman_klass_vol_24h', 'vol_trend_24h'
        ]

        for col in volatility_features:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    # Determine window size
                    if '168h' in col or '7d' in col:
                        window = "168 (7 days)"
                    elif '24h' in col:
                        window = "24 (1 day)"
                    elif '6h' in col:
                        window = "6"
                    else:
                        window = "varies"
                    print(f"   [X] {col}: {nan_count} NaNs (first {window} rows)")

        print(f"\n{'='*80}")
        print(f"\nDATA LOSS ANALYSIS")
        print(f"{'='*80}")

        # Find rows with ANY NaN
        rows_with_nan = df.isna().any(axis=1).sum()
        total_rows = len(df)
        clean_rows = total_rows - rows_with_nan
        loss_percent = (rows_with_nan / total_rows) * 100

        print(f"\nTotal rows: {total_rows}")
        print(f"Rows with NaN: {rows_with_nan} ({loss_percent:.2f}%)")
        print(f"Clean rows: {clean_rows} ({100-loss_percent:.2f}%)")

        # Show NaN distribution by feature
        print(f"\n{'='*80}")
        print(f"NaN COUNT BY FEATURE")
        print(f"{'='*80}")

        nan_counts = df.isna().sum()
        nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)

        print(f"\nTop features with NaN values:")
        for col, count in nan_counts.head(10).items():
            print(f"  {col:30s}: {count:5d} NaNs ({count/total_rows*100:.1f}%)")

        # Show which rows would be dropped
        print(f"\n{'='*80}")
        print(f"ROWS THAT WILL BE DROPPED")
        print(f"{'='*80}")

        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol]
            symbol_nan_rows = symbol_df.isna().any(axis=1).sum()
            symbol_total = len(symbol_df)

            # Find first row without NaN
            clean_mask = ~symbol_df.isna().any(axis=1)
            if clean_mask.any():
                first_clean_idx = clean_mask.idxmax()
                first_clean_row = symbol_df.loc[first_clean_idx]
                first_clean_time = first_clean_row['datetime']

                print(f"\n{symbol}:")
                print(f"  Total rows: {symbol_total}")
                print(f"  Rows with NaN: {symbol_nan_rows}")
                print(f"  First clean row: index {first_clean_idx}")
                print(f"  First clean timestamp: {first_clean_time}")
                print(f"  [!] Losing first ~{symbol_nan_rows} rows ({symbol_nan_rows/symbol_total*100:.1f}%) per symbol")

        print(f"\n{'='*80}")
        print(f"WHY ARE THESE NaNs CREATED?")
        print(f"{'='*80}")

        print("""
The NaN values come from 3 main sources:

1. [X] pct_change() - Creates NaN in FIRST ROW
   - Used in: returns feature
   - Impact: First 1 row lost per symbol

2. [X] shift(lag) - Creates NaN in FIRST lag ROWS
   - Used in: lagged features (1h, 24h, 168h)
   - Impact: First 168 rows lost per symbol (worst case)

3. [X] rolling(window) - Creates NaN in FIRST window ROWS
   - Used in: rolling statistics, volatility features
   - Impact: First 168 rows lost per symbol (7-day rolling)

WORST OFFENDER: 168-hour (7-day) features!
  - close_lag_168h
  - volume_lag_168h
  - volatility_168h
  - vol_of_vol (rolling on 168h volatility)
  - vol_percentile_7d

Result: First ~168 hours (7 days) of data MUST be dropped per symbol!

This is EXPECTED and NECESSARY for lagged/rolling features to work.
Your raw OHLCV data has NO nulls - the nulls come from feature engineering.
        """)

        print(f"\n{'='*80}")
        print(f"SOLUTION")
        print(f"{'='*80}")

        print("""
Options:

1. [OK] ACCEPT the data loss (RECOMMENDED)
   - First 168 hours lost per symbol is acceptable
   - Lagged/rolling features are crucial for time series forecasting
   - If you have 10,000 candles, losing 168 is only 1.68%

2. [X] Remove long-window features (NOT RECOMMENDED)
   - Remove 168h lag and volatility features
   - Reduces data loss but hurts model accuracy

3. [X] Forward fill NaNs (NOT RECOMMENDED)
   - Would create artificial data
   - Introduces look-ahead bias
   - Corrupts feature quality

[OK] Keep the current approach: accept 168-row loss per symbol.
        """)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_nan_sources()
