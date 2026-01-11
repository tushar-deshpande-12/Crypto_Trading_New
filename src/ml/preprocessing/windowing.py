"""
Sliding Window Creation for Time Series
Creates context windows for TFT training
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


def create_sliding_windows(
    df: pd.DataFrame,
    context_length: int = 40000,
    horizon: int = 10,
    stride: int = 100,
    target_columns: List[str] = ['close'],
    verbose: bool = True
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Create sliding windows from time series data

    Args:
        df: Input DataFrame (must be sorted by datetime)
        context_length: Length of input sequence (40,000 for 40k hours)
        horizon: Length of prediction sequence (10 for 10 hours)
        stride: Step size between windows (100 = windows overlap by context_length - 100)
        target_columns: Columns to predict (default: ['close'])
        verbose: Print progress information

    Returns:
        Tuple of (context_windows, target_windows)
        - context_windows: List of DataFrames with context_length rows
        - target_windows: List of DataFrames with horizon rows
    """
    if verbose:
        print(f"\n[WINDOWING] Creating sliding windows...")
        print(f"[WINDOWING]   - Input shape: {df.shape}")
        print(f"[WINDOWING]   - Context length: {context_length}")
        print(f"[WINDOWING]   - Prediction horizon: {horizon}")
        print(f"[WINDOWING]   - Stride: {stride}")

    # Verify DataFrame is sorted
    if 'datetime' in df.columns:
        if not df['datetime'].is_monotonic_increasing:
            print(f"[WINDOWING] ⚠ Warning: DataFrame not sorted by datetime, sorting...")
            df = df.sort_values('datetime').reset_index(drop=True)

    context_windows = []
    target_windows = []

    total_length_needed = context_length + horizon
    n_samples = len(df)

    if n_samples < total_length_needed:
        raise ValueError(
            f"Insufficient data: need {total_length_needed} samples "
            f"(context={context_length} + horizon={horizon}), "
            f"but only have {n_samples}"
        )

    # Calculate number of windows
    max_start_idx = n_samples - total_length_needed
    n_windows = (max_start_idx // stride) + 1

    if verbose:
        print(f"[WINDOWING]   - Total samples: {n_samples}")
        print(f"[WINDOWING]   - Max start index: {max_start_idx}")
        print(f"[WINDOWING]   - Number of windows: {n_windows}")

    # Create windows
    for i, start_idx in enumerate(range(0, max_start_idx + 1, stride)):
        if verbose and (i + 1) % 10 == 0:
            print(f"[WINDOWING]   - Creating window {i+1}/{n_windows}...")

        # Context window (input)
        context_end = start_idx + context_length
        context_df = df.iloc[start_idx:context_end].copy()

        # Target window (output)
        target_start = context_end
        target_end = target_start + horizon
        target_df = df.iloc[target_start:target_end].copy()

        # Verify window sizes
        if len(context_df) != context_length:
            if verbose:
                print(f"[WINDOWING] ⚠ Skipping window {i}: context size {len(context_df)} != {context_length}")
            continue

        if len(target_df) != horizon:
            if verbose:
                print(f"[WINDOWING] ⚠ Skipping window {i}: target size {len(target_df)} != {horizon}")
            continue

        context_windows.append(context_df)
        target_windows.append(target_df)

    if verbose:
        print(f"[WINDOWING] ✓ Created {len(context_windows)} windows")

        if len(context_windows) > 0:
            # Show sample window info
            print(f"[WINDOWING]   - Sample window 0:")
            print(f"[WINDOWING]     - Context: rows {0} to {context_length-1}")
            print(f"[WINDOWING]     - Target:  rows {context_length} to {context_length+horizon-1}")

            if 'datetime' in df.columns:
                print(f"[WINDOWING]     - Context dates: {context_windows[0]['datetime'].iloc[0]} "
                      f"to {context_windows[0]['datetime'].iloc[-1]}")
                print(f"[WINDOWING]     - Target dates:  {target_windows[0]['datetime'].iloc[0]} "
                      f"to {target_windows[0]['datetime'].iloc[-1]}")

    return context_windows, target_windows


def create_windows_per_symbol(
    df: pd.DataFrame,
    context_length: int = 40000,
    horizon: int = 10,
    stride: int = 100,
    target_columns: List[str] = ['close'],
    verbose: bool = True
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[str]]:
    """
    Create sliding windows separately for each symbol

    Args:
        df: Input DataFrame with 'symbol' column
        context_length: Length of input sequence
        horizon: Length of prediction sequence
        stride: Step size between windows
        target_columns: Columns to predict
        verbose: Print progress information

    Returns:
        Tuple of (all_context_windows, all_target_windows, all_symbols)
    """
    if verbose:
        print(f"\n[WINDOWING] Creating windows per symbol...")
        print(f"[WINDOWING]   - Input shape: {df.shape}")

    if 'symbol' not in df.columns:
        raise ValueError("DataFrame must have 'symbol' column for per-symbol windowing")

    symbols = df['symbol'].unique()

    if verbose:
        print(f"[WINDOWING]   - Found {len(symbols)} symbols: {list(symbols)}")

    all_context_windows = []
    all_target_windows = []
    all_symbols = []

    for i, symbol in enumerate(symbols, 1):
        if verbose:
            print(f"\n[WINDOWING] [{i}/{len(symbols)}] Processing {symbol}...")

        # Filter data for this symbol
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('datetime').reset_index(drop=True)

        if verbose:
            print(f"[WINDOWING]   - Symbol data: {len(symbol_df)} rows")

        # Create windows for this symbol
        try:
            context_wins, target_wins = create_sliding_windows(
                symbol_df,
                context_length=context_length,
                horizon=horizon,
                stride=stride,
                target_columns=target_columns,
                verbose=verbose
            )

            # Add to combined lists
            all_context_windows.extend(context_wins)
            all_target_windows.extend(target_wins)
            all_symbols.extend([symbol] * len(context_wins))

            if verbose:
                print(f"[WINDOWING] ✓ Created {len(context_wins)} windows for {symbol}")

        except ValueError as e:
            if verbose:
                print(f"[WINDOWING] ⚠ Skipping {symbol}: {e}")
            continue

    if verbose:
        print(f"\n[WINDOWING] ✓ Total windows created: {len(all_context_windows)}")
        print(f"[WINDOWING]   - Distribution by symbol:")

        for symbol in symbols:
            count = all_symbols.count(symbol)
            print(f"[WINDOWING]     - {symbol}: {count} windows")

    return all_context_windows, all_target_windows, all_symbols


def verify_windows(
    context_windows: List[pd.DataFrame],
    target_windows: List[pd.DataFrame],
    expected_context_length: int,
    expected_horizon: int
) -> bool:
    """
    Verify window integrity

    Args:
        context_windows: List of context DataFrames
        target_windows: List of target DataFrames
        expected_context_length: Expected context length
        expected_horizon: Expected horizon length

    Returns:
        True if all windows are valid
    """
    print(f"\n[WINDOWING] Verifying windows...")
    print(f"[WINDOWING]   - Number of windows: {len(context_windows)}")

    if len(context_windows) != len(target_windows):
        print(f"[WINDOWING] ✗ Mismatch: {len(context_windows)} context vs {len(target_windows)} target windows")
        return False

    if len(context_windows) == 0:
        print(f"[WINDOWING] ✗ No windows created!")
        return False

    # Check sizes
    issues = 0

    for i, (ctx, tgt) in enumerate(zip(context_windows, target_windows)):
        if len(ctx) != expected_context_length:
            print(f"[WINDOWING] ✗ Window {i}: context length {len(ctx)} != {expected_context_length}")
            issues += 1

        if len(tgt) != expected_horizon:
            print(f"[WINDOWING] ✗ Window {i}: target length {len(tgt)} != {expected_horizon}")
            issues += 1

        # Check for NaN
        if ctx.isna().any().any():
            nan_count = ctx.isna().sum().sum()
            print(f"[WINDOWING] ⚠ Window {i}: context has {nan_count} NaN values")

        if tgt.isna().any().any():
            nan_count = tgt.isna().sum().sum()
            print(f"[WINDOWING] ⚠ Window {i}: target has {nan_count} NaN values")

        if issues >= 5:  # Limit output
            print(f"[WINDOWING]   - (stopping after 5 issues, may be more...)")
            break

    if issues == 0:
        print(f"[WINDOWING] ✓ All windows verified successfully")
        print(f"[WINDOWING]   - Context shape: ({expected_context_length}, {context_windows[0].shape[1]})")
        print(f"[WINDOWING]   - Target shape: ({expected_horizon}, {target_windows[0].shape[1]})")
        return True
    else:
        print(f"[WINDOWING] ✗ Found {issues} issues")
        return False


def get_window_statistics(
    context_windows: List[pd.DataFrame],
    target_windows: List[pd.DataFrame]
) -> dict:
    """
    Get statistics about created windows

    Args:
        context_windows: List of context DataFrames
        target_windows: List of target DataFrames

    Returns:
        Dictionary with statistics
    """
    print(f"\n[WINDOWING] Computing window statistics...")

    stats = {
        'n_windows': len(context_windows),
        'context_length': len(context_windows[0]) if len(context_windows) > 0 else 0,
        'horizon': len(target_windows[0]) if len(target_windows) > 0 else 0,
        'n_features': context_windows[0].shape[1] if len(context_windows) > 0 else 0,
    }

    # Calculate time span
    if len(context_windows) > 0 and 'datetime' in context_windows[0].columns:
        first_date = context_windows[0]['datetime'].iloc[0]
        last_date = target_windows[-1]['datetime'].iloc[-1]
        stats['time_span_days'] = (last_date - first_date).days

        print(f"[WINDOWING]   - Time span: {stats['time_span_days']} days")
        print(f"[WINDOWING]   - First datetime: {first_date}")
        print(f"[WINDOWING]   - Last datetime: {last_date}")

    print(f"[WINDOWING] ✓ Statistics computed")

    for key, value in stats.items():
        print(f"[WINDOWING]   - {key}: {value}")

    return stats


if __name__ == "__main__":
    # Test windowing
    print("Testing Windowing Functions")
    print("=" * 60)

    # Create sample data
    print("\n[TEST] Creating sample data...")
    dates = pd.date_range('2020-01-01', periods=50000, freq='1H')

    sample_df = pd.DataFrame({
        'datetime': dates,
        'symbol': 'TESTUSDT',
        'close': np.random.randn(50000).cumsum() + 3000,
        'volume': np.random.rand(50000) * 1000,
        'feature1': np.random.randn(50000),
        'feature2': np.random.randn(50000),
    })

    print(f"[TEST]   - Created sample data: {sample_df.shape}")

    # Test windowing
    try:
        context_windows, target_windows = create_sliding_windows(
            sample_df,
            context_length=1000,  # Use smaller context for testing
            horizon=10,
            stride=100,
            verbose=True
        )

        # Verify
        is_valid = verify_windows(context_windows, target_windows, 1000, 10)

        # Get statistics
        stats = get_window_statistics(context_windows, target_windows)

        if is_valid:
            print("\n[TEST] ✓ Windowing test passed!")
        else:
            print("\n[TEST] ✗ Windowing test failed!")

    except Exception as e:
        print(f"\n[TEST] ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
