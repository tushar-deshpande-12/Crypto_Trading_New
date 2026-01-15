"""
Deep analysis of data quality to find why model can't learn.
Checks: target distribution, feature-target correlations, signal-to-noise ratio.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from scipy import stats

from src.ml.preprocessing.preprocessor import CryptoPreprocessor

def analyze_target(train_df, val_df):
    """Analyze target variable quality."""
    print("\n" + "="*80)
    print("1. TARGET VARIABLE ANALYSIS")
    print("="*80)

    target = 'target_return'
    train_target = train_df[target].values
    val_target = val_df[target].values

    print(f"\nTrain target statistics:")
    print(f"  Mean:   {train_target.mean():.6f}")
    print(f"  Std:    {train_target.std():.6f}")
    print(f"  Min:    {train_target.min():.6f}")
    print(f"  Max:    {train_target.max():.6f}")
    print(f"  Median: {np.median(train_target):.6f}")

    print(f"\nVal target statistics:")
    print(f"  Mean:   {val_target.mean():.6f}")
    print(f"  Std:    {val_target.std():.6f}")
    print(f"  Min:    {val_target.min():.6f}")
    print(f"  Max:    {val_target.max():.6f}")

    # Check for extreme values
    train_outliers = np.sum(np.abs(train_target) > 3)
    print(f"\n  Train outliers (|z| > 3): {train_outliers} ({100*train_outliers/len(train_target):.2f}%)")

    # Autocorrelation (is there any pattern?)
    autocorr_1 = np.corrcoef(train_target[:-1], train_target[1:])[0,1]
    autocorr_24 = np.corrcoef(train_target[:-24], train_target[24:])[0,1]
    print(f"\n  Autocorrelation lag-1:  {autocorr_1:.4f}")
    print(f"  Autocorrelation lag-24: {autocorr_24:.4f}")

    if abs(autocorr_1) < 0.05:
        print("  >> WARNING: Near-zero autocorrelation = target looks like random noise!")

    # Distribution test
    _, p_value = stats.normaltest(train_target[:5000])  # Sample for speed
    print(f"\n  Normality test p-value: {p_value:.6f}")

    return train_target, val_target


def analyze_feature_correlations(train_df, target='target_return'):
    """Find which features correlate with target."""
    print("\n" + "="*80)
    print("2. FEATURE-TARGET CORRELATIONS")
    print("="*80)

    # Get numeric columns only
    exclude_cols = ['datetime', 'symbol', 'time_idx', 'target_close', 'actual_direction']
    numeric_cols = [col for col in train_df.select_dtypes(include=np.number).columns
                   if col not in exclude_cols and col != target]

    correlations = {}
    for col in numeric_cols:
        corr = train_df[col].corr(train_df[target])
        if not np.isnan(corr):
            correlations[col] = corr

    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"\nTop 15 features by |correlation| with target:")
    print("-" * 50)
    for feat, corr in sorted_corrs[:15]:
        print(f"  {feat:40s}: {corr:+.4f}")

    print(f"\nBottom 5 features (lowest correlation):")
    print("-" * 50)
    for feat, corr in sorted_corrs[-5:]:
        print(f"  {feat:40s}: {corr:+.4f}")

    # Check if ANY feature has meaningful correlation
    max_corr = max(abs(c) for c in correlations.values())
    print(f"\n  Maximum |correlation|: {max_corr:.4f}")

    if max_corr < 0.1:
        print("  >> CRITICAL: No feature correlates >0.1 with target!")
        print("  >> This means features have NO predictive power for target.")
    elif max_corr < 0.2:
        print("  >> WARNING: Weak correlations. Model will struggle to learn.")

    return correlations


def analyze_signal_to_noise(train_df, target='target_return'):
    """Estimate signal-to-noise ratio."""
    print("\n" + "="*80)
    print("3. SIGNAL-TO-NOISE ANALYSIS")
    print("="*80)

    target_vals = train_df[target].values

    # Method 1: Compare variance of smoothed vs raw target
    # If smoothed variance << raw variance, mostly noise
    window = 24  # 24-hour smoothing
    smoothed = pd.Series(target_vals).rolling(window).mean().dropna().values
    raw_subset = target_vals[window-1:]

    raw_var = np.var(raw_subset)
    smooth_var = np.var(smoothed)
    noise_var = raw_var - smooth_var

    snr = smooth_var / noise_var if noise_var > 0 else float('inf')

    print(f"\n  Raw variance:      {raw_var:.6f}")
    print(f"  Smoothed variance: {smooth_var:.6f}")
    print(f"  Noise variance:    {noise_var:.6f}")
    print(f"  Signal-to-Noise:   {snr:.4f}")

    if snr < 0.1:
        print("  >> CRITICAL: SNR < 0.1 means >90% of target is noise!")
        print("  >> Model cannot learn meaningful patterns from this.")
    elif snr < 0.5:
        print("  >> WARNING: Low SNR. Target is very noisy.")

    # Method 2: Check predictability with simple lag model
    # If lag-1 can't predict, nothing can
    lag1_corr = np.corrcoef(target_vals[:-1], target_vals[1:])[0,1]
    lag1_r2 = lag1_corr ** 2

    print(f"\n  Lag-1 R-squared: {lag1_r2:.6f}")
    print(f"  (Best possible R2 with just previous value)")

    if lag1_r2 < 0.01:
        print("  >> Target has almost no temporal structure!")

    return snr


def analyze_raw_returns(train_df):
    """Analyze the raw (unnormalized) returns to see true scale."""
    print("\n" + "="*80)
    print("4. RAW PRICE MOVEMENT ANALYSIS")
    print("="*80)

    # Check if we can compute raw returns from close prices
    if 'close' in train_df.columns:
        # The close is normalized, so we can't get raw returns directly
        # But we can analyze the normalized target
        target = train_df['target_return'].values

        # Distribution of target
        percentiles = np.percentile(target, [1, 5, 25, 50, 75, 95, 99])
        print(f"\n  Target percentiles:")
        print(f"    1%:   {percentiles[0]:.4f}")
        print(f"    5%:   {percentiles[1]:.4f}")
        print(f"    25%:  {percentiles[2]:.4f}")
        print(f"    50%:  {percentiles[3]:.4f}")
        print(f"    75%:  {percentiles[4]:.4f}")
        print(f"    95%:  {percentiles[5]:.4f}")
        print(f"    99%:  {percentiles[6]:.4f}")

        # What % are basically zero?
        near_zero = np.sum(np.abs(target) < 0.1)
        print(f"\n  Targets near zero (|t| < 0.1): {100*near_zero/len(target):.1f}%")

        # Direction balance
        up = np.sum(target > 0)
        down = np.sum(target < 0)
        print(f"\n  Direction balance: {100*up/len(target):.1f}% up, {100*down/len(target):.1f}% down")


def suggest_solutions(correlations, snr, autocorr):
    """Suggest data preprocessing improvements."""
    print("\n" + "="*80)
    print("5. RECOMMENDED SOLUTIONS")
    print("="*80)

    max_corr = max(abs(c) for c in correlations.values()) if correlations else 0

    suggestions = []

    if max_corr < 0.1:
        suggestions.append(
            "1. FEATURE ENGINEERING:\n"
            "   - Current features don't correlate with target\n"
            "   - Try: momentum indicators, order book imbalance, sentiment\n"
            "   - Try: cross-asset features (BTC leading ETH)\n"
            "   - Try: volatility regime features"
        )

    if snr < 0.2:
        suggestions.append(
            "2. TARGET ENGINEERING:\n"
            "   - Current target is too noisy\n"
            "   - Try: predict longer horizon (4h, 24h instead of 1h)\n"
            "   - Try: predict direction only (classification)\n"
            "   - Try: predict volatility instead of returns\n"
            "   - Try: smooth target with moving average"
        )

    if abs(autocorr) < 0.05:
        suggestions.append(
            "3. CONSIDER DIFFERENT APPROACH:\n"
            "   - Returns show no temporal structure (random walk)\n"
            "   - Standard LSTM may not be suitable\n"
            "   - Try: regime detection, anomaly detection\n"
            "   - Try: predict probability distributions, not point estimates"
        )

    suggestions.append(
        "4. QUICK FIXES TO TRY:\n"
        "   a) Increase prediction horizon to 24h\n"
        "   b) Use classification (up/down) instead of regression\n"
        "   c) Add more predictive features\n"
        "   d) Filter out low-volatility periods"
    )

    for s in suggestions:
        print(f"\n{s}")


if __name__ == "__main__":
    print("="*80)
    print("DATA QUALITY ANALYSIS FOR CRYPTO PREDICTION")
    print("="*80)

    # Load data
    prep = CryptoPreprocessor("dataset")
    train_df, val_df, test_df = prep.process_all(["BTCUSDT", "ETHUSDT"])

    # Run analyses
    train_target, val_target = analyze_target(train_df, val_df)
    correlations = analyze_feature_correlations(train_df)
    snr = analyze_signal_to_noise(train_df)
    analyze_raw_returns(train_df)

    # Autocorrelation for suggestions
    autocorr = np.corrcoef(train_target[:-1], train_target[1:])[0,1]

    # Suggestions
    suggest_solutions(correlations, snr, autocorr)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
