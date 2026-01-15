"""Diagnose train vs val distribution mismatch."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from src.ml.preprocessing.preprocessor import CryptoPreprocessor

prep = CryptoPreprocessor("dataset")
train_df, val_df, test_df = prep.process_all(["BTCUSDT", "ETHUSDT"])

print("\n" + "="*70)
print("TRAIN vs VAL DISTRIBUTION ANALYSIS")
print("="*70)

# 1. Check target distribution
print("\n[1] TARGET DISTRIBUTION")
print("-"*50)
train_target = train_df['target_return'].values
val_target = val_df['target_return'].values

print(f"Train target: mean={train_target.mean():.4f}, std={train_target.std():.4f}")
print(f"Val target:   mean={val_target.mean():.4f}, std={val_target.std():.4f}")
print(f"Difference:   mean_diff={abs(train_target.mean() - val_target.mean()):.4f}")

# 2. Check key feature distributions
print("\n[2] KEY FEATURE DISTRIBUTIONS")
print("-"*50)

key_features = ['close', 'volume', 'rsi', 'macd', 'volatility', 'return_24h']
for feat in key_features:
    if feat in train_df.columns:
        train_mean = train_df[feat].mean()
        train_std = train_df[feat].std()
        val_mean = val_df[feat].mean()
        val_std = val_df[feat].std()

        mean_shift = abs(val_mean - train_mean)
        std_ratio = val_std / train_std if train_std > 0 else 0

        status = "OK" if mean_shift < 0.5 else "SHIFT!"
        print(f"{feat:20s}: train({train_mean:+.2f}, {train_std:.2f}) "
              f"val({val_mean:+.2f}, {val_std:.2f}) [{status}]")

# 3. Check time periods
print("\n[3] TIME PERIOD ANALYSIS")
print("-"*50)
for symbol in train_df['symbol'].unique():
    train_dates = train_df[train_df['symbol']==symbol]['datetime']
    val_dates = val_df[val_df['symbol']==symbol]['datetime']

    print(f"\n{symbol}:")
    print(f"  Train: {train_dates.min()} to {train_dates.max()}")
    print(f"  Val:   {val_dates.min()} to {val_dates.max()}")

    # Time gap
    gap = (val_dates.min() - train_dates.max()).total_seconds() / 3600
    print(f"  Gap: {gap:.0f} hours")

# 4. Check if val is out-of-distribution
print("\n[4] OUT-OF-DISTRIBUTION CHECK")
print("-"*50)

# For each feature, check how many val samples are outside train range
ood_features = []
for feat in train_df.select_dtypes(include=np.number).columns:
    if feat in ['time_idx', 'target_return', 'target_close', 'actual_direction']:
        continue
    if feat not in val_df.columns:
        continue

    train_min, train_max = train_df[feat].min(), train_df[feat].max()
    val_below = (val_df[feat] < train_min).sum()
    val_above = (val_df[feat] > train_max).sum()
    ood_pct = 100 * (val_below + val_above) / len(val_df)

    if ood_pct > 5:
        ood_features.append((feat, ood_pct))

if ood_features:
    print(f"Features with >5% val samples outside train range:")
    for feat, pct in sorted(ood_features, key=lambda x: -x[1])[:15]:
        print(f"  {feat:35s}: {pct:.1f}% out-of-distribution")
else:
    print("No major out-of-distribution features found.")

# 5. Recommendation
print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

# Check for major issues
mean_shift = abs(train_target.mean() - val_target.mean())
std_ratio = val_target.std() / train_target.std()

if mean_shift > 0.5:
    print("\n>> CRITICAL: Target mean shifted significantly between train/val")
    print("   This means the market regime changed.")

if std_ratio < 0.7 or std_ratio > 1.3:
    print(f"\n>> WARNING: Target std ratio = {std_ratio:.2f}")
    print("   Validation period has different volatility than training.")

if ood_features:
    print(f"\n>> WARNING: {len(ood_features)} features have out-of-distribution values in val")
    print("   Model is extrapolating beyond training range.")

print("\n>> LIKELY CAUSE: Market regime change between train and val periods.")
print("   The model learned patterns from one market condition that don't apply to another.")
