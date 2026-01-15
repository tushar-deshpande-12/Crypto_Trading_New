"""Check distribution of LSTM features specifically."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import numpy as np
from src.ml.preprocessing.preprocessor import CryptoPreprocessor
from src.ml.models.lstm_model import SimpleLSTMDataset

prep = CryptoPreprocessor("dataset")
train_df, val_df, test_df = prep.process_all(["BTCUSDT", "ETHUSDT"])

# Create dataset to get actual features used
train_dataset = SimpleLSTMDataset(train_df, sequence_length=168)
feature_cols = train_dataset.feature_columns

print("\n" + "="*70)
print(f"LSTM FEATURE DISTRIBUTION CHECK ({len(feature_cols)} features)")
print("="*70)

# Check each feature
print("\n[FEATURES WITH DISTRIBUTION SHIFT]")
print("-"*70)
shifted_features = []

for feat in feature_cols:
    train_mean = train_df[feat].mean()
    train_std = train_df[feat].std()
    val_mean = val_df[feat].mean()
    val_std = val_df[feat].std()

    # Check for shift (mean difference > 0.5 std)
    mean_shift = abs(val_mean - train_mean)

    # Check for OOD
    train_min, train_max = train_df[feat].min(), train_df[feat].max()
    val_below = (val_df[feat] < train_min).sum()
    val_above = (val_df[feat] > train_max).sum()
    ood_pct = 100 * (val_below + val_above) / len(val_df)

    if mean_shift > 0.5 or ood_pct > 5:
        shifted_features.append((feat, mean_shift, ood_pct, train_mean, val_mean))
        print(f"  {feat:35s}: shift={mean_shift:.2f}, OOD={ood_pct:.1f}%")

if not shifted_features:
    print("  None! All features have similar distributions.")

print(f"\n[SUMMARY]")
print("-"*70)
print(f"Total LSTM features: {len(feature_cols)}")
print(f"Features with shift: {len(shifted_features)}")
print(f"Clean features: {len(feature_cols) - len(shifted_features)}")

if shifted_features:
    print(f"\n>> WARNING: {len(shifted_features)} features still have distribution issues")
else:
    print(f"\n>> OK: All LSTM features are stationary!")
