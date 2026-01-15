"""Check which features have extreme values (not normalized)."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import numpy as np
from src.ml.preprocessing.preprocessor import CryptoPreprocessor
from src.ml.models.lstm_model import SimpleLSTMDataset

prep = CryptoPreprocessor("dataset")
train_df, val_df, test_df = prep.process_all(["BTCUSDT", "ETHUSDT"])

# Create dataset
train_dataset = SimpleLSTMDataset(train_df, sequence_length=168)

print("\n" + "="*80)
print("FEATURE ANALYSIS")
print("="*80)

print(f"\nPreprocessor normalized {len(prep.feature_columns)} features")
print(f"Dataset using {len(train_dataset.feature_columns)} features")

# Find features not in preprocessor's list
extra_features = set(train_dataset.feature_columns) - set(prep.feature_columns)
print(f"\nFeatures NOT normalized by preprocessor ({len(extra_features)}):")
for feat in sorted(extra_features):
    vals = train_df[feat].values
    print(f"  {feat}: mean={vals.mean():.2f}, std={vals.std():.2f}, max={vals.max():.2f}")

# Check which features have extreme values
print("\n" + "="*80)
print("FEATURES WITH EXTREME VALUES (|mean| > 10 or std > 10)")
print("="*80)

for i, col in enumerate(train_dataset.feature_columns):
    vals = train_dataset.features[:, i]
    if abs(vals.mean()) > 10 or vals.std() > 10 or abs(vals.max()) > 1000:
        print(f"  {col}: mean={vals.mean():.2e}, std={vals.std():.2e}, max={vals.max():.2e}")
