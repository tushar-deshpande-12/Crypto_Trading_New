"""Verify no target leakage in features."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

from src.ml.preprocessing.preprocessor import CryptoPreprocessor
from src.ml.models.lstm_model import SimpleLSTMDataset

prep = CryptoPreprocessor("dataset")
train_df, val_df, test_df = prep.process_all(["BTCUSDT", "ETHUSDT"])

# Create dataset
train_dataset = SimpleLSTMDataset(train_df, sequence_length=168)

print("\n" + "="*60)
print("FEATURE VERIFICATION")
print("="*60)

print(f"\nTotal DataFrame columns: {len(train_df.columns)}")
print(f"LSTM Dataset features: {len(train_dataset.feature_columns)}")

# Check for target-related columns
target_cols = [c for c in train_dataset.feature_columns if 'target' in c.lower()]
print(f"\nTarget columns in features: {target_cols}")

if target_cols:
    print(">> WARNING: DATA LEAKAGE - target columns in features!")
else:
    print(">> OK: No target columns in features")

# List all features
print(f"\nAll features used ({len(train_dataset.feature_columns)}):")
for i, col in enumerate(sorted(train_dataset.feature_columns)):
    print(f"  {i+1:2d}. {col}")
