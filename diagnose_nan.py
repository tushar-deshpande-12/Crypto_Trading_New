"""Diagnose where NaN values are coming from"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
from src.ml.preprocessing.preprocessor import CryptoPreprocessor

print("="*80)
print("DIAGNOSING NaN VALUES IN PIPELINE")
print("="*80)

preprocessor = CryptoPreprocessor(dataset_dir="dataset")
symbols = ['ETHUSDT']

# Step 1: Load raw data
print("\n[STEP 1] Loading raw data...")
df = preprocessor.load_multi_symbol_data(symbols)
print(f"Raw data shape: {df.shape}")
print(f"Raw data NaN count: {df.isna().sum().sum()}")
print(f"Raw 'close' column NaN: {df['close'].isna().sum()} / {len(df)} ({df['close'].isna().sum()/len(df)*100:.2f}%)")
print(f"Raw 'close' min/max: {df['close'].min():.2f} / {df['close'].max():.2f}")

# Step 2: After temporal features
print("\n[STEP 2] After generating temporal features...")
df = preprocessor.generate_temporal_features(df)
print(f"Data shape: {df.shape}")
print(f"Total NaN count: {df.isna().sum().sum()}")
print(f"'close' column NaN: {df['close'].isna().sum()} / {len(df)}")

# Step 3: After technical indicators
print("\n[STEP 3] After generating technical indicators...")
df = preprocessor.generate_technical_indicators(df)
print(f"Data shape: {df.shape}")
print(f"Total NaN count: {df.isna().sum().sum()}")
print(f"'close' column NaN: {df['close'].isna().sum()} / {len(df)}")
print("\nNaN by column:")
nan_cols = df.isna().sum()
nan_cols = nan_cols[nan_cols > 0].sort_values(ascending=False)
for col, count in nan_cols.items():
    print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")

# Step 4: After lagged features
print("\n[STEP 4] After generating lagged features...")
df = preprocessor.generate_lagged_features(df)
print(f"Data shape: {df.shape}")
print(f"Total NaN count: {df.isna().sum().sum()}")
print(f"'close' column NaN: {df['close'].isna().sum()} / {len(df)}")

# Step 5: After rolling features
print("\n[STEP 5] After generating rolling features...")
df = preprocessor.generate_rolling_features(df)
print(f"Data shape: {df.shape}")
print(f"Total NaN count: {df.isna().sum().sum()}")
print(f"'close' column NaN: {df['close'].isna().sum()} / {len(df)}")

# Step 6: After dropping NaN
print("\n[STEP 6] After dropping NaN rows...")
df_before_drop = df.copy()
df = df.dropna().reset_index(drop=True)
print(f"Data shape: {df.shape}")
print(f"Rows dropped: {len(df_before_drop) - len(df)}")
print(f"Total NaN count after drop: {df.isna().sum().sum()}")
print(f"'close' column NaN after drop: {df['close'].isna().sum()} / {len(df)}")

# Step 7: Split data
print("\n[STEP 7] After splitting train/val/test...")
train_df, val_df, test_df = preprocessor.split_train_val_test(df)
print(f"Train shape: {train_df.shape}, NaN: {train_df.isna().sum().sum()}")
print(f"Val shape: {val_df.shape}, NaN: {val_df.isna().sum().sum()}")
print(f"Test shape: {test_df.shape}, NaN: {test_df.isna().sum().sum()}")
print(f"Train 'close' NaN: {train_df['close'].isna().sum()}")
print(f"Val 'close' NaN: {val_df['close'].isna().sum()}")
print(f"Test 'close' NaN: {test_df['close'].isna().sum()}")

# Step 8: After normalization
print("\n[STEP 8] After normalization...")
train_df_normalized = preprocessor.normalize(train_df, fit=True)
print(f"Train shape after norm: {train_df_normalized.shape}")
print(f"Train NaN after norm: {train_df_normalized.isna().sum().sum()}")
print(f"Train 'close' NaN after norm: {train_df_normalized['close'].isna().sum()} / {len(train_df_normalized)}")

# Check for Inf values
print(f"\nTrain 'close' Inf values: {np.isinf(train_df_normalized['close']).sum()}")
print(f"Train 'close' min/max after norm: {train_df_normalized['close'].min():.6f} / {train_df_normalized['close'].max():.6f}")
print(f"Train 'close' mean/std after norm: {train_df_normalized['close'].mean():.6f} / {train_df_normalized['close'].std():.6f}")

# Check which columns have NaN after normalization
print("\n[CRITICAL] NaN by column after normalization:")
nan_cols_norm = train_df_normalized.isna().sum()
nan_cols_norm = nan_cols_norm[nan_cols_norm > 0].sort_values(ascending=False)
for col, count in nan_cols_norm.items():
    print(f"  {col}: {count} ({count/len(train_df_normalized)*100:.2f}%)")

# Check scaler statistics
print("\n[SCALER DEBUG] Checking scaler statistics...")
for symbol in train_df['symbol'].unique():
    scaler = preprocessor.scalers[symbol]
    print(f"\n  Symbol: {symbol}")
    print(f"  Features with zero std (will cause NaN):")
    zero_std_features = []
    for i, (feature, std) in enumerate(zip(preprocessor.feature_columns, scaler.scale_)):
        if std < 1e-10:
            zero_std_features.append(feature)
            print(f"    {feature}: std={std:.10f}")

    if not zero_std_features:
        print(f"    None - all features have non-zero std")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
