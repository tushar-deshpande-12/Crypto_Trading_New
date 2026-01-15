"""
Diagnostic script to identify why training loss increases and val loss stays flat.
Run this BEFORE training to check data quality.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import torch

from src.ml.preprocessing.preprocessor import CryptoPreprocessor, validate_data_for_training
from src.ml.models.lstm_model import SimpleLSTMDataset, get_lstm_feature_columns

def diagnose_data():
    print("="*80)
    print("DIAGNOSTIC: Checking Data Pipeline")
    print("="*80)

    # Load and preprocess
    prep = CryptoPreprocessor("dataset")

    # Get available symbols
    symbols = []
    for d in prep.dataset_dir.iterdir():
        if d.is_dir() and not d.name.startswith('.'):
            symbols.append(d.name)

    print(f"\nAvailable symbols: {symbols}")
    symbols_to_use = symbols[:2] if len(symbols) >= 2 else symbols
    print(f"Using: {symbols_to_use}")

    # Process data
    train_df, val_df, test_df = prep.process_all(symbols_to_use)

    # Run validation
    print("\n")
    results = validate_data_for_training(train_df, val_df, test_df)

    return train_df, val_df, test_df, prep

def diagnose_dataset(train_df, val_df):
    print("\n" + "="*80)
    print("DIAGNOSTIC: Checking Dataset Creation")
    print("="*80)

    # Create datasets
    train_dataset = SimpleLSTMDataset(train_df, sequence_length=168)
    val_dataset = SimpleLSTMDataset(val_df, sequence_length=168,
                                     feature_columns=train_dataset.feature_columns)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} sequences")
    print(f"  Val: {len(val_dataset)} sequences")
    print(f"  Features: {len(train_dataset.feature_columns)}")

    # Check a sample batch
    print("\n[SAMPLE BATCH CHECK]")
    x_train, y_train = train_dataset[0]
    x_val, y_val = val_dataset[0]

    print(f"\nTrain sample:")
    print(f"  Input shape: {x_train.shape}")
    print(f"  Input mean: {x_train.mean():.4f}, std: {x_train.std():.4f}")
    print(f"  Input min: {x_train.min():.4f}, max: {x_train.max():.4f}")
    print(f"  Target: {y_train.numpy()}")

    print(f"\nVal sample:")
    print(f"  Input shape: {x_val.shape}")
    print(f"  Input mean: {x_val.mean():.4f}, std: {x_val.std():.4f}")
    print(f"  Input min: {x_val.min():.4f}, max: {x_val.max():.4f}")
    print(f"  Target: {y_val.numpy()}")

    # Check for NaN/Inf in tensors
    print("\n[NaN/Inf CHECK]")
    has_issues = False

    # Check multiple samples
    for i in range(min(100, len(train_dataset))):
        x, y = train_dataset[i]
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"  ❌ Train sample {i} has NaN/Inf in input!")
            has_issues = True
            break
        if torch.isnan(y).any() or torch.isinf(y).any():
            print(f"  ❌ Train sample {i} has NaN/Inf in target!")
            has_issues = True
            break

    if not has_issues:
        print("  ✓ No NaN/Inf found in first 100 train samples")

    # Check target distribution
    print("\n[TARGET DISTRIBUTION]")
    all_train_targets = []
    all_val_targets = []

    for i in range(min(1000, len(train_dataset))):
        _, y = train_dataset[i]
        all_train_targets.append(y.numpy())

    for i in range(min(1000, len(val_dataset))):
        _, y = val_dataset[i]
        all_val_targets.append(y.numpy())

    train_targets = np.concatenate(all_train_targets)
    val_targets = np.concatenate(all_val_targets)

    print(f"\nTrain targets (n={len(train_targets)}):")
    print(f"  Mean: {train_targets.mean():.6f}")
    print(f"  Std:  {train_targets.std():.6f}")
    print(f"  Min:  {train_targets.min():.6f}")
    print(f"  Max:  {train_targets.max():.6f}")

    print(f"\nVal targets (n={len(val_targets)}):")
    print(f"  Mean: {val_targets.mean():.6f}")
    print(f"  Std:  {val_targets.std():.6f}")
    print(f"  Min:  {val_targets.min():.6f}")
    print(f"  Max:  {val_targets.max():.6f}")

    # Check if targets are constant (would cause val loss to stay flat)
    if train_targets.std() < 0.001:
        print("\n  ❌ CRITICAL: Train targets have near-zero variance!")
        print("     This means the model has nothing to learn.")

    if val_targets.std() < 0.001:
        print("\n  ❌ CRITICAL: Val targets have near-zero variance!")
        print("     This explains why val loss stays constant.")

    # Check feature variance
    print("\n[FEATURE VARIANCE CHECK]")
    low_var_features = []
    for i, col in enumerate(train_dataset.feature_columns):
        feature_vals = train_dataset.features[:, i]
        if feature_vals.std() < 0.001:
            low_var_features.append(col)

    if low_var_features:
        print(f"  ⚠️  {len(low_var_features)} features have near-zero variance:")
        for f in low_var_features[:10]:
            print(f"      - {f}")
        if len(low_var_features) > 10:
            print(f"      ... and {len(low_var_features) - 10} more")
    else:
        print("  ✓ All features have reasonable variance")

    return train_dataset, val_dataset

def diagnose_model_forward():
    print("\n" + "="*80)
    print("DIAGNOSTIC: Testing Model Forward Pass")
    print("="*80)

    from src.ml.models.lstm_model import SimpleLSTM

    # Create a small test model
    model = SimpleLSTM(
        input_size=64,  # Dummy size
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001
    )

    # Test with random input
    batch_size = 4
    seq_len = 168
    input_size = 64

    x = torch.randn(batch_size, seq_len, input_size)

    model.eval()
    with torch.no_grad():
        reg_out, dir_out = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Regression output shape: {reg_out.shape}")
    print(f"Direction output shape: {dir_out.shape}")
    print(f"\nRegression output stats:")
    print(f"  Mean: {reg_out.mean():.6f}")
    print(f"  Std:  {reg_out.std():.6f}")

    if torch.isnan(reg_out).any():
        print("  ❌ Output contains NaN!")
    else:
        print("  ✓ Output is valid (no NaN)")

def suggest_fixes(train_targets, val_targets):
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    issues = []

    # Check target scale
    if abs(train_targets.mean()) > 0.5 or train_targets.std() > 2.0:
        issues.append("Target values may not be properly normalized")

    if train_targets.std() < 0.01:
        issues.append("Target variance is too low - model can't learn meaningful patterns")

    # Check distribution shift
    mean_diff = abs(val_targets.mean() - train_targets.mean())
    if mean_diff > 0.5:
        issues.append(f"Large distribution shift between train/val (mean diff: {mean_diff:.4f})")

    if not issues:
        print("\n✓ Data looks OK. Try these hyperparameter changes:")
        print("  1. Lower learning rate: 0.0001 instead of 0.001")
        print("  2. Increase batch size: 128 instead of 64")
        print("  3. Add gradient clipping: already set to 1.0")
        print("  4. Reduce model complexity: hidden_size=64, num_layers=1")
    else:
        print("\n❌ Found issues:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

        print("\nSuggested fixes:")
        print("  1. Check preprocessor normalization")
        print("  2. Verify target_return calculation")
        print("  3. Try predicting 'close' instead of 'target_return'")

if __name__ == "__main__":
    try:
        # Run diagnostics
        train_df, val_df, test_df, prep = diagnose_data()
        train_dataset, val_dataset = diagnose_dataset(train_df, val_df)
        diagnose_model_forward()

        # Get targets for recommendations
        train_targets = train_df['target_return'].values
        val_targets = val_df['target_return'].values
        suggest_fixes(train_targets, val_targets)

        print("\n" + "="*80)
        print("DIAGNOSTIC COMPLETE")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error during diagnostics: {e}")
        import traceback
        traceback.print_exc()
