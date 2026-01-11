"""
Example: How to use the 5 validation steps in your training pipeline

Run this after training starts to validate your data pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.ml.preprocessing.preprocessor import CryptoPreprocessor
from src.ml.training.dataset import create_dataloaders, validate_data_format, validate_target_normalization
from src.ml.models.model_config import TFTConfig
from src.ml.models.tft_model import CryptoTFT
from src.ml.training.metrics import validate_loss_function, validate_baseline_comparison

print("="*80)
print("TRAINING PIPELINE VALIDATION EXAMPLE")
print("="*80)

# Step 1: Load data
preprocessor = CryptoPreprocessor(dataset_dir="dataset")
symbols = ['ETHUSDT']  # Adjust to your available symbols

train_df, val_df, test_df = preprocessor.process_all(symbols=symbols)
train_df = preprocessor.add_time_index(train_df)
val_df = preprocessor.add_time_index(val_df)
test_df = preprocessor.add_time_index(test_df)

# Step 2: Create dataloaders
config = TFTConfig()
train_loader, val_loader, test_loader = create_dataloaders(
    train_df, val_df, test_df,
    context_length=config.max_encoder_length,
    prediction_length=config.max_prediction_length,
    batch_size=config.batch_size,
    verbose=False
)

# Step 3: Create model
crypto_tft = CryptoTFT(config, verbose=False)
crypto_tft.create_from_dataset(train_loader.dataset)
model = crypto_tft.model

# ============================================================================
# RUN ALL 5 VALIDATION STEPS
# ============================================================================

print("\n" + "="*80)
print("RUNNING 5 VALIDATION STEPS")
print("="*80)

results = {}

# STEP 1: Validate data format
results['step1_data_format'] = validate_data_format(train_loader, verbose=True)

# STEP 2: Validate loss function behavior
results['step2_loss_function'] = validate_loss_function(model, train_loader, verbose=True)

# STEP 3: Check for data leakage
results['step3_data_leakage'] = CryptoPreprocessor.validate_no_data_leakage(
    train_df, val_df, test_df, verbose=True
)

# STEP 4: Validate target normalization
results['step4_target_normalization'] = validate_target_normalization(train_loader, verbose=True)

# STEP 5: Compare to random baseline
results['step5_random_baseline'] = validate_baseline_comparison(model, val_loader, verbose=True)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

all_passed = all(results.values())

for step, passed in results.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {step}")

if all_passed:
    print("\n🎉 ALL VALIDATION STEPS PASSED!")
    print("Your training pipeline is correctly configured.")
    print("\nYou can now proceed with full training.")
else:
    print("\n⚠️  SOME VALIDATION STEPS FAILED!")
    print("Fix the issues above before training your model.")
    print("\nCommon fixes:")
    print("  1. Step 2 fails: Check that targets are correctly normalized")
    print("  2. Step 3 fails: Verify temporal split is working")
    print("  3. Step 4 fails: Adjust normalization in preprocessing")
    print("  4. Step 5 fails: Model architecture or data issues")

print("="*80)
