"""
Quick validation test - Run this to validate your training pipeline
"""

import sys
import os
from pathlib import Path

# Fix Windows encoding issues
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.ml.preprocessing.preprocessor import CryptoPreprocessor
from src.ml.training.dataset import create_dataloaders
from src.ml.models.model_config import TFTConfig
from src.ml.training.trainer import TFTTrainer

print("="*80)
print("QUICK VALIDATION TEST")
print("="*80)

try:
    # Load data
    print("\n[TEST] Loading data...")
    preprocessor = CryptoPreprocessor(dataset_dir="dataset")
    symbols = ['ETHUSDT']

    train_df, val_df, test_df = preprocessor.process_all(symbols=symbols)
    train_df = preprocessor.add_time_index(train_df)
    val_df = preprocessor.add_time_index(val_df)
    test_df = preprocessor.add_time_index(test_df)

    # Step 3: Check data leakage
    print("\n" + "="*80)
    print("STEP 3: CHECKING DATA LEAKAGE")
    print("="*80)
    leakage_ok = CryptoPreprocessor.validate_no_data_leakage(train_df, val_df, test_df, verbose=False)
    print(f"[LEAKAGE CHECK] Result: {'PASS' if leakage_ok else 'FAIL'}")

    # Create dataloaders
    print("\n[TEST] Creating dataloaders...")
    config = TFTConfig()
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        context_length=config.max_encoder_length,
        prediction_length=config.max_prediction_length,
        batch_size=config.batch_size,
        verbose=False
    )

    # Create trainer
    print("\n[TEST] Creating model...")
    trainer = TFTTrainer(config=config, verbose=False)
    trainer.setup_model(train_loader.dataset)
    trainer.setup_trainer(gpus=0)

    # Run validation manually (shorter version)
    print("\n" + "="*80)
    print("RUNNING VALIDATION CHECKS")
    print("="*80)

    from src.ml.training.dataset import validate_data_format, validate_target_normalization
    from src.ml.training.metrics import validate_loss_function, validate_baseline_comparison

    results = {}
    results['data_format'] = validate_data_format(train_loader, verbose=False)
    print(f"[STEP 1] Data Format: {'PASS' if results['data_format'] else 'FAIL'}")

    results['loss_function'] = validate_loss_function(trainer.model_wrapper.model, train_loader, verbose=False)
    print(f"[STEP 2] Loss Function: {'PASS' if results['loss_function'] else 'FAIL'}")

    results['data_leakage'] = leakage_ok
    print(f"[STEP 3] Data Leakage: {'PASS' if results['data_leakage'] else 'FAIL'}")

    results['target_normalization'] = validate_target_normalization(train_loader, verbose=False)
    print(f"[STEP 4] Target Normalization: {'PASS' if results['target_normalization'] else 'FAIL'}")

    results['random_baseline'] = validate_baseline_comparison(trainer.model_wrapper.model, val_loader, verbose=False)
    print(f"[STEP 5] Random Baseline: {'PASS' if results['random_baseline'] else 'FAIL'}")

    # Final verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    all_passed = all(results.values())
    if all_passed:
        print(">> ALL CHECKS PASSED - Training pipeline is correctly configured!")
    else:
        print(">> SOME CHECKS FAILED - Fix issues before training")
        failed = [k for k, v in results.items() if not v]
        print(f">> Failed checks: {failed}")

except Exception as e:
    print(f"\nTEST FAILED: {e}")
    import traceback
    traceback.print_exc()
