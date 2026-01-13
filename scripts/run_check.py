"""Run validation and save to file"""
import sys
import os
import json
from pathlib import Path

# Set UTF-8 encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.ml.preprocessing.preprocessor import CryptoPreprocessor
from src.ml.training.dataset import create_dataloaders, validate_data_format, validate_target_normalization
from src.ml.models.model_config import TFTConfig
from src.ml.training.trainer import TFTTrainer
from src.ml.training.metrics import validate_loss_function, validate_baseline_comparison

try:
    # Load data
    preprocessor = CryptoPreprocessor(dataset_dir="dataset")
    train_df, val_df, test_df = preprocessor.process_all(symbols=['ETHUSDT'])
    train_df = preprocessor.add_time_index(train_df)
    val_df = preprocessor.add_time_index(val_df)
    test_df = preprocessor.add_time_index(test_df)

    # Check leakage
    leakage_ok = CryptoPreprocessor.validate_no_data_leakage(train_df, val_df, test_df, verbose=False)

    # Create dataloaders
    config = TFTConfig()
    train_loader, val_loader, _ = create_dataloaders(
        train_df, val_df, test_df,
        context_length=config.max_encoder_length,
        prediction_length=config.max_prediction_length,
        batch_size=config.batch_size,
        verbose=False
    )

    # Create model
    trainer = TFTTrainer(config=config, verbose=False)
    trainer.setup_model(train_loader.dataset)
    trainer.setup_trainer(gpus=0)

    # Run validations
    results = {
        'data_format': bool(validate_data_format(train_loader, verbose=False)),
        'loss_function': bool(validate_loss_function(trainer.model_wrapper.model, train_loader, verbose=False)),
        'data_leakage': bool(leakage_ok),
        'target_normalization': bool(validate_target_normalization(train_loader, verbose=False)),
        'random_baseline': bool(validate_baseline_comparison(trainer.model_wrapper.model, val_loader, verbose=False))
    }

    # Save to file
    with open('validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\nVALIDATION COMPLETE - Results saved to validation_results.json")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    all_passed = all(results.values())
    print(f"\nVERDICT: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
