"""Quick validation check - minimal output"""
import sys
import os
from pathlib import Path
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Silence most output
import warnings
warnings.filterwarnings('ignore')

from src.ml.preprocessing.preprocessor import CryptoPreprocessor
from src.ml.training.dataset import create_dataloaders, validate_data_format, validate_target_normalization
from src.ml.models.model_config import TFTConfig
from src.ml.training.trainer import TFTTrainer
from src.ml.training.metrics import validate_loss_function, validate_baseline_comparison

# Load data silently
preprocessor = CryptoPreprocessor(dataset_dir="dataset")
preprocessor.verbose = False
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
    'data_format': validate_data_format(train_loader, verbose=False),
    'loss_function': validate_loss_function(trainer.model_wrapper.model, train_loader, verbose=False),
    'data_leakage': leakage_ok,
    'target_normalization': validate_target_normalization(train_loader, verbose=False),
    'random_baseline': validate_baseline_comparison(trainer.model_wrapper.model, val_loader, verbose=False)
}

# Print results
print("\n" + "="*80)
print("VALIDATION RESULTS")
print("="*80)
for step, passed in results.items():
    print(f"  [{step}]: {'PASS' if passed else 'FAIL'}")

print("\n" + "="*80)
all_passed = all(results.values())
if all_passed:
    print("VERDICT: Training pipeline is CORRECT - proceed with training")
else:
    print("VERDICT: Issues found - fix before training")
    print(f"Failed: {[k for k, v in results.items() if not v]}")
print("="*80)
