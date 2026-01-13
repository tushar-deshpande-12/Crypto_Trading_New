"""Debug validation to find exact error"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.ml.preprocessing.preprocessor import CryptoPreprocessor
from src.ml.training.dataset import create_dataloaders
from src.ml.models.model_config import TFTConfig
from src.ml.training.trainer import TFTTrainer

try:
    print("Step 1: Loading data...")
    preprocessor = CryptoPreprocessor(dataset_dir="dataset")
    train_df, val_df, test_df = preprocessor.process_all(symbols=['ETHUSDT'])
    train_df = preprocessor.add_time_index(train_df)
    val_df = preprocessor.add_time_index(val_df)
    test_df = preprocessor.add_time_index(test_df)
    print("  OK\n")

    print("Step 2: Creating dataloaders...")
    config = TFTConfig()
    train_loader, val_loader, _ = create_dataloaders(
        train_df, val_df, test_df,
        context_length=config.max_encoder_length,
        prediction_length=config.max_prediction_length,
        batch_size=config.batch_size,
        verbose=False
    )
    print("  OK\n")

    print("Step 3: Creating trainer...")
    trainer = TFTTrainer(config=config, verbose=False)
    print("  OK\n")

    print("Step 4: Setting up model...")
    trainer.setup_model(train_loader.dataset)
    print("  OK\n")

    print("Step 5: Setting up trainer...")
    trainer.setup_trainer(gpus=0)
    print("  OK\n")

    print("Step 6: Testing model forward pass...")
    batch = next(iter(train_loader))
    print(f"  Batch keys: {batch.keys()}")
    print(f"  encoder_cont shape: {batch['encoder_cont'].shape}")
    print(f"  decoder_target shape: {batch['decoder_target'].shape}")

    model = trainer.model_wrapper.model
    model.eval()
    import torch
    with torch.no_grad():
        output = model(batch)
        print(f"  Output type: {type(output)}")
        if isinstance(output, tuple):
            print(f"  Output is tuple, length: {len(output)}")
            print(f"  output[0] shape: {output[0].shape}")
        elif isinstance(output, dict):
            print(f"  Output is dict, keys: {output.keys()}")
        else:
            print(f"  Output shape: {output.shape}")
    print("  OK\n")

    print("Step 7: Running data format validation...")
    from src.ml.training.dataset import validate_data_format
    result = validate_data_format(train_loader, verbose=True)
    print(f"  Result: {result}\n")

except Exception as e:
    print(f"\nERROR at current step: {e}")
    import traceback
    traceback.print_exc()
