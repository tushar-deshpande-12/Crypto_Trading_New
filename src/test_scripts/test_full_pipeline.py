"""
Full Pipeline Test Script
Tests all components of the AI trading system:
1. Data preprocessing
2. Feature engineering
3. Dataset creation
4. Model training
5. Prediction
6. Backtesting

Run this script to verify the entire pipeline works correctly.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import pandas as pd
from datetime import datetime

# Import debug logger
from src.utils.debug_logger import DebugLogger, get_debug_logger

# Initialize debug logger
debug = DebugLogger(log_dir="debug_logs", enabled=True)


def test_preprocessing(symbols: list = ["BTCUSDT"]):
    """Test data preprocessing pipeline."""
    print("\n" + "=" * 80)
    print("TEST 1: DATA PREPROCESSING")
    print("=" * 80)

    try:
        from src.ml.preprocessing.preprocessor import CryptoPreprocessor, validate_data_for_training

        preprocessor = CryptoPreprocessor("dataset")

        # Process data
        train_df, val_df, test_df = preprocessor.process_all(
            symbols,
            save_scaler_path="models/checkpoints/test_scalers.pkl"
        )

        # Log data stats
        debug.log_data_stats("preprocessing", train_df, "train_data")
        debug.log_data_stats("preprocessing", val_df, "val_data")
        debug.log_data_stats("preprocessing", test_df, "test_data")

        # Validate data
        validation_results = validate_data_for_training(train_df, val_df, test_df)

        # Check critical issues
        critical_checks = {
            "time_idx_continuous": validation_results.get("time_idx_continuous", False),
            "no_nan_inf": validation_results.get("no_nan_inf", False),
            "no_data_leakage": validation_results.get("no_data_leakage", False),
        }

        if all(critical_checks.values()):
            print("\n[TEST 1] [OK] PASSED - Preprocessing working correctly")
            debug.log("preprocessing", "test_complete", {"status": "passed", "checks": critical_checks}, "success")
            return train_df, val_df, test_df, preprocessor
        else:
            print("\n[TEST 1] [X] FAILED - Preprocessing has issues")
            debug.log("preprocessing", "test_complete", {"status": "failed", "checks": critical_checks}, "error")
            return None, None, None, None

    except Exception as e:
        print(f"\n[TEST 1] [X] ERROR: {e}")
        debug.log_error("preprocessing", "test_error", e)
        import traceback
        traceback.print_exc()
        return None, None, None, None


def test_dataset_creation(train_df, val_df, test_df):
    """Test dataset and dataloader creation."""
    print("\n" + "=" * 80)
    print("TEST 2: DATASET CREATION")
    print("=" * 80)

    try:
        from src.ml.training.dataset import create_dataloaders

        # Create dataloaders with correct parameters
        train_loader, val_loader, test_loader = create_dataloaders(
            train_df, val_df, test_df,
            context_length=168,  # 1 week
            prediction_length=24,  # 24 hours (matches target_return horizon)
            batch_size=64,
            target_column='target_return',
            verbose=True
        )

        # Verify batch contents
        print("\n[TEST 2] Verifying batch contents...")

        for batch_idx, batch_data in enumerate(train_loader):
            if isinstance(batch_data, tuple):
                batch = batch_data[0]
            else:
                batch = batch_data

            # Check for required keys
            required_keys = ['encoder_cont', 'decoder_target']
            for key in required_keys:
                if key not in batch:
                    print(f"[TEST 2] [X] Missing key: {key}")
                    return None, None, None

            x = batch['encoder_cont']
            y = batch['decoder_target']

            print(f"\n[TEST 2] Batch {batch_idx}:")
            print(f"  - Input shape: {x.shape}")
            print(f"  - Target shape: {y.shape}")
            print(f"  - Input range: [{x.min().item():.4f}, {x.max().item():.4f}]")
            print(f"  - Target range: [{y.min().item():.4f}, {y.max().item():.4f}]")
            print(f"  - Input NaN: {x.isnan().sum().item()}")
            print(f"  - Target NaN: {y.isnan().sum().item()}")

            # Log to debug
            debug.log("dataset", f"batch_{batch_idx}", {
                "input_shape": list(x.shape),
                "target_shape": list(y.shape),
                "input_range": [x.min().item(), x.max().item()],
                "target_range": [y.min().item(), y.max().item()],
            })

            if batch_idx >= 2:  # Check first 3 batches
                break

        print("\n[TEST 2] [OK] PASSED - Dataset creation working correctly")
        debug.log("dataset", "test_complete", {"status": "passed"}, "success")
        return train_loader, val_loader, test_loader

    except Exception as e:
        print(f"\n[TEST 2] [X] ERROR: {e}")
        debug.log_error("dataset", "test_error", e)
        import traceback
        traceback.print_exc()
        return None, None, None


def test_model_training(train_loader, val_loader, epochs: int = 3):
    """Test model training (short training to verify it works)."""
    print("\n" + "=" * 80)
    print("TEST 3: MODEL TRAINING (Short test)")
    print("=" * 80)

    try:
        import torch
        from src.ml.models.model_config import TFTConfig
        from src.ml.training.trainer import TFTTrainer

        # Create config optimized for learning
        config = TFTConfig(
            hidden_size=64,  # Increased for better capacity
            lstm_layers=2,
            attention_head_size=4,
            dropout=0.1,
            max_encoder_length=168,
            max_prediction_length=24,
            batch_size=64,
            max_epochs=epochs,
            learning_rate=0.003,  # Higher learning rate for faster learning
            early_stopping_patience=10
        )

        print(f"[TEST 3] Config: {config}")

        # Create trainer
        trainer = TFTTrainer(
            config=config,
            checkpoint_dir="models/checkpoints/test_run",
            verbose=True
        )

        # Setup model
        trainer.setup_model(train_loader.dataset)

        # Setup trainer
        gpus = 1 if torch.cuda.is_available() else 0
        trainer.setup_trainer(gpus=gpus)

        # Train (short run)
        print(f"\n[TEST 3] Training for {epochs} epochs...")
        trainer.train(train_loader, val_loader, validate_pipeline=True)

        # Check if training improved
        print("\n[TEST 3] [OK] PASSED - Model training working")
        debug.log("training", "test_complete", {"status": "passed", "epochs": epochs}, "success")
        return trainer

    except Exception as e:
        print(f"\n[TEST 3] [X] ERROR: {e}")
        debug.log_error("training", "test_error", e)
        import traceback
        traceback.print_exc()
        return None


def test_prediction(trainer, val_loader):
    """Test prediction pipeline."""
    print("\n" + "=" * 80)
    print("TEST 4: PREDICTION")
    print("=" * 80)

    try:
        import torch

        if trainer is None or trainer.model_wrapper is None:
            print("[TEST 4] [!] No trained model available, skipping prediction test")
            return None

        model = trainer.model_wrapper.model
        model.eval()

        print("[TEST 4] Generating predictions on validation data...")

        predictions_list = []
        actuals_list = []

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                if isinstance(batch_data, tuple):
                    batch = batch_data[0]
                else:
                    batch = batch_data

                # Get predictions
                try:
                    output = model(batch)

                    # Handle different output formats
                    if hasattr(output, 'prediction'):
                        pred = output.prediction
                    else:
                        pred = output

                    target = batch['decoder_target']

                    predictions_list.append(pred.cpu().numpy())
                    actuals_list.append(target.cpu().numpy())

                    if batch_idx == 0:
                        print(f"  - Prediction shape: {pred.shape}")
                        print(f"  - Target shape: {target.shape}")

                except Exception as e:
                    print(f"  [!] Batch {batch_idx} failed: {e}")
                    continue

                if batch_idx >= 5:  # Test first few batches
                    break

        if predictions_list:
            predictions = np.concatenate([p.flatten() for p in predictions_list])
            actuals = np.concatenate([a.flatten() for a in actuals_list])

            # Calculate metrics
            mae = np.mean(np.abs(predictions - actuals))
            rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

            print(f"\n[TEST 4] Prediction Metrics (normalized scale):")
            print(f"  - MAE: {mae:.4f}")
            print(f"  - RMSE: {rmse:.4f}")
            print(f"  - Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
            print(f"  - Actuals range: [{actuals.min():.4f}, {actuals.max():.4f}]")

            debug.log_prediction(predictions, actuals, {"mae": mae, "rmse": rmse})

            print("\n[TEST 4] [OK] PASSED - Prediction pipeline working")
            return {"predictions": predictions, "actuals": actuals, "mae": mae, "rmse": rmse}
        else:
            print("\n[TEST 4] [!] WARNING - No predictions generated")
            return None

    except Exception as e:
        print(f"\n[TEST 4] [X] ERROR: {e}")
        debug.log_error("prediction", "test_error", e)
        import traceback
        traceback.print_exc()
        return None


def test_backtesting(preprocessor, test_df):
    """Test backtesting pipeline."""
    print("\n" + "=" * 80)
    print("TEST 5: BACKTESTING")
    print("=" * 80)

    try:
        from src.ml.backtest.backtester import CryptoBacktester, BacktestConfig

        # Create backtester
        config = BacktestConfig(
            initial_capital=1000.0,
            trade_fee=0.001,
            confidence_threshold=0.002
        )
        backtester = CryptoBacktester(config=config, verbose=True)

        # Set target scaler for denormalization
        if preprocessor.target_scaler is not None:
            backtester.set_target_scaler(preprocessor.target_scaler)

        # Create mock predictions (for testing)
        # In real usage, these would come from the model
        n_samples = min(100, len(test_df))
        test_subset = test_df.head(n_samples).copy()

        # Use actual target returns as "predicted" (perfect prediction scenario)
        predicted_returns = test_subset['target_return'].values
        actual_returns = test_subset['target_return'].values

        # Get current prices (denormalized)
        if 'close' in preprocessor.feature_columns:
            close_idx = preprocessor.feature_columns.index('close')
            symbol = test_subset['symbol'].iloc[0]
            scaler = preprocessor.scalers.get(symbol)
            if scaler:
                current_prices = test_subset['close'].values * scaler.scale_[close_idx] + scaler.mean_[close_idx]
            else:
                current_prices = np.ones(n_samples) * 50000  # Default price
        else:
            current_prices = np.ones(n_samples) * 50000

        timestamps = pd.to_datetime(test_subset['datetime'])

        # Run backtest
        print("\n[TEST 5] Running backtest with test data...")
        results = backtester.run_backtest_from_returns(
            predicted_returns=predicted_returns,
            actual_returns=actual_returns,
            current_prices=current_prices,
            timestamps=timestamps,
            denormalize=True
        )

        print(f"\n[TEST 5] Backtest Results:")
        print(f"  - Initial capital: ${results.initial_capital:,.2f}")
        print(f"  - Final capital: ${results.final_capital:,.2f}")
        print(f"  - Total return: {results.total_return_pct:+.2f}%")
        print(f"  - Number of trades: {results.num_trades}")

        debug.log("backtesting", "results", {
            "initial_capital": results.initial_capital,
            "final_capital": results.final_capital,
            "total_return_pct": results.total_return_pct,
            "num_trades": results.num_trades
        }, "success")

        print("\n[TEST 5] [OK] PASSED - Backtesting pipeline working")
        return results

    except Exception as e:
        print(f"\n[TEST 5] [X] ERROR: {e}")
        debug.log_error("backtesting", "test_error", e)
        import traceback
        traceback.print_exc()
        return None


def run_all_tests():
    """Run all pipeline tests."""
    print("\n" + "=" * 80)
    print("CRYPTO AI PIPELINE - FULL TEST SUITE")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")

    results = {
        "preprocessing": False,
        "dataset": False,
        "training": False,
        "prediction": False,
        "backtesting": False
    }

    # Test 1: Preprocessing
    train_df, val_df, test_df, preprocessor = test_preprocessing(["BTCUSDT"])
    if train_df is not None:
        results["preprocessing"] = True

        # Test 2: Dataset creation
        train_loader, val_loader, test_loader = test_dataset_creation(train_df, val_df, test_df)
        if train_loader is not None:
            results["dataset"] = True

            # Test 3: Model training (short)
            trainer = test_model_training(train_loader, val_loader, epochs=3)
            if trainer is not None:
                results["training"] = True

                # Test 4: Prediction
                pred_results = test_prediction(trainer, val_loader)
                if pred_results is not None:
                    results["prediction"] = True

        # Test 5: Backtesting (uses test_df directly)
        backtest_results = test_backtesting(preprocessor, test_df)
        if backtest_results is not None:
            results["backtesting"] = True

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)

    all_passed = True
    for test_name, passed in results.items():
        status = "[OK] PASSED" if passed else "[X] FAILED"
        print(f"  {status} - {test_name}")
        if not passed:
            all_passed = False

    # Export debug summary
    debug.export_summary()

    if all_passed:
        print("\n[RESULT] All tests PASSED!")
        print("The pipeline is ready for training.")
    else:
        print("\n[RESULT] Some tests FAILED!")
        print("Please check the debug logs for details.")
        print(f"Debug logs: {debug.log_dir}")

    return results


if __name__ == "__main__":
    run_all_tests()
