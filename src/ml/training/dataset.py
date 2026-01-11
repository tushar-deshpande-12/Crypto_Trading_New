"""
PyTorch Dataset for Cryptocurrency Time Series
Compatible with pytorch-forecasting's TimeSeriesDataSet
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Note: torch and pytorch_forecasting need to be installed
# pip install torch pytorch-lightning pytorch-forecasting
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    # Suppress triton warnings (optional dependency for GPU kernels)
    warnings.filterwarnings('ignore', message='.*triton not found.*')

    TORCH_AVAILABLE = True
except ImportError:
    print("[DATASET] ⚠ Warning: PyTorch not installed. Install with: pip install torch pytorch-lightning pytorch-forecasting")
    TORCH_AVAILABLE = False


class CryptoTimeSeriesDataset:
    """
    Time series dataset for cryptocurrency prediction using TFT

    This class creates a pytorch-forecasting TimeSeriesDataSet which is
    optimized for Temporal Fusion Transformer training.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        context_length: int = 2000,
        prediction_length: int = 10,
        target_column: str = 'close',
        time_varying_known_reals: Optional[List[str]] = None,
        time_varying_unknown_reals: Optional[List[str]] = None,
        static_categoricals: Optional[List[str]] = None,
        verbose: bool = True
    ):
        """
        Initialize cryptocurrency time series dataset

        Args:
            data: Preprocessed DataFrame with all features
            context_length: Length of encoder (input) sequence
            prediction_length: Length of decoder (prediction) sequence
            target_column: Column to predict (default: 'close')
            time_varying_known_reals: Features known in the future (e.g., time features)
            time_varying_unknown_reals: Features unknown in the future (e.g., price, volume)
            static_categoricals: Static categorical features (e.g., symbol)
            verbose: Print progress information
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Please install: pip install torch pytorch-lightning pytorch-forecasting")

        if verbose:
            print(f"\n[DATASET] Initializing CryptoTimeSeriesDataset")
            print(f"[DATASET]   - Input shape: {data.shape}")
            print(f"[DATASET]   - Context length: {context_length}")
            print(f"[DATASET]   - Prediction length: {prediction_length}")
            print(f"[DATASET]   - Target column: {target_column}")

        self.data = data.copy()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.target_column = target_column
        self.verbose = verbose

        # Prepare data
        self._prepare_data()

        # Set default features if not provided
        if time_varying_known_reals is None:
            time_varying_known_reals = self._get_temporal_features()

        if time_varying_unknown_reals is None:
            time_varying_unknown_reals = self._get_price_volume_features()

        if static_categoricals is None:
            static_categoricals = ['symbol'] if 'symbol' in self.data.columns else []

        if verbose:
            print(f"[DATASET]   - Time-varying known reals: {len(time_varying_known_reals)} features")
            print(f"[DATASET]   - Time-varying unknown reals: {len(time_varying_unknown_reals)} features")
            print(f"[DATASET]   - Static categoricals: {static_categoricals}")

        self.time_varying_known_reals = time_varying_known_reals
        self.time_varying_unknown_reals = time_varying_unknown_reals
        self.static_categoricals = static_categoricals

        # Create TimeSeriesDataSet
        self.dataset = self._create_time_series_dataset()

        if verbose:
            print(f"[DATASET] ✓ Dataset initialized")
            print(f"[DATASET]   - Total sequences: {len(self.dataset)}")

    def _prepare_data(self):
        """Prepare data for TimeSeriesDataSet"""
        if self.verbose:
            print(f"\n[DATASET] Preparing data...")

        # Ensure datetime is datetime type
        if 'datetime' in self.data.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.data['datetime']):
                if self.verbose:
                    print(f"[DATASET]   - Converting datetime column to datetime type")
                self.data['datetime'] = pd.to_datetime(self.data['datetime'])

        # Create time index (required by pytorch-forecasting)
        if 'time_idx' not in self.data.columns:
            if self.verbose:
                print(f"[DATASET]   - Creating time index...")

            # Create time_idx per symbol
            self.data['time_idx'] = 0

            for symbol in self.data['symbol'].unique():
                mask = self.data['symbol'] == symbol
                self.data.loc[mask, 'time_idx'] = range(mask.sum())
        else:
            if self.verbose:
                print(f"[DATASET]   - Time index already exists (range: {self.data['time_idx'].min()} to {self.data['time_idx'].max()})")

        # Ensure symbol is string type
        if 'symbol' in self.data.columns:
            self.data['symbol'] = self.data['symbol'].astype(str)

        # Verify required columns
        required_cols = ['time_idx', self.target_column]
        missing_cols = [col for col in required_cols if col not in self.data.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if self.verbose:
            print(f"[DATASET] ✓ Data prepared")
            print(f"[DATASET]   - Columns: {list(self.data.columns)}")
            print(f"[DATASET]   - Time index range: {self.data['time_idx'].min()} to {self.data['time_idx'].max()}")

    def _get_temporal_features(self) -> List[str]:
        """Get list of temporal features (known in advance)"""
        temporal_patterns = ['hour', 'day', 'month', '_sin', '_cos']
        temporal_features = []

        for col in self.data.columns:
            if any(pattern in col for pattern in temporal_patterns):
                # Exclude raw hour/day/month (we want the sin/cos versions)
                if col in ['hour', 'day_of_week', 'day_of_month', 'month']:
                    continue
                temporal_features.append(col)

        if self.verbose:
            print(f"[DATASET]   - Auto-detected temporal features: {temporal_features}")

        return temporal_features

    def _get_price_volume_features(self) -> List[str]:
        """Get list of price/volume features (unknown in advance)"""
        # Include OHLCV and derived features
        price_volume_patterns = [
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'log_volume', 'price_range',
            'trade', 'taker', 'quote',
            '_lag_', 'rolling_'
        ]

        price_volume_features = []

        for col in self.data.columns:
            if any(pattern in col for pattern in price_volume_patterns):
                price_volume_features.append(col)

        # Remove duplicates and ensure target is included
        price_volume_features = list(set(price_volume_features))

        if self.target_column not in price_volume_features:
            price_volume_features.append(self.target_column)

        if self.verbose:
            print(f"[DATASET]   - Auto-detected price/volume features: {len(price_volume_features)} features")

        return price_volume_features

    def _create_time_series_dataset(self) -> TimeSeriesDataSet:
        """Create pytorch-forecasting TimeSeriesDataSet"""
        if self.verbose:
            print(f"\n[DATASET] Creating TimeSeriesDataSet...")

        # Find maximum time_idx to determine which samples can be used
        max_time_idx = self.data.groupby('symbol')['time_idx'].transform('max')
        training_cutoff = max_time_idx - self.prediction_length

        if self.verbose:
            print(f"[DATASET]   - Max time index: {self.data['time_idx'].max()}")
            print(f"[DATASET]   - Training cutoff: {training_cutoff.max()}")

        try:
            dataset = TimeSeriesDataSet(
                self.data,
                time_idx='time_idx',
                target=self.target_column,
                group_ids=['symbol'] if 'symbol' in self.static_categoricals else [],
                max_encoder_length=self.context_length,
                max_prediction_length=self.prediction_length,
                time_varying_known_reals=self.time_varying_known_reals,
                time_varying_unknown_reals=self.time_varying_unknown_reals,
                static_categoricals=self.static_categoricals,
                target_normalizer=GroupNormalizer(
                    groups=['symbol'] if 'symbol' in self.static_categoricals else [],
                    transformation='softplus'  # Ensures positive predictions
                ),
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
            )

            if self.verbose:
                print(f"[DATASET] ✓ TimeSeriesDataSet created successfully")

            return dataset

        except Exception as e:
            print(f"[DATASET] ✗ Failed to create TimeSeriesDataSet: {e}")
            raise

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True
    ) -> DataLoader:
        """
        Create DataLoader for training

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Pin memory for faster GPU transfer

        Returns:
            DataLoader instance
        """
        if self.verbose:
            print(f"\n[DATASET] Creating DataLoader...")
            print(f"[DATASET]   - Batch size: {batch_size}")
            print(f"[DATASET]   - Shuffle: {shuffle}")
            print(f"[DATASET]   - Num workers: {num_workers}")

        try:
            dataloader = self.dataset.to_dataloader(
                train=shuffle,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory
            )

            if self.verbose:
                print(f"[DATASET] ✓ DataLoader created")
                print(f"[DATASET]   - Total batches: {len(dataloader)}")

            return dataloader

        except Exception as e:
            print(f"[DATASET] ✗ Failed to create DataLoader: {e}")
            raise


def create_dataloaders(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    context_length: int = 2000,
    prediction_length: int = 10,
    batch_size: int = 64,
    verbose: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders

    Args:
        train_data: Training DataFrame
        val_data: Validation DataFrame
        test_data: Test DataFrame
        context_length: Encoder length
        prediction_length: Prediction horizon
        batch_size: Batch size
        verbose: Print progress

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    print(f"\n[DATASET] Creating train/val/test DataLoaders...")
    print(f"[DATASET]   - Train samples: {len(train_data)}")
    print(f"[DATASET]   - Val samples: {len(val_data)}")
    print(f"[DATASET]   - Test samples: {len(test_data)}")

    # Create training dataset
    print(f"\n[DATASET] Creating training dataset...")
    train_dataset = CryptoTimeSeriesDataset(
        train_data,
        context_length=context_length,
        prediction_length=prediction_length,
        verbose=verbose
    )

    # Create validation dataset (using same parameters as training)
    # CRITICAL FIX #2: Use predict=True and stop_randomization=True to prevent data augmentation
    print(f"\n[DATASET] Creating validation dataset...")
    print(f"[DATASET]   ℹ️  Using predict=True to inherit TRAINING normalizer (no refit)")
    print(f"[DATASET]   ℹ️  Using stop_randomization=True to prevent augmentation")
    val_dataset_obj = TimeSeriesDataSet.from_dataset(
        train_dataset.dataset,
        val_data,
        predict=True,  # Inherit training dataset's normalizer - NO REFITTING
        stop_randomization=True  # No data augmentation on validation
    )

    # Create test dataset
    print(f"\n[DATASET] Creating test dataset...")
    print(f"[DATASET]   ℹ️  Using predict=True to inherit TRAINING normalizer (no refit)")
    print(f"[DATASET]   ℹ️  Using stop_randomization=True to prevent augmentation")
    test_dataset_obj = TimeSeriesDataSet.from_dataset(
        train_dataset.dataset,
        test_data,
        predict=True,  # Inherit training dataset's normalizer - NO REFITTING
        stop_randomization=True  # No data augmentation on test
    )

    # Create DataLoaders
    print(f"\n[DATASET] Creating DataLoaders...")

    # Training loader: shuffle=True, augmentation enabled
    train_loader = train_dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=True,  # ✅ Shuffle training data
        num_workers=0  # Set to 0 to avoid multiprocessing issues on Windows
    )
    print(f"[DATASET]   ✅ Train loader: shuffle=True (training mode)")

    # CRITICAL FIX #2: Validation loader must NOT shuffle data
    val_loader = val_dataset_obj.to_dataloader(
        train=False,  # ✅ NO shuffling, NO augmentation
        batch_size=batch_size,
        num_workers=0
    )
    print(f"[DATASET]   ✅ Validation loader: train=False (no shuffle, no augmentation)")

    # Test loader: same as validation
    test_loader = test_dataset_obj.to_dataloader(
        train=False,  # ✅ NO shuffling, NO augmentation
        batch_size=batch_size,
        num_workers=0
    )
    print(f"[DATASET]   ✅ Test loader: train=False (no shuffle, no augmentation)")

    print(f"\n[DATASET] ✓ All DataLoaders created successfully")
    print(f"[DATASET]   - Train batches: {len(train_loader)}")
    print(f"[DATASET]   - Val batches: {len(val_loader)}")
    print(f"[DATASET]   - Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


# ============================================================================
# VALIDATION FUNCTIONS FOR DATA PIPELINE
# ============================================================================

def validate_data_format(dataloader, verbose: bool = True) -> bool:
    """
    STEP 1: Validate that inputs and targets are correctly formatted

    Args:
        dataloader: DataLoader to validate
        verbose: Print detailed output

    Returns:
        True if data format is valid
    """
    try:
        import torch
    except ImportError:
        print("[VALIDATION] PyTorch not available, skipping data format validation")
        return False

    if verbose:
        print(f"\n{'='*80}")
        print("STEP 1: VALIDATING DATA FORMAT")
        print('='*80)

    issues = []

    # Get first batch
    for batch_idx, batch_data in enumerate(dataloader):
        # Handle both tuple and dict batch formats
        if isinstance(batch_data, tuple):
            batch = batch_data[0] if len(batch_data) > 0 else batch_data
        else:
            batch = batch_data

        x = batch['encoder_cont']
        y_target = batch['decoder_target']

        if verbose:
            print(f"\n[DATA FORMAT] Batch {batch_idx} shapes:")
            print(f"  - Input (encoder_cont): {x.shape}")
            print(f"  - Target (decoder_target): {y_target.shape}")

        # Check for NaN
        if torch.isnan(x).any():
            nan_count = torch.isnan(x).sum().item()
            issues.append(f"❌ Batch {batch_idx}: Input has {nan_count} NaN values")
            if verbose:
                print(f"  ❌ Input contains {nan_count} NaN values!")

        if torch.isinf(x).any():
            inf_count = torch.isinf(x).sum().item()
            issues.append(f"❌ Batch {batch_idx}: Input has {inf_count} Inf values")
            if verbose:
                print(f"  ❌ Input contains {inf_count} Inf values!")

        if torch.isnan(y_target).any():
            nan_count = torch.isnan(y_target).sum().item()
            issues.append(f"❌ Batch {batch_idx}: Target has {nan_count} NaN values")
            if verbose:
                print(f"  ❌ Target contains {nan_count} NaN values!")

        if torch.isinf(y_target).any():
            inf_count = torch.isinf(y_target).sum().item()
            issues.append(f"❌ Batch {batch_idx}: Target has {inf_count} Inf values")
            if verbose:
                print(f"  ❌ Target contains {inf_count} Inf values!")

        # Check value ranges
        if verbose:
            print(f"\n[DATA FORMAT] Value statistics:")
            print(f"  - Input  min/max: {x.min().item():.4f} / {x.max().item():.4f}")
            print(f"  - Input  mean/std: {x.mean().item():.4f} / {x.std().item():.4f}")
            print(f"  - Target min/max: {y_target.min().item():.4f} / {y_target.max().item():.4f}")
            print(f"  - Target mean/std: {y_target.mean().item():.4f} / {y_target.std().item():.4f}")

        # Only check first batch
        break

    if issues:
        if verbose:
            print(f"\n❌ STEP 1 FAILED: Found {len(issues)} issues")
            for issue in issues:
                print(f"  {issue}")
        return False
    else:
        if verbose:
            print(f"\n✅ STEP 1 PASSED: Data format is valid")
        return True


def validate_target_normalization(dataloader, verbose: bool = True) -> bool:
    """
    STEP 4: Validate that target values are properly normalized

    Args:
        dataloader: DataLoader to check
        verbose: Print detailed output

    Returns:
        True if targets look reasonable
    """
    try:
        import torch
    except ImportError:
        print("[VALIDATION] PyTorch not available, skipping target validation")
        return False

    if verbose:
        print(f"\n{'='*80}")
        print("STEP 4: VALIDATING TARGET NORMALIZATION")
        print('='*80)

    all_targets = []

    # Collect target statistics
    for batch_data in dataloader:
        # Handle both tuple and dict batch formats
        if isinstance(batch_data, tuple):
            batch = batch_data[0] if len(batch_data) > 0 else batch_data
        else:
            batch = batch_data

        target = batch['decoder_target']
        all_targets.append(target)

    all_targets = torch.cat(all_targets, dim=0)

    target_mean = all_targets.mean().item()
    target_std = all_targets.std().item()
    target_min = all_targets.min().item()
    target_max = all_targets.max().item()

    if verbose:
        print(f"\n[TARGET VALIDATION] Statistics across all batches:")
        print(f"  - Mean: {target_mean:.4f}")
        print(f"  - Std:  {target_std:.4f}")
        print(f"  - Min:  {target_min:.4f}")
        print(f"  - Max:  {target_max:.4f}")

    issues = []

    # Check if targets collapsed
    if target_std < 1e-6:
        issues.append("❌ Target std nearly zero - all targets are same value!")
        if verbose:
            print(f"\n  ❌ CRITICAL: Targets collapsed to constant value!")

    # Warning for very large values (might not be normalized)
    if abs(target_mean) > 10000:
        issues.append(f"⚠️  Target mean very large ({target_mean:.2f})")
        if verbose:
            print(f"\n  ⚠️  WARNING: Target mean is very large, check normalization")

    if target_std > 10000 or target_std < 0.001:
        issues.append(f"⚠️  Unusual target std ({target_std:.4f})")
        if verbose:
            print(f"\n  ⚠️  WARNING: Target std is {target_std:.4f}, verify normalization")

    if issues:
        if verbose:
            print(f"\n⚠️  STEP 4 WARNING: Found {len(issues)} potential issues")
        # Return True if only warnings, False if critical
        return not any("❌" in issue for issue in issues)
    else:
        if verbose:
            print(f"\n✅ STEP 4 PASSED: Target normalization looks reasonable")
        return True


if __name__ == "__main__":
    # Test dataset creation
    print("Testing CryptoTimeSeriesDataset")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("PyTorch not installed. Skipping test.")
        print("Install with: pip install torch pytorch-lightning pytorch-forecasting")
    else:
        # Create sample data
        print("\n[TEST] Creating sample data...")

        dates = pd.date_range('2020-01-01', periods=10000, freq='1H')
        n_samples = len(dates)

        sample_df = pd.DataFrame({
            'datetime': dates,
            'symbol': 'ETHUSDT',
            'close': np.random.randn(n_samples).cumsum() + 3000,
            'open': np.random.randn(n_samples).cumsum() + 3000,
            'high': np.random.randn(n_samples).cumsum() + 3050,
            'low': np.random.randn(n_samples).cumsum() + 2950,
            'volume': np.random.rand(n_samples) * 1000,
            'hour_sin': np.sin(2 * np.pi * np.arange(n_samples) / 24),
            'hour_cos': np.cos(2 * np.pi * np.arange(n_samples) / 24),
        })

        print(f"[TEST]   - Created sample data: {sample_df.shape}")

        try:
            # Create dataset
            dataset = CryptoTimeSeriesDataset(
                sample_df,
                context_length=1000,  # Smaller for testing
                prediction_length=10,
                verbose=True
            )

            # Create dataloader
            dataloader = dataset.get_dataloader(batch_size=4)

            # Test iteration
            print(f"\n[TEST] Testing DataLoader iteration...")
            for i, batch in enumerate(dataloader):
                print(f"[TEST]   - Batch {i}: {type(batch)}")

                if i >= 2:  # Only test first 3 batches
                    break

            print(f"\n[TEST] ✓ Dataset test passed!")

        except Exception as e:
            print(f"\n[TEST] ✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
