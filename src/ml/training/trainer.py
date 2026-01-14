"""
TFT Training Pipeline
Orchestrates model training with PyTorch Lightning
"""

import time
import warnings
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from datetime import datetime

try:
    import torch
    from lightning.pytorch import Trainer as LightningTrainer
    from lightning.pytorch import LightningModule
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    from lightning.pytorch.loggers import TensorBoardLogger

    # Suppress triton warnings (optional dependency for GPU kernels)
    warnings.filterwarnings('ignore', message='.*triton not found.*')

    # Import GUI callback
    from .gui_callback import GUIProgressCallback

    # Check Lightning version
    import importlib.metadata
    try:
        lightning_version = importlib.metadata.version('lightning')
        print(f"[TRAINER] Lightning version: {lightning_version}")
    except:
        pass

    TORCH_AVAILABLE = True
except ImportError:
    print("[TRAINER] [!] Warning: PyTorch/Lightning not installed")
    TORCH_AVAILABLE = False

from ..models import TFTConfig
from ..models.tft_model import CryptoTFT
from .metrics import MetricsTracker, calculate_all_metrics


class TFTTrainer:
    """
    Trainer for Temporal Fusion Transformer model

    Handles:
    - Model creation and initialization
    - Training loop with validation
    - Checkpointing and early stopping
    - Metrics tracking and logging
    - TensorBoard integration
    """

    def __init__(
        self,
        config: TFTConfig,
        checkpoint_dir: str = "models/checkpoints",
        log_dir: str = "logs/training",
        verbose: bool = True,
        progress_callback: Optional[callable] = None,
        root: Optional[any] = None
    ):
        """
        Initialize TFT trainer

        Args:
            config: TFTConfig with hyperparameters
            checkpoint_dir: Directory for model checkpoints
            log_dir: Directory for training logs
            verbose: Print detailed progress
            progress_callback: Optional GUI callback(epoch, max_epochs, train_loss, val_loss)
            root: Optional Tkinter root for thread-safe GUI updates
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        self.verbose = verbose
        self.progress_callback = progress_callback
        self.root = root

        if self.verbose:
            print(f"\n[TRAINER] Initializing TFTTrainer")
            print(f"[TRAINER]   - Checkpoint directory: {checkpoint_dir}")
            print(f"[TRAINER]   - Log directory: {log_dir}")
            if progress_callback:
                print(f"[TRAINER]   - GUI progress callback: Registered")

        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.model_wrapper: Optional[CryptoTFT] = None
        self.pl_trainer: Optional[LightningTrainer] = None
        self.metrics_tracker = MetricsTracker()

        # Training state
        self.training_start_time: Optional[float] = None
        self.current_epoch: int = 0

        if self.verbose:
            print(f"[TRAINER] [OK] TFTTrainer initialized")

    def setup_model(self, train_dataset, pretrained_path: Optional[str] = None):
        """
        Setup model from dataset or load from pretrained checkpoint

        Args:
            train_dataset: TimeSeriesDataSet for training
            pretrained_path: Optional path to pretrained checkpoint for fine-tuning
        """
        if self.verbose:
            print(f"\n[TRAINER] Setting up model...")

        try:
            # Create model wrapper
            self.model_wrapper = CryptoTFT(self.config, verbose=self.verbose)

            if pretrained_path and Path(pretrained_path).exists():
                # Load from pretrained checkpoint
                if self.verbose:
                    print(f"[TRAINER]   - Loading pretrained model from: {pretrained_path}")

                from pytorch_forecasting import TemporalFusionTransformer
                pretrained_model = TemporalFusionTransformer.load_from_checkpoint(pretrained_path)
                self.model_wrapper.model = pretrained_model

                if self.verbose:
                    print(f"[TRAINER] [OK] Loaded pretrained model for fine-tuning")
            else:
                # Create model from dataset (train from scratch)
                self.model_wrapper.create_from_dataset(train_dataset)

                if self.verbose:
                    print(f"[TRAINER] [OK] Created new model from scratch")

            if self.verbose:
                print(f"[TRAINER] [OK] Model setup complete")

        except Exception as e:
            print(f"[TRAINER] [X] Model setup failed: {e}")
            raise

    def setup_trainer(
        self,
        gpus: int = 0,
        enable_progress_bar: bool = True,
        enable_tensorboard: bool = True,
        verbose_callbacks: bool = True
    ):
        """
        Setup PyTorch Lightning trainer

        Args:
            gpus: Number of GPUs (0 = CPU only)
            enable_progress_bar: Show training progress bar
            enable_tensorboard: Enable TensorBoard logging
        """
        if self.verbose:
            print(f"\n[TRAINER] Setting up PyTorch Lightning trainer...")
            print(f"[TRAINER]   - GPUs: {gpus}")
            print(f"[TRAINER]   - Progress bar: {enable_progress_bar}")
            print(f"[TRAINER]   - TensorBoard: {enable_tensorboard}")

        try:
            # Create timestamp for this training run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"tft_{timestamp}"

            if self.verbose:
                print(f"[TRAINER]   - Run name: {run_name}")

            # Checkpoint callback
            checkpoint_callback = ModelCheckpoint(
                dirpath=self.checkpoint_dir / run_name,
                filename='tft-{epoch:02d}-{val_loss:.4f}',
                save_top_k=3,
                monitor='val_loss',
                mode='min',
                save_last=True,
                verbose=verbose_callbacks
            )

            # Early stopping callback
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                mode='min',
                verbose=verbose_callbacks
            )

            # Learning rate monitor
            lr_monitor = LearningRateMonitor(logging_interval='epoch')

            # Standard callbacks
            callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]

            # Add advanced training callbacks for better performance
            # Temporarily disabled due to PyTorch Lightning 2.x compatibility
            # Will be re-enabled after fixing callback signatures
            if self.verbose:
                print(f"[TRAINER]   [!] Advanced callbacks temporarily disabled")
                print(f"[TRAINER]       (PyTorch Lightning 2.x compatibility fix in progress)")

            # TODO: Re-enable after fixing callback signatures for Lightning 2.x
            # try:
            #     from src.ml.training.callbacks import create_training_callbacks
            #     advanced_callbacks = create_training_callbacks(
            #         self.config,
            #         enable_warmup=self.config.warmup_enabled
            #     )
            #     callbacks.extend(advanced_callbacks)
            #     if self.verbose:
            #         print(f"[TRAINER]   OK Added {len(advanced_callbacks)} advanced callbacks:")
            #         for cb in advanced_callbacks:
            #             print(f"[TRAINER]      - {cb.__class__.__name__}")
            # except Exception as e:
            #     if self.verbose:
            #         print(f"[TRAINER]   [!] Could not load advanced callbacks: {e}")

            # Add GUI progress callback if provided
            if self.progress_callback:
                gui_callback = GUIProgressCallback(
                    progress_callback=self.progress_callback,
                    root=self.root
                )
                callbacks.append(gui_callback)
                if self.verbose:
                    print(f"[TRAINER]   OK GUI progress callback added to callbacks")

            # TensorBoard logger
            logger = None
            if enable_tensorboard:
                logger = TensorBoardLogger(
                    save_dir=str(self.log_dir),
                    name=run_name,
                    default_hp_metric=False
                )

                if self.verbose:
                    print(f"[TRAINER]   - TensorBoard log dir: {self.log_dir / run_name}")

            # Create trainer
            # Use modern PyTorch Lightning API (accelerator + devices instead of gpus)
            accelerator = "gpu" if gpus > 0 else "cpu"
            devices = gpus if gpus > 0 else "auto"

            # Log GPU information
            if accelerator == "gpu" and self.verbose:
                try:
                    print(f"[TRAINER]   - CUDA available: {torch.cuda.is_available()}")
                    if torch.cuda.is_available():
                        for i in range(min(gpus, torch.cuda.device_count())):
                            gpu_name = torch.cuda.get_device_name(i)
                            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                            print(f"[TRAINER]   - GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
                except Exception as e:
                    print(f"[TRAINER]   - Warning: Could not get GPU info: {e}")

            # Deterministic mode causes issues with some CUDA operations on GPU
            # Use deterministic=False for GPU, True for CPU for reproducibility where possible
            use_deterministic = (accelerator == "cpu")

            if accelerator == "gpu" and self.verbose:
                print(f"[TRAINER]   - Deterministic mode disabled for GPU (some CUDA ops lack deterministic implementations)")

            self.pl_trainer = LightningTrainer(
                max_epochs=self.config.max_epochs,
                accelerator=accelerator,
                devices=devices,
                gradient_clip_val=self.config.gradient_clip_val,
                callbacks=callbacks,
                logger=logger,
                enable_progress_bar=enable_progress_bar,
                enable_model_summary=True,
                log_every_n_steps=10,
                limit_train_batches=1.0,
                limit_val_batches=1.0,
                deterministic=use_deterministic
            )

            if self.verbose:
                print(f"[TRAINER] [OK] PyTorch Lightning trainer setup complete")
                print(f"[TRAINER]   - Accelerator: {accelerator}")
                print(f"[TRAINER]   - Devices: {devices}")
                print(f"[TRAINER]   - Deterministic: {use_deterministic}")
                print(f"[TRAINER]   - Max epochs: {self.config.max_epochs}")
                print(f"[TRAINER]   - Gradient clip: {self.config.gradient_clip_val}")
                print(f"[TRAINER]   - Early stopping patience: {self.config.early_stopping_patience}")

        except Exception as e:
            print(f"[TRAINER] [X] Trainer setup failed: {e}")
            raise

    def train(
        self,
        train_dataloader,
        val_dataloader,
        progress_callback: Optional[Callable] = None,
        validate_pipeline: bool = True
    ):
        """
        Train the model

        Args:
            train_dataloader: Training DataLoader
            val_dataloader: Validation DataLoader
            progress_callback: Optional callback function(epoch, train_loss, val_loss, metrics)
            validate_pipeline: Run validation checks before training
        """
        if self.model_wrapper is None:
            raise ValueError("Model not setup. Call setup_model() first.")

        if self.pl_trainer is None:
            raise ValueError("Trainer not setup. Call setup_trainer() first.")

        # Enable Tensor Core optimization for CUDA devices
        import torch
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('medium')
            if self.verbose:
                print(f"\n[TRAINER] Tensor Core optimization enabled (medium precision)")

        # VALIDATION: Run 5-step validation before training
        if validate_pipeline and self.verbose:
            print(f"\n[TRAINER] {'='*60}")
            print(f"[TRAINER] RUNNING PIPELINE VALIDATION")
            print(f"[TRAINER] {'='*60}")

            from ..training.dataset import validate_data_format, validate_target_normalization
            from ..training.metrics import validate_loss_function, validate_baseline_comparison

            validation_results = {}

            # Step 1: Data format
            validation_results['data_format'] = validate_data_format(train_dataloader, verbose=self.verbose)

            # CRITICAL FIX #3: Explicitly set model to eval mode before validation
            print(f"[TRAINER]   INFO  Setting model to eval() mode for validation checks...")
            self.model_wrapper.model.eval()

            # Step 2: Loss function
            validation_results['loss_function'] = validate_loss_function(
                self.model_wrapper.model, train_dataloader, verbose=self.verbose
            )

            # Step 4: Target normalization
            validation_results['target_normalization'] = validate_target_normalization(
                train_dataloader, verbose=self.verbose
            )

            # Step 5: Random baseline
            validation_results['random_baseline'] = validate_baseline_comparison(
                self.model_wrapper.model, val_dataloader, verbose=self.verbose
            )

            # Print validation summary
            all_passed = all(validation_results.values())
            print(f"\n[TRAINER] {'='*60}")
            print(f"[TRAINER] VALIDATION SUMMARY")
            print(f"[TRAINER] {'='*60}")
            for step, passed in validation_results.items():
                status = "OK PASS" if passed else "❌ FAIL"
                print(f"[TRAINER] {status} - {step}")

            if not all_passed:
                print(f"\n[TRAINER] [!]️  WARNING: Some validation checks failed!")
                print(f"[TRAINER] Training will continue, but results may be poor.")
            else:
                print(f"\n[TRAINER] OK All validation checks passed! Ready to train.")

            # CRITICAL FIX #3: Set model back to training mode after validation checks
            self.model_wrapper.model.train()
            print(f"[TRAINER]   OK Model set to train() mode for training")

        if self.verbose:
            print(f"\n[TRAINER] {'='*60}")
            print(f"[TRAINER] STARTING TRAINING")
            print(f"[TRAINER] {'='*60}")
            print(f"[TRAINER]   - Training batches: {len(train_dataloader)}")
            print(f"[TRAINER]   - Validation batches: {len(val_dataloader)}")
            print(f"[TRAINER]   - Max epochs: {self.config.max_epochs}")
            print(f"[TRAINER]   - Batch size: {self.config.batch_size}")
            print(f"[TRAINER]   - Learning rate: {self.config.learning_rate}")

            # Show GPU memory if using GPU
            if self.pl_trainer.accelerator == "gpu":
                try:
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            allocated = torch.cuda.memory_allocated(i) / 1024**3
                            reserved = torch.cuda.memory_reserved(i) / 1024**3
                            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                            print(f"[TRAINER]   - GPU {i} Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved / {total:.2f}GB total")
                except Exception as e:
                    print(f"[TRAINER]   - Warning: Could not get GPU memory info: {e}")

        self.training_start_time = time.time()

        try:
            # Get the model
            model = self.model_wrapper.model

            # CRITICAL FIX #6: PyTorch Lightning handles train/eval mode switching automatically
            # We just need to verify it's working by checking the model has normalization layers
            has_norm_layers = False
            for name, module in model.named_modules():
                if 'norm' in name.lower() or isinstance(module, (torch.nn.BatchNorm1d, torch.nn.LayerNorm)):
                    has_norm_layers = True
                    if self.verbose:
                        print(f"[TRAINER]   INFO Found normalization layer: {name}")
                    break

            if has_norm_layers and self.verbose:
                print(f"[TRAINER]   OK Model has normalization layers - Lightning will handle train/eval switching")
            elif self.verbose:
                print(f"[TRAINER]   INFO No normalization layers found")

            # Start training
            if self.verbose:
                print(f"\n[TRAINER] Starting PyTorch Lightning training loop...")
                print(f"[TRAINER]   - Model type: {type(model).__name__}")
                print(f"[TRAINER]   - Model module: {type(model).__module__}")

                # Check inheritance chain
                mro = [c.__name__ for c in type(model).__mro__]
                print(f"[TRAINER]   - Method Resolution Order (MRO): {mro[:10]}")

                # Try multiple isinstance checks
                try:
                    from pytorch_lightning import LightningModule as PL_LightningModule
                    print(f"[TRAINER]   - Is LightningModule (imported): {isinstance(model, PL_LightningModule)}")
                except:
                    pass

                # Check if it has required methods
                has_training_step = hasattr(model, 'training_step')
                has_configure_optimizers = hasattr(model, 'configure_optimizers')
                print(f"[TRAINER]   - Has training_step: {has_training_step}")
                print(f"[TRAINER]   - Has configure_optimizers: {has_configure_optimizers}")

            # Fit the model using PyTorch Lightning trainer
            # Use pytorch-forecasting's recommended approach to handle version compatibility
            try:
                # Try direct fit (works with compatible versions)
                self.pl_trainer.fit(
                    model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader
                )
            except TypeError as e:
                if "LightningModule" in str(e):
                    # Version mismatch - use pytorch-forecasting's Trainer wrapper
                    if self.verbose:
                        print(f"[TRAINER] [!] PyTorch Lightning version mismatch detected")
                        print(f"[TRAINER]   - Using pytorch-forecasting's training approach...")

                    # Import pytorch-forecasting's Trainer
                    from pytorch_forecasting import TemporalFusionTransformer

                    # Use the model's fit method which handles the trainer internally
                    # This approach works around version incompatibilities
                    trainer_params = {
                        'max_epochs': self.config.max_epochs,
                        'accelerator': self.pl_trainer.accelerator,
                        'devices': self.pl_trainer.devices if hasattr(self.pl_trainer, 'devices') else 'auto',
                        'gradient_clip_val': self.config.gradient_clip_val,
                        'callbacks': self.pl_trainer.callbacks,
                        'logger': self.pl_trainer.logger,
                        'enable_progress_bar': True,
                        'enable_model_summary': True,
                        'log_every_n_steps': 10,
                    }

                    # Create a new compatible trainer
                    from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer as TFT

                    # Re-create trainer with compatibility mode
                    compat_trainer = LightningTrainer(**trainer_params)

                    if self.verbose:
                        print(f"[TRAINER]   - Created compatibility trainer")
                        print(f"[TRAINER]   - Training using fit method...")

                    # Call fit with the compatibility trainer
                    compat_trainer.fit(
                        model,
                        train_dataloaders=train_dataloader,
                        val_dataloaders=val_dataloader
                    )
                else:
                    # Different error, re-raise
                    raise

            training_time = time.time() - self.training_start_time

            if self.verbose:
                print(f"\n[TRAINER] {'='*60}")
                print(f"[TRAINER] TRAINING COMPLETED")
                print(f"[TRAINER] {'='*60}")
                print(f"[TRAINER]   - Total time: {training_time/60:.2f} minutes")
                print(f"[TRAINER]   - Final epoch: {self.pl_trainer.current_epoch}")

            # Print final metrics summary
            self.metrics_tracker.print_summary()

        except KeyboardInterrupt:
            print(f"\n[TRAINER] [!] Training interrupted by user")
            print(f"[TRAINER]   - Completed epochs: {self.current_epoch}")

            # Save current state
            if self.verbose:
                print(f"[TRAINER]   - Saving interrupted training state...")

        except Exception as e:
            print(f"\n[TRAINER] [X] Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def train_epoch(self, epoch: int, train_dataloader) -> float:
        """
        Train for one epoch (for custom training loop)

        Args:
            epoch: Current epoch number
            train_dataloader: Training DataLoader

        Returns:
            Average training loss
        """
        if self.verbose:
            print(f"\n[TRAINER] Epoch {epoch}/{self.config.max_epochs}")
            print(f"[TRAINER]   - Training...")

        self.current_epoch = epoch
        epoch_start_time = time.time()

        # Training handled by PyTorch Lightning internally
        # This method is for compatibility with custom training loops

        epoch_time = time.time() - epoch_start_time

        if self.verbose:
            print(f"[TRAINER]   - Epoch time: {epoch_time:.2f}s")

        return 0.0  # Placeholder

    def validate_epoch(self, epoch: int, val_dataloader) -> float:
        """
        Validate for one epoch (for custom training loop)

        Args:
            epoch: Current epoch number
            val_dataloader: Validation DataLoader

        Returns:
            Average validation loss
        """
        if self.verbose:
            print(f"[TRAINER]   - Validating...")

        # Validation handled by PyTorch Lightning internally

        return 0.0  # Placeholder

    def save_best_model(self, path: str):
        """
        Save best model checkpoint

        Args:
            path: Output path for best model
        """
        if self.verbose:
            print(f"\n[TRAINER] Saving best model to {path}")

        try:
            if self.model_wrapper is None:
                raise ValueError("No model to save")

            # Get best checkpoint from Lightning trainer
            best_model_path = self.pl_trainer.checkpoint_callback.best_model_path

            if self.verbose:
                print(f"[TRAINER]   - Best checkpoint: {best_model_path}")

            # Copy to specified path
            import shutil
            shutil.copy(best_model_path, path)

            # Save config
            config_path = Path(path).parent / "model_config.json"
            self.config.save_json(str(config_path))

            if self.verbose:
                print(f"[TRAINER] [OK] Best model saved")
                print(f"[TRAINER]   - Model: {path}")
                print(f"[TRAINER]   - Config: {config_path}")

        except Exception as e:
            print(f"[TRAINER] [X] Failed to save best model: {e}")
            raise

    def export_metrics(self, output_dir: str):
        """
        Export training metrics

        Args:
            output_dir: Output directory for metrics
        """
        if self.verbose:
            print(f"\n[TRAINER] Exporting metrics to {output_dir}")

        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Export to CSV
            csv_path = output_dir / "training_metrics.csv"
            self.metrics_tracker.export_to_csv(str(csv_path))

            # Export to JSON
            json_path = output_dir / "training_metrics.json"
            self.metrics_tracker.export_to_json(str(json_path))

            if self.verbose:
                print(f"[TRAINER] [OK] Metrics exported")
                print(f"[TRAINER]   - CSV: {csv_path}")
                print(f"[TRAINER]   - JSON: {json_path}")

        except Exception as e:
            print(f"[TRAINER] [X] Failed to export metrics: {e}")
            raise

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get training summary

        Returns:
            Dictionary with training summary
        """
        if self.verbose:
            print(f"\n[TRAINER] Generating training summary...")

        summary = {
            'config': self.config.__dict__,
            'current_epoch': self.current_epoch,
            'max_epochs': self.config.max_epochs,
            'checkpoint_dir': str(self.checkpoint_dir),
            'log_dir': str(self.log_dir),
        }

        if self.training_start_time:
            training_time = time.time() - self.training_start_time
            summary['training_time_minutes'] = training_time / 60

        # Add best metrics
        if self.metrics_tracker.best_metrics:
            summary['best_metrics'] = {
                metric: {'epoch': epoch, 'value': value}
                for metric, (epoch, value) in self.metrics_tracker.best_metrics.items()
            }

        if self.verbose:
            print(f"[TRAINER] [OK] Training summary generated")
            for key, value in summary.items():
                if key != 'config':  # Skip config dict for brevity
                    print(f"[TRAINER]   - {key}: {value}")

        return summary


def train_model(
    train_dataloader,
    val_dataloader,
    config: Optional[TFTConfig] = None,
    checkpoint_dir: str = "models/checkpoints",
    gpus: int = 0,
    verbose: bool = True
):
    """
    Convenient function to train a model

    Args:
        train_dataloader: Training DataLoader with TimeSeriesDataSet
        val_dataloader: Validation DataLoader
        config: TFTConfig (uses default if None)
        checkpoint_dir: Directory for checkpoints
        gpus: Number of GPUs
        verbose: Print progress

    Returns:
        Trained CryptoTFT model wrapper
    """
    print(f"\n[TRAIN_MODEL] Starting model training pipeline...")

    # Use default config if not provided
    if config is None:
        from ..models.model_config import TFTConfig
        config = TFTConfig()
        print(f"[TRAIN_MODEL]   - Using default configuration")

    # Create trainer
    trainer = TFTTrainer(
        config=config,
        checkpoint_dir=checkpoint_dir,
        verbose=verbose
    )

    # Setup model from training dataset
    print(f"[TRAIN_MODEL] Setting up model from dataset...")
    trainer.setup_model(train_dataloader.dataset)

    # Setup PyTorch Lightning trainer
    print(f"[TRAIN_MODEL] Setting up trainer...")
    trainer.setup_trainer(gpus=gpus)

    # Train
    print(f"[TRAIN_MODEL] Starting training...")
    trainer.train(train_dataloader, val_dataloader)

    # Export metrics
    print(f"[TRAIN_MODEL] Exporting metrics...")
    trainer.export_metrics(checkpoint_dir)

    # Save best model
    best_model_path = Path(checkpoint_dir) / "best_model.ckpt"
    trainer.save_best_model(str(best_model_path))

    print(f"\n[TRAIN_MODEL] [OK] Training pipeline complete")
    print(f"[TRAIN_MODEL]   - Best model: {best_model_path}")

    return trainer.model_wrapper


if __name__ == "__main__":
    # Test trainer initialization
    print("Testing TFTTrainer")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("PyTorch not installed. Skipping test.")
    else:
        from ..models.model_config import ConfigPresets

        # Create config
        config = ConfigPresets.small()

        # Create trainer
        trainer = TFTTrainer(
            config=config,
            checkpoint_dir="test_checkpoints",
            verbose=True
        )

        # Get summary
        summary = trainer.get_training_summary()

        print(f"\n[TEST] [OK] Trainer initialization test passed!")
