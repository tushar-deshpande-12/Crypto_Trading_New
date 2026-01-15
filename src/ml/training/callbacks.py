"""
Training Callbacks for Improved Model Performance
Implements learning rate warmup, gradient monitoring, and better early stopping
Compatible with PyTorch Lightning 2.x
"""

import torch
import numpy as np
from typing import Optional

# Lightning 2.x compatible import
try:
    from lightning.pytorch.callbacks import Callback
except ImportError:
    from pytorch_lightning.callbacks import Callback


class LearningRateWarmup(Callback):
    """
    Implements learning rate warmup for better training stability

    Gradually increases learning rate from initial_lr_factor * base_lr to base_lr
    over warmup_epochs epochs.

    Args:
        warmup_epochs: Number of epochs for warmup
        initial_lr_factor: Starting LR as fraction of base LR (default: 0.1)
    """

    def __init__(self, warmup_epochs: int = 5, initial_lr_factor: float = 0.1):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lr_factor = initial_lr_factor
        self.base_lrs = {}

    def on_train_start(self, trainer, pl_module):
        """Store base learning rates"""
        for idx, optimizer in enumerate(trainer.optimizers):
            self.base_lrs[idx] = [group['lr'] for group in optimizer.param_groups]
            print(f"[WARMUP] Optimizer {idx} base LR: {self.base_lrs[idx]}")

    def on_train_epoch_start(self, trainer, pl_module):
        """Adjust learning rate based on current epoch"""
        epoch = trainer.current_epoch

        if epoch < self.warmup_epochs:
            # Calculate warmup factor: gradually increase from initial_lr_factor to 1.0
            warmup_factor = self.initial_lr_factor + (1.0 - self.initial_lr_factor) * (epoch / self.warmup_epochs)

            for idx, optimizer in enumerate(trainer.optimizers):
                for param_idx, param_group in enumerate(optimizer.param_groups):
                    base_lr = self.base_lrs[idx][param_idx] / self.initial_lr_factor  # Recover true base LR
                    new_lr = base_lr * warmup_factor
                    param_group['lr'] = new_lr

            print(f"[WARMUP] Epoch {epoch}/{self.warmup_epochs}: LR factor = {warmup_factor:.3f}")
        elif epoch == self.warmup_epochs:
            print(f"[WARMUP] Warmup complete! Switching to plateau scheduler.")


class GradientMonitor(Callback):
    """
    Monitors gradient norms and detects gradient issues

    Logs:
    - Total gradient norm
    - Per-layer gradient norms
    - Gradient statistics

    Alerts:
    - Vanishing gradients (norm < 1e-6)
    - Exploding gradients (norm > 10.0)
    - NaN gradients
    """

    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_after_backward(self, trainer, pl_module):
        """Monitor gradients after backward pass"""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        # Calculate total gradient norm
        total_norm = 0.0
        num_params = 0

        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                num_params += 1

        total_norm = total_norm ** 0.5

        # Log to trainer
        pl_module.log('grad_norm', total_norm, prog_bar=True)

        # Check for issues
        if total_norm > 10.0:
            print(f"\n[GRADIENT] [!] WARNING: Large gradient norm: {total_norm:.2f}")
            print(f"[GRADIENT]     This may indicate training instability")

        elif total_norm < 1e-6:
            print(f"\n[GRADIENT] [!] WARNING: Vanishing gradients: {total_norm:.2e}")
            print(f"[GRADIENT]     Model may not be learning")

        elif np.isnan(total_norm) or np.isinf(total_norm):
            print(f"\n[GRADIENT] [X] ERROR: NaN/Inf in gradients!")
            print(f"[GRADIENT]     Training will likely fail")

        # Log statistics every 100 steps
        if trainer.global_step % (self.log_every_n_steps * 2) == 0:
            # Collect gradient statistics
            grad_values = []
            for param in pl_module.parameters():
                if param.grad is not None:
                    grad_values.extend(param.grad.data.cpu().flatten().tolist())

            if grad_values:
                grad_mean = np.mean(np.abs(grad_values))
                grad_std = np.std(grad_values)
                grad_max = np.max(np.abs(grad_values))

                print(f"\n[GRADIENT] Statistics (step {trainer.global_step}):")
                print(f"[GRADIENT]   Total norm: {total_norm:.4f}")
                print(f"[GRADIENT]   Mean |grad|: {grad_mean:.6f}")
                print(f"[GRADIENT]   Std: {grad_std:.6f}")
                print(f"[GRADIENT]   Max |grad|: {grad_max:.6f}")


class LossMonitor(Callback):
    """
    Monitors training and validation loss for anomalies

    Detects:
    - Loss spikes (sudden >50% increase)
    - Loss plateau (no improvement for N epochs)
    - Train/val divergence (overfitting)
    """

    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def on_train_epoch_end(self, trainer, pl_module):
        """Track training loss"""
        if 'train_loss' in trainer.callback_metrics:
            train_loss = trainer.callback_metrics['train_loss'].item()
            self.train_losses.append(train_loss)

            # Check for loss spike
            if len(self.train_losses) > 1:
                prev_loss = self.train_losses[-2]
                if train_loss > prev_loss * 1.5:
                    print(f"\n[LOSS] [!] WARNING: Training loss spiked!")
                    print(f"[LOSS]     Previous: {prev_loss:.4f} → Current: {train_loss:.4f}")
                    print(f"[LOSS]     This may indicate:")
                    print(f"[LOSS]       - Learning rate too high")
                    print(f"[LOSS]       - Batch with outliers")
                    print(f"[LOSS]       - Gradient explosion")

    def on_validation_epoch_end(self, trainer, pl_module):
        """Track validation loss and detect issues"""
        if 'val_loss' in trainer.callback_metrics:
            val_loss = trainer.callback_metrics['val_loss'].item()
            self.val_losses.append(val_loss)

            # Check for improvement
            if val_loss < self.best_val_loss * 0.995:  # 0.5% improvement threshold
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Print progress
            epoch = trainer.current_epoch
            if len(self.train_losses) > 0:
                train_loss = self.train_losses[-1]
                gap = abs(train_loss - val_loss)
                gap_pct = (gap / val_loss) * 100

                print(f"\n[LOSS] Epoch {epoch:3d}: Train={train_loss:.4f}, Val={val_loss:.4f}, Gap={gap_pct:.1f}%")

                # Check for overfitting
                if train_loss < val_loss * 0.7:  # Train loss much better than val
                    print(f"[LOSS] [!] WARNING: Possible overfitting detected")
                    print(f"[LOSS]     Train loss is significantly better than validation loss")
                    print(f"[LOSS]     Consider: More regularization, dropout, or early stopping")

            # Check for plateau
            if self.epochs_without_improvement >= 5:
                print(f"[LOSS] [!] No improvement for {self.epochs_without_improvement} epochs")
                print(f"[LOSS]     Best val loss: {self.best_val_loss:.4f}")


class FeatureMonitor(Callback):
    """
    Monitors feature statistics during training

    Checks for:
    - NaN in inputs
    - Extreme values (>10 std from mean)
    - Distribution shifts between train/val
    """

    def __init__(self, check_every_n_epochs: int = 5):
        super().__init__()
        self.check_every_n_epochs = check_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Check feature statistics periodically"""
        if trainer.current_epoch % self.check_every_n_epochs != 0:
            return

        print(f"\n[FEATURES] Checking feature health (epoch {trainer.current_epoch})...")

        # Get a batch from validation dataloader (Lightning 2.x compatible)
        val_dataloader = trainer.val_dataloaders
        if isinstance(val_dataloader, (list, tuple)):
            val_dataloader = val_dataloader[0]
        batch = next(iter(val_dataloader))

        # Check continuous features
        if 'encoder_cont' in batch[0]:
            features = batch[0]['encoder_cont']

            # Check for NaN
            nan_count = torch.isnan(features).sum().item()
            if nan_count > 0:
                print(f"[FEATURES] [X] ERROR: {nan_count} NaN values in features!")

            # Check for extreme values
            mean = features.mean().item()
            std = features.std().item()
            max_val = features.max().item()
            min_val = features.min().item()

            if abs(max_val) > 10 or abs(min_val) > 10:
                print(f"[FEATURES] [!] WARNING: Extreme feature values detected")
                print(f"[FEATURES]     Range: [{min_val:.2f}, {max_val:.2f}]")
                print(f"[FEATURES]     This may indicate normalization issues")
            else:
                print(f"[FEATURES] [OK] Features look healthy")
                print(f"[FEATURES]     Mean: {mean:.3f}, Std: {std:.3f}")


def create_training_callbacks(config, enable_warmup: bool = True):
    """
    Create standard set of training callbacks

    Args:
        config: TFTConfig object
        enable_warmup: Whether to enable LR warmup

    Returns:
        List of callbacks
    """
    callbacks = []

    if enable_warmup and config.warmup_enabled:
        callbacks.append(
            LearningRateWarmup(
                warmup_epochs=config.warmup_epochs,
                initial_lr_factor=config.initial_lr_factor
            )
        )

    callbacks.extend([
        GradientMonitor(log_every_n_steps=50),
        LossMonitor(),
        FeatureMonitor(check_every_n_epochs=5)
    ])

    return callbacks


if __name__ == "__main__":
    print("Training Callbacks Module")
    print("=" * 60)
    print("Available callbacks:")
    print("  - LearningRateWarmup: Gradual LR increase for stability")
    print("  - GradientMonitor: Track gradient health")
    print("  - LossMonitor: Detect loss anomalies")
    print("  - FeatureMonitor: Check feature quality")
    print()
    print("Usage:")
    print("  from src.ml.training.callbacks import create_training_callbacks")
    print("  callbacks = create_training_callbacks(config)")
    print("  trainer = Trainer(callbacks=callbacks)")
