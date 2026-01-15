"""
Training History Logger - Saves training metrics to JSON for debugging.
Saves automatically on each epoch and on interruption (Ctrl+C).
"""
import json
import os
import signal
import atexit
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

try:
    from lightning.pytorch.callbacks import Callback
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    Callback = object


class TrainingHistoryLogger(Callback if LIGHTNING_AVAILABLE else object):
    """
    Logs training history to JSON file for debugging.

    Saves:
    - Loss values (train/val)
    - Learning rates
    - Metrics (accuracy, MAE, etc.)
    - Hyperparameters
    - Timestamps

    Features:
    - Saves after each epoch
    - Saves on Ctrl+C / interruption
    - Appends to existing history if resuming
    """

    def __init__(self, save_dir: str = "training_history",
                 filename: Optional[str] = None,
                 model_name: str = "model"):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{timestamp}.json"

        self.filepath = self.save_dir / filename
        self.model_name = model_name

        # Initialize history
        self.history: Dict[str, Any] = {
            "model_name": model_name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "status": "running",
            "hyperparameters": {},
            "epochs": [],
            "best_epoch": None,
            "best_val_loss": float('inf'),
            "total_epochs": 0,
            "interruption_reason": None
        }

        # Register signal handlers for graceful shutdown
        self._register_handlers()

        print(f"[HISTORY] Training history will be saved to: {self.filepath}")

    def _register_handlers(self):
        """Register handlers for Ctrl+C and exit."""
        # Save on exit
        atexit.register(self._save_on_exit)

        # Try to register SIGINT handler (Ctrl+C)
        try:
            self._original_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handle_interrupt)
        except (ValueError, OSError):
            # Can't set signal handler in this context (e.g., not main thread)
            pass

    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n[HISTORY] Ctrl+C detected - saving training history...")
        self.history["status"] = "interrupted"
        self.history["interruption_reason"] = "User interrupted (Ctrl+C)"
        self._save()

        # Call original handler
        if self._original_sigint and callable(self._original_sigint):
            self._original_sigint(signum, frame)
        else:
            raise KeyboardInterrupt

    def _save_on_exit(self):
        """Save history on program exit."""
        if self.history["status"] == "running":
            self.history["status"] = "exit"
        self._save()

    def _save(self):
        """Save history to JSON file."""
        self.history["end_time"] = datetime.now().isoformat()
        self.history["total_epochs"] = len(self.history["epochs"])

        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.history, f, indent=2, default=str)
            print(f"[HISTORY] Saved to {self.filepath}")
        except Exception as e:
            print(f"[HISTORY] Error saving: {e}")

    def setup(self, trainer, pl_module, stage=None):
        """Called when trainer is set up."""
        # Log hyperparameters
        if hasattr(pl_module, 'hparams'):
            self.history["hyperparameters"] = dict(pl_module.hparams)

        # Log trainer config
        self.history["trainer_config"] = {
            "max_epochs": trainer.max_epochs,
            "accelerator": str(trainer.accelerator),
            "gradient_clip_val": trainer.gradient_clip_val,
        }

        self._save()

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each training epoch."""
        epoch_data = {
            "epoch": trainer.current_epoch,
            "timestamp": datetime.now().isoformat(),
            "metrics": {}
        }

        # Collect all logged metrics
        for key, value in trainer.callback_metrics.items():
            try:
                epoch_data["metrics"][key] = float(value)
            except (TypeError, ValueError):
                epoch_data["metrics"][key] = str(value)

        # Get current learning rate
        if trainer.optimizers:
            opt = trainer.optimizers[0]
            epoch_data["learning_rate"] = opt.param_groups[0]['lr']

        # Track best validation loss
        val_loss = epoch_data["metrics"].get("val_loss")
        if val_loss is not None and val_loss < self.history["best_val_loss"]:
            self.history["best_val_loss"] = val_loss
            self.history["best_epoch"] = trainer.current_epoch

        self.history["epochs"].append(epoch_data)

        # Save after each epoch
        self._save()

    def on_train_end(self, trainer, pl_module):
        """Called when training ends."""
        self.history["status"] = "completed"
        self.history["final_metrics"] = {}

        for key, value in trainer.callback_metrics.items():
            try:
                self.history["final_metrics"][key] = float(value)
            except (TypeError, ValueError):
                self.history["final_metrics"][key] = str(value)

        self._save()
        print(f"[HISTORY] Training completed. History saved to: {self.filepath}")

    def on_exception(self, trainer, pl_module, exception):
        """Called when an exception occurs."""
        self.history["status"] = "error"
        self.history["interruption_reason"] = str(exception)
        self._save()


def load_training_history(filepath: str) -> Dict[str, Any]:
    """Load and analyze training history from JSON file."""
    with open(filepath, 'r') as f:
        history = json.load(f)
    return history


def print_history_summary(filepath: str):
    """Print a summary of training history."""
    history = load_training_history(filepath)

    print("\n" + "="*60)
    print(f"TRAINING HISTORY: {history['model_name']}")
    print("="*60)

    print(f"\nStatus: {history['status']}")
    print(f"Total epochs: {history['total_epochs']}")
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Best val_loss: {history['best_val_loss']:.6f}")

    if history.get('interruption_reason'):
        print(f"Interruption: {history['interruption_reason']}")

    print("\nHyperparameters:")
    for key, value in history.get('hyperparameters', {}).items():
        print(f"  {key}: {value}")

    if history['epochs']:
        print("\nLast 5 epochs:")
        for epoch in history['epochs'][-5:]:
            metrics = epoch.get('metrics', {})
            lr = epoch.get('learning_rate', 'N/A')
            train_loss = metrics.get('train_loss', 'N/A')
            val_loss = metrics.get('val_loss', 'N/A')
            print(f"  Epoch {epoch['epoch']}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={lr:.2e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print_history_summary(sys.argv[1])
    else:
        print("Usage: python history_logger.py <path_to_history.json>")
