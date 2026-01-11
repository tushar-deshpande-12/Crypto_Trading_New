"""
PyTorch Lightning callback for GUI progress updates
"""
from typing import Optional, Callable, Any
try:
    from lightning.pytorch.callbacks import Callback
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    Callback = object


class GUIProgressCallback(Callback):
    """
    PyTorch Lightning callback to update GUI during training

    Captures train/val losses and calls GUI update callback in thread-safe way
    """

    def __init__(
        self,
        progress_callback: Optional[Callable] = None,
        root: Optional[Any] = None
    ):
        """
        Initialize GUI progress callback

        Args:
            progress_callback: Function(epoch, max_epochs, train_loss, val_loss)
            root: Tkinter root for thread-safe GUI updates (use root.after())
        """
        super().__init__()
        self.progress_callback = progress_callback
        self.root = root
        self.train_loss = 0.0
        self.val_loss = 0.0

        print(f"[GUI_CALLBACK] Initialized")
        print(f"[GUI_CALLBACK]   - Callback provided: {progress_callback is not None}")
        print(f"[GUI_CALLBACK]   - Root provided: {root is not None}")

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at end of each training epoch"""
        # Get training loss from logged metrics
        if hasattr(trainer, 'callback_metrics') and trainer.callback_metrics:
            # Try different possible metric names
            for key in ['train_loss', 'train_loss_epoch', 'loss']:
                if key in trainer.callback_metrics:
                    self.train_loss = float(trainer.callback_metrics[key])
                    break

    def on_validation_end(self, trainer, pl_module):
        """Called after validation completes"""
        if not self.progress_callback:
            return

        # Get validation loss
        if hasattr(trainer, 'callback_metrics') and trainer.callback_metrics:
            # Try different possible metric names
            for key in ['val_loss', 'val_loss_epoch']:
                if key in trainer.callback_metrics:
                    self.val_loss = float(trainer.callback_metrics[key])
                    break

        # Get epoch info
        epoch = trainer.current_epoch + 1  # 1-indexed for display
        max_epochs = trainer.max_epochs

        # Call GUI update in thread-safe way
        try:
            if self.root:
                # Use Tkinter's after() for thread-safe GUI updates
                self.root.after(0, lambda: self._safe_callback(
                    epoch, max_epochs, self.train_loss, self.val_loss
                ))
            else:
                # Direct callback (not thread-safe, but works if called from main thread)
                self._safe_callback(epoch, max_epochs, self.train_loss, self.val_loss)

            print(f"[GUI_CALLBACK] Updated: Epoch {epoch}/{max_epochs}, "
                  f"Train={self.train_loss:.4f}, Val={self.val_loss:.4f}")

        except Exception as e:
            print(f"[GUI_CALLBACK] Error calling progress callback: {e}")

    def _safe_callback(self, epoch, max_epochs, train_loss, val_loss):
        """Safely call the progress callback with error handling"""
        try:
            self.progress_callback(epoch, max_epochs, train_loss, val_loss)
        except Exception as e:
            print(f"[GUI_CALLBACK] Error in progress callback: {e}")


if __name__ == "__main__":
    # Test callback
    print("Testing GUIProgressCallback")
    print("="*60)

    def test_callback(epoch, max_epochs, train_loss, val_loss):
        print(f"Callback received: Epoch {epoch}/{max_epochs}, "
              f"Train={train_loss:.4f}, Val={val_loss:.4f}")

    callback = GUIProgressCallback(progress_callback=test_callback)

    # Simulate trainer
    class MockTrainer:
        def __init__(self):
            self.current_epoch = 0
            self.max_epochs = 10
            self.callback_metrics = {'train_loss': 0.5, 'val_loss': 0.6}

    trainer = MockTrainer()

    print("\nTesting on_train_epoch_end...")
    callback.on_train_epoch_end(trainer, None)

    print("\nTesting on_validation_end...")
    callback.on_validation_end(trainer, None)

    print("\nOK Test complete!")
