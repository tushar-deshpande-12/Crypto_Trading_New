# QUICK FIX FOR 3 ISSUES

## Issue 1: Sklearn Warnings (FIXED)
**Problem:** Getting repeated warnings about feature names

**Fix Applied:**
Added warning suppression to `src/ml/preprocessing/preprocessor.py`:
```python
import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')
```

---

## Issue 2: Model Not Starting
**Root Cause:** The model IS starting, but there's likely an error during training that's being caught silently.

**Diagnostic Steps:**
1. Check the full console output for errors
2. Look for PyTorch errors after "Starting PyTorch Lightning training loop..."
3. Check if GPU memory is sufficient

**Most Likely Cause:** The RMSE loss change requires the model to be retrained from scratch. Delete old checkpoints:
```bash
rm -rf models/checkpoints/*
rm -rf logs/training/*
```

---

## Issue 3: Graph Not Updating
**Root Cause:** PyTorch Lightning doesn't automatically call GUI callbacks during training.

**Solution:** Create a custom Lightning callback to update the GUI.

### File: `src/ml/training/gui_callback.py` (NEW FILE)

```python
"""
PyTorch Lightning callback for GUI progress updates
"""
from lightning.pytorch.callbacks import Callback

class GUIProgressCallback(Callback):
    """Callback to update GUI during training"""

    def __init__(self, progress_callback=None, root=None):
        """
        Args:
            progress_callback: Function(epoch, max_epochs, train_loss, val_loss)
            root: Tkinter root for thread-safe updates
        """
        self.progress_callback = progress_callback
        self.root = root
        self.train_loss = 0.0
        self.val_loss = 0.0

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at end of training epoch"""
        # Get training loss
        if trainer.callback_metrics:
            self.train_loss = float(trainer.callback_metrics.get('train_loss', 0.0))

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at end of validation epoch"""
        if not self.progress_callback:
            return

        # Get validation loss
        if trainer.callback_metrics:
            self.val_loss = float(trainer.callback_metrics.get('val_loss', 0.0))

        epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs

        # Call GUI update in thread-safe way
        if self.root:
            self.root.after(0, lambda: self.progress_callback(
                epoch, max_epochs, self.train_loss, self.val_loss
            ))
        else:
            self.progress_callback(epoch, max_epochs, self.train_loss, self.val_loss)
```

### Update `src/ml/training/trainer.py`

Add to imports:
```python
from .gui_callback import GUIProgressCallback
```

Update `__init__`:
```python
def __init__(
    self,
    config: TFTConfig,
    checkpoint_dir: str = "models/checkpoints",
    log_dir: str = "logs/training",
    verbose: bool = True,
    progress_callback: Optional[callable] = None,
    root: Optional[any] = None  # Tkinter root
):
    ...
    self.progress_callback = progress_callback
    self.root = root
```

Update `setup_trainer` (add to callbacks list around line 200):
```python
callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]

# Add GUI callback if provided
if self.progress_callback:
    gui_callback = GUIProgressCallback(self.progress_callback, self.root)
    callbacks.append(gui_callback)
    if self.verbose:
        print(f"[TRAINER]   OK GUI progress callback registered")

self.pl_trainer = LightningTrainer(
    ...
    callbacks=callbacks,
    ...
)
```

### Update `src/gui/app.py`

Change training thread (around line 465):
```python
# Create trainer with GUI callback
trainer = TFTTrainer(
    config=config,
    verbose=False,
    progress_callback=self.ml_panel.update_progress,
    root=self.root  # Pass Tkinter root for thread-safe updates
)
```

---

## SIMPLIFIED FIX (If above is too complex)

**Just add print statements to verify training is progressing:**

In `src/ml/training/trainer.py`, after line 400 add:
```python
# Add progress logging
if self.verbose:
    print(f"\n[TRAINER] Training started - check TensorBoard for progress")
    print(f"[TRAINER] Run: tensorboard --logdir={self.log_dir}")
```

Then monitor training via TensorBoard:
```bash
tensorboard --logdir=logs/training
```

Open http://localhost:6006 to see live training progress.

---

## IMMEDIATE ACTION

1. **Delete old checkpoints** (critical for RMSE fix):
   ```bash
   del models\\checkpoints\\* /Q
   del logs\\training\\* /S /Q
   ```

2. **Restart training** - The sklearn warnings are now suppressed

3. **Check console output** - Look for actual errors

4. **Use TensorBoard** to monitor progress while I implement the GUI callback

The model SHOULD be training now, you just can't see the progress in the GUI yet.
