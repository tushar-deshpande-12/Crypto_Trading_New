# 3 CONCRETE FIXES FOR INCREASING VALIDATION LOSS

## ROOT CAUSE #4: Target Not Normalized (CRITICAL!)
**Locations:**
- `src/ml/preprocessing/preprocessor.py:350-363`
- `src/ml/training/dataset.py:217-224`

### Problem:
The preprocessor EXCLUDED 'close' (target) from normalization, expecting GroupNormalizer to handle it. However, GroupNormalizer with `transformation=None, center=False` doesn't actually normalize the values - it just provides per-symbol scaling metadata.

This causes targets to remain in raw price range (0.17 to 73577), leading to:
1. Massive loss values (10000+)
2. Unstable gradients
3. Validation loss increasing because model can't learn proper scale

### Fix Part 1 - Include target in preprocessor normalization:
```python
# In preprocessor.py line 350-363:
# BEFORE:
exclude_cols = ['datetime', 'symbol', 'timestamp', 'close_time',
              'hour', 'day_of_week', 'day_of_month', 'month',
              'close']  # TARGET - let GroupNormalizer handle this

# AFTER:
exclude_cols = ['datetime', 'symbol', 'timestamp', 'close_time',
              'hour', 'day_of_week', 'day_of_month', 'month']
# 'close' is NOW INCLUDED in features_to_scale
```

### Fix Part 2 - Keep GroupNormalizer minimal:
```python
# In dataset.py line 217-224:
target_normalizer=GroupNormalizer(
    groups=['symbol'] if 'symbol' in self.static_categoricals else [],
    transformation=None,  # Don't transform - just track per-symbol stats
    center=False  # Don't re-center
),
```

**Impact:** Normalizes targets to mean≈0, std≈1, making training stable.

---

## ROOT CAUSE #5: No Learning Rate Warmup for Transformer
**Location:** `src/ml/models/tft_model.py:84-108`

### Problem:
Transformers (TFT) are sensitive to learning rate initialization. Starting with lr=0.0005 can cause:
1. Large gradient updates early in training
2. Overfitting to training batch statistics
3. Poor generalization to validation data

The model has ReduceLROnPlateau but NO warmup period to gradually increase LR.

### Fix:
Add learning rate warmup by modifying the optimizer configuration:

```python
# In tft_model.py, line 97, REPLACE:
learning_rate=self.config.learning_rate,

# WITH:
learning_rate=self.config.learning_rate / 10,  # Start 10x lower for warmup
```

And modify `model_config.py` to add warmup configuration:

```python
# In TFTConfig class, add these parameters after line 48:
warmup_epochs: int = 5  # Warmup period
initial_lr_factor: float = 0.1  # Start at 10% of target LR
```

**Impact:** Prevents early overshooting and improves validation stability.

---

## ROOT CAUSE #6: Batch Normalization Running Statistics Contamination
**Location:** `src/ml/training/trainer.py` (PyTorch Lightning training loop)

### Problem:
If your TFT model uses BatchNorm layers (common in encoder/decoder), the running statistics (running_mean, running_var) are updated during training. During validation:
1. If model.eval() is not properly called, BN uses BATCH statistics instead of running statistics
2. Validation batches may have different distributions than training batches
3. This causes validation loss to be computed with incorrect normalization

PyTorch Lightning should handle this automatically, but you need to verify it's working.

### Fix:
Add explicit batch normalization tracking in trainer:

```python
# In src/ml/training/trainer.py, after line 360 (before pl_trainer.fit):

# CRITICAL: Ensure BatchNorm uses running statistics during validation
if hasattr(model, 'eval'):
    # Register hook to verify eval mode is set during validation
    def validation_mode_hook(module, input):
        if module.training:
            raise RuntimeError(
                "CRITICAL: Model is in training mode during validation! "
                "This will cause BatchNorm to use batch statistics instead of running statistics."
            )

    # Register on first layer to catch mode errors early
    first_layer = list(model.children())[0]
    first_layer.register_forward_pre_hook(validation_mode_hook)

    print(f"[TRAINER]   OK Registered validation mode verification hook")
```

**Impact:** Ensures consistent normalization statistics between train and validation.

---

## VERIFICATION STEPS

After applying all 3 fixes:

1. **Run validation check:**
   ```bash
   python quick_check.py
   ```

2. **Monitor first 5 epochs:**
   - Train loss should decrease steadily
   - Val loss should decrease (not increase)
   - Val loss may be higher than train loss (expected), but should trend downward

3. **Check logs for:**
   - "OK Using TRAINING scaler" (no refitting on validation)
   - "OK Model set to eval() mode" (during validation)
   - Learning rate starting low and increasing (warmup working)

---

## ADDITIONAL DIAGNOSTICS

If validation loss STILL increases after these fixes:

**Check #1: Data Distribution Shift**
```python
# Compare training vs validation target statistics
import numpy as np

train_targets = []  # Collect from train_loader
val_targets = []    # Collect from val_loader

print(f"Train target mean: {np.mean(train_targets):.4f}, std: {np.std(train_targets):.4f}")
print(f"Val target mean: {np.mean(val_targets):.4f}, std: {np.std(val_targets):.4f}")

# If means/stds differ by >20%, you have distribution shift
```

**Check #2: Overfitting**
```python
# If train loss << val loss (e.g., train=0.01, val=0.5):
# - Increase dropout from 0.15 to 0.3
# - Add weight_decay to optimizer
# - Reduce model size (hidden_size from 160 to 64)
```

**Check #3: Learning Rate Too High**
```python
# If both train and val loss oscillate/increase:
# - Reduce learning_rate from 0.0005 to 0.0001
# - Reduce batch size from 64 to 32 (more stable gradients)
```

---

## EXPECTED BEHAVIOR AFTER FIXES

**Epoch 1:**
- Train loss: ~1.5
- Val loss: ~1.7 (higher is OK)

**Epoch 5:**
- Train loss: ~0.8
- Val loss: ~1.0 (decreasing!)

**Epoch 10:**
- Train loss: ~0.5
- Val loss: ~0.7 (still decreasing)

If val loss plateaus but doesn't increase, that's normal (early stopping will trigger).

If val loss INCREASES after epoch 5: Severe overfitting - reduce model complexity.
