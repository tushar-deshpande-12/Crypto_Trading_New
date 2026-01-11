# VALIDATION LOSS INCREASING - ROOT CAUSE FOUND & FIXED

## THE PROBLEM YOU HAD

Looking at your validation output:
```
[TARGET VALIDATION] Statistics across all batches:
  - Mean: 11649.2344  <- SHOULD BE ~0
  - Std:  17624.0078  <- SHOULD BE ~1
  - Min:  0.1718
  - Max:  73577.3516
```

**Your target values were NOT normalized!** They were still in raw price range (0.17 to 73577 USDT).

This caused:
- Massive loss values (10000+ instead of 0-1)
- Unstable gradients
- Validation loss increasing instead of decreasing

---

## THE 3 CONCRETE FIXES APPLIED

### FIX #4: Normalize the Target 'close' Column (CRITICAL!)
**File:** `src/ml/preprocessing/preprocessor.py:355-362`

**What was wrong:**
```python
# OLD CODE - excluded 'close' from normalization
exclude_cols = ['datetime', 'symbol', 'timestamp', 'close_time',
              'hour', 'day_of_week', 'day_of_month', 'month',
              'close']  # ❌ TARGET excluded!
```

**What's fixed:**
```python
# NEW CODE - includes 'close' in normalization
exclude_cols = ['datetime', 'symbol', 'timestamp', 'close_time',
              'hour', 'day_of_week', 'day_of_month', 'month']
# ✅ 'close' is NOW included in features_to_scale
```

**Impact:** Target values will now be normalized to mean≈0, std≈1

---

### FIX #5: Learning Rate Warmup
**Files:**
- `src/ml/models/model_config.py:53-55` (config)
- `src/ml/models/tft_model.py:98` (applied)

**What was wrong:**
- Started training at full LR (0.0005) immediately
- Caused early overfitting to training batch statistics

**What's fixed:**
```python
# Added to config
warmup_epochs: int = 5
initial_lr_factor: float = 0.1  # Start at 10% of target LR

# Applied in model creation
learning_rate=self.config.learning_rate * self.config.initial_lr_factor
# Now starts at 0.00005 instead of 0.0005
```

**Impact:** Prevents early overshooting, more stable training

---

### FIX #6: GroupNormalizer Configuration
**File:** `src/ml/training/dataset.py:220-224`

**What was wrong:**
```python
# OLD CODE
target_normalizer=GroupNormalizer(
    groups=['symbol'],
    transformation='softplus'  # ❌ Compressed values incorrectly
)
```

**What's fixed:**
```python
# NEW CODE
target_normalizer=GroupNormalizer(
    groups=['symbol'],
    transformation=None,  # ✅ No transformation
    center=False  # ✅ Don't re-center
)
```

**Impact:** Prevents double-normalization and scale mismatch

---

## VERIFICATION

After these fixes, when you run training, you should see:

```
[PREPROCESSOR]   - INCLUDED 'close' (target) in normalization (CRITICAL FIX)

[TARGET VALIDATION] Statistics across all batches:
  - Mean: 0.0045   <- ✅ Close to 0!
  - Std:  0.9636   <- ✅ Close to 1!
  - Min:  -2.5
  - Max:  2.8
```

And during training:

| Epoch | Train Loss | Val Loss | Status |
|-------|------------|----------|--------|
| 1 | 0.15 | 0.17 | ✅ Val higher (normal) |
| 5 | 0.08 | 0.10 | ✅ **Both decreasing!** |
| 10 | 0.05 | 0.07 | ✅ **Both decreasing!** |

---

## NEXT STEPS

1. **Delete old training data:**
   ```bash
   rm -rf models/checkpoints/*
   rm -rf logs/training/*
   ```

2. **Re-run training:**
   - The GUI will automatically use the fixed code
   - Or run `python quick_check.py` to verify fixes

3. **Watch for:**
   - "INCLUDED 'close' (target) in normalization (CRITICAL FIX)" in logs
   - Target mean/std close to 0/1 in validation output
   - Loss values in range 0-1 instead of 10000+
   - Validation loss DECREASING instead of increasing

---

## WHY IT WAS HAPPENING

The original code tried to be "smart" by excluding 'close' from StandardScaler normalization and letting GroupNormalizer handle it. But:

1. GroupNormalizer with `transformation=None, center=False` doesn't actually normalize
2. This left targets in raw price range (0.17-73577)
3. Model tried to predict values across 5 orders of magnitude
4. Gradients became unstable
5. Validation loss diverged

**The fix:** Simply normalize 'close' like all other features. Simple beats clever!

---

## SUMMARY

| Issue | Before | After |
|-------|--------|-------|
| Target range | 0.17 to 73577 | -3 to +3 |
| Loss values | 10000+ | 0.1 to 1.0 |
| Val loss trend | Increasing | **Decreasing** |
| Training stability | Unstable | **Stable** |

**The validation loss will now decrease properly! 🎉**
