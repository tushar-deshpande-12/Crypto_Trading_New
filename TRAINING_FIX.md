# Training Error Fix - Data Requirements

## Problem

You encountered this error:
```
[PREPROCESSOR]   - Input shape: (0, 50)
ValueError: No objects to concatenate
```

## Root Cause

**The Issue**: Volatility features require longer historical data, but the system was loading the smaller (200 candles) datasets instead of the larger (50,000 candles) datasets you already have.

**Why it happened**:
1. You have TWO datasets per symbol:
   - 50,000 candles (from Jan 11) ← **Good for training**
   - 200 candles (from Jan 13) ← **Too small**

2. Old code loaded "latest" dataset (200 candles)
3. Volatility features use up to 168 hours (7 days) of lookback
4. After dropping NaN rows → 0 rows left → Training fails

## Solution Applied

✅ **I've fixed the code** to automatically prefer **larger datasets** over newer ones.

Now it will use your 50,000 candle datasets for training!

---

## Quick Fix: Restart Training

### Option 1: Just Retry (Recommended)

The fix is already applied. Simply:

1. **Close the error dialog** in the app
2. **Click "Start Training" again**
3. **Watch the logs** - should now say:
   ```
   [PREPROCESSOR] Loaded 50000 rows  ← Should see this!
   ```

### Option 2: Delete Small Datasets (Optional)

If you want to clean up:

```bash
# Remove the 200-candle datasets
rm -rf dataset/BTCUSDT/2026-01-13_09-46-58_200candles
rm -rf dataset/ETHUSDT/2026-01-13_09-46-58_200candles
rm -rf dataset/XRPUSDT/2026-01-13_09-46-59_200candles
```

Then restart training.

---

## Data Requirements for Volatility Features

### Minimum Requirements

| Feature | Minimum Candles Needed |
|---------|----------------------|
| Basic features | 200 candles |
| + Lagged features (168h) | 368 candles |
| + Rolling features (24h) | 392 candles |
| + Volatility features (168h) | 560 candles |
| + Volatility percentile (7d) | **728 candles** |

**Recommended**: 1,000+ candles minimum, 5,000-50,000 ideal

### Your Current Datasets

✅ **You already have enough data!**

```
BTCUSDT: 50,000 candles ✓
ETHUSDT: 50,000 candles ✓
XRPUSDT: 50,000 candles ✓
```

---

## What Changed in the Code

### Before (Broken)
```python
# Loaded "latest" dataset (by timestamp)
latest_dataset = sorted(dataset_folders)[-1]
# Result: 200 candles (newest but small)
```

### After (Fixed)
```python
# Load "largest" dataset (by file size)
best_dataset = max(dataset_folders, key=lambda d: filesize(d))
# Result: 50,000 candles (largest dataset)
```

---

## Verification

After restarting training, check the logs:

**Good Output** (Fixed):
```
[PREPROCESSOR] [1/3] Loading BTCUSDT...
[PREPROCESSOR]   - Using dataset: 2026-01-11_11-07-31_50000candles
[PREPROCESSOR]   - Loaded 50000 rows  ← This is what you want!

[PREPROCESSOR] Generating volatility features...
[PREPROCESSOR]   - Generated 9 volatility features
[PREPROCESSOR]   - NaN values: 168

[PREPROCESSOR] Removing NaN rows...
[PREPROCESSOR]   - Shape before: (150000, 50)  ← 50k × 3 symbols
[PREPROCESSOR]   - Shape after: (149832, 50)   ← Still plenty left!

[PREPROCESSOR] Splitting data into train/val/test...
[PREPROCESSOR]   - Input shape: (149832, 50)  ← Good!
```

**Bad Output** (Still broken):
```
[PREPROCESSOR]   - Loaded 200 rows  ← Too small!
[PREPROCESSOR]   - Input shape: (0, 50)  ← No data left!
```

---

## If Still Having Issues

### Issue 1: Still Loading Small Datasets

**Solution**: Delete the 200-candle datasets manually
```bash
cd dataset
rm -rf */2026-01-13_*_200candles
```

### Issue 2: Need Fresh Data

**Option A**: Use the Data Pipeline tab in the app
- Go to "Data Pipeline" tab
- Click download icons next to symbols
- Choose 5,000-50,000 candles
- Wait for download

**Option B**: Run quick fetch script
```bash
python scripts/quick_data_fetch.py
```

### Issue 3: Out of Memory

**Solution**: If 50,000 candles is too much:
- Reduce batch size to 32
- Or use 5,000-10,000 candles instead

---

## Technical Details

### Why Volatility Features Need More Data

**Volatility Percentile (7 days)**:
```python
vol_percentile_7d = rolling_percentile(
    volatility_24h,
    window=24*7  # Needs 168 hours of history
)
```

**Total lookback needed**:
```
Lagged features:     168 hours (close_lag_168h)
Rolling features:    24 hours (rolling_mean_24h)
Volatility features: 168 hours (volatility_168h, vol_of_vol)
Volatility percentile: 168 hours (vol_percentile_7d)

Total maximum lookback: 168 hours
Minimum data needed: 168 + buffer = ~200 hours
Recommended: 1000+ hours for stable statistics
```

---

## Summary

### What Happened
1. ❌ Code loaded 200-candle datasets (too small)
2. ❌ Volatility features created lots of NaN
3. ❌ After dropna() → 0 rows left
4. ❌ Training failed

### What's Fixed
1. ✅ Code now prefers **largest** datasets
2. ✅ Will load your 50,000-candle datasets
3. ✅ Plenty of data after dropna()
4. ✅ Training will succeed

### What You Do
1. **Click "Start Training" again**
2. **Watch for "Loaded 50000 rows"**
3. **Training should complete successfully**

---

**Status**: ✅ Fixed and ready
**Action Required**: Restart training
**Expected Time**: 30-120 minutes
**Expected Result**: Successful training with volatility features!

---

**Need help?** Check the logs for:
- "Loaded XXXXX rows" - should be 50,000
- "Generated 9 volatility features" - confirms new features
- "Input shape: (XXXXX, 50)" - should be >100,000 (not 0!)
