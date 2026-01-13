# COMPREHENSIVE METRICS & EVALUATION - COMPLETE GUIDE

## What's Been Added

After training completes, the system now automatically:

1. ✅ **Evaluates model** on train/val/test sets
2. ✅ **Computes 6 key metrics** for each split
3. ✅ **Generates visualization graphs**
4. ✅ **Saves metrics as JSON and CSV**
5. ✅ **Calculates per-horizon accuracy** (1h, 2h, ..., 10h)
6. ✅ **Displays forecasting accuracy**

---

## Metrics Computed

### Overall Performance Metrics:

| Metric | Description | Good Value |
|--------|-------------|------------|
| **MAE** | Mean Absolute Error | Lower is better (< 0.5 for normalized) |
| **RMSE** | Root Mean Squared Error | Lower is better (< 1.0 for normalized) |
| **MSE** | Mean Squared Error | Lower is better |
| **MAPE** | Mean Absolute Percentage Error | < 10% is good, < 5% is excellent |
| **R²** | Coefficient of Determination | 0-1 scale, higher is better (>0.7 is good) |
| **Direction Accuracy** | % of correct up/down predictions | > 55% is better than random |

### Per-Horizon Metrics:

- `mae_horizon_1h` - Accuracy for 1 hour ahead
- `mae_horizon_2h` - Accuracy for 2 hours ahead
- ...
- `mae_horizon_10h` - Accuracy for 10 hours ahead

**Typically:** Short-term forecasts (1-3h) are more accurate than long-term (8-10h)

---

## Files Generated After Training

### Location: `models/checkpoints/evaluation/`

```
evaluation/
├── evaluation_metrics.json      # All metrics in JSON format
├── evaluation_metrics.csv       # Metrics in CSV for Excel
├── evaluation_metrics.png       # 4-panel comparison graph
└── horizon_accuracy.png         # Per-horizon accuracy plot
```

---

## Visualization Graphs

### 1. `evaluation_metrics.png` (4-Panel Graph)

**Top Left: MAE Comparison**
- Bar chart showing MAE for train/val/test
- Green (train), Orange (val), Red (test)
- Lower is better

**Top Right: RMSE Comparison**
- Bar chart showing RMSE for train/val/test
- Helps identify overfitting

**Bottom Left: R² Score**
- Shows how well model explains variance
- Scale 0-1, higher is better
- > 0.7 means model captures 70% of patterns

**Bottom Right: Direction Accuracy**
- % of correct trend predictions
- > 55% is better than random guessing
- Important for trading decisions

### 2. `horizon_accuracy.png`

**Line graph showing:**
- X-axis: Hours ahead (1 to 10)
- Y-axis: MAE at that horizon
- 3 lines: Train (green), Val (orange), Test (red)

**What to look for:**
- ✅ Flat line = consistent accuracy across all horizons
- ❌ Steep upward slope = accuracy degrades quickly
- ✅ Train/Val/Test lines close together = good generalization
- ❌ Large gap between lines = overfitting

---

## Example Output

### Console During Evaluation:

```
================================================================================
COMPREHENSIVE MODEL EVALUATION
================================================================================

[EVALUATOR] Evaluating on train set...
[EVALUATOR]   - Batches: 100
[EVALUATOR]   - Predictions shape: (6400, 10)
[EVALUATOR]   - Targets shape: (6400, 10)

[EVALUATOR] TRAIN Metrics:
[EVALUATOR]   - MAE:  0.3245
[EVALUATOR]   - RMSE: 0.4521
[EVALUATOR]   - MAPE: 8.23%
[EVALUATOR]   - R²:   0.7834
[EVALUATOR]   - Direction Accuracy: 62.15%

[EVALUATOR] Evaluating on val set...
[EVALUATOR]   - Batches: 1
[EVALUATOR]   - Predictions shape: (64, 10)
[EVALUATOR]   - Targets shape: (64, 10)

[EVALUATOR] VAL Metrics:
[EVALUATOR]   - MAE:  0.3892
[EVALUATOR]   - RMSE: 0.5234
[EVALUATOR]   - MAPE: 9.87%
[EVALUATOR]   - R²:   0.7245
[EVALUATOR]   - Direction Accuracy: 58.34%

[EVALUATOR] Evaluating on test set...

[EVALUATOR] TEST Metrics:
[EVALUATOR]   - MAE:  0.4123
[EVALUATOR]   - RMSE: 0.5678
[EVALUATOR]   - MAPE: 10.45%
[EVALUATOR]   - R²:   0.6987
[EVALUATOR]   - Direction Accuracy: 56.78%

[EVALUATOR] OK Metrics saved to models/checkpoints/evaluation/evaluation_metrics.json
[EVALUATOR] OK Metrics saved to models/checkpoints/evaluation/evaluation_metrics.csv
[EVALUATOR] OK Metrics plot saved to models/checkpoints/evaluation/evaluation_metrics.png
[EVALUATOR] OK Horizon plot saved to models/checkpoints/evaluation/horizon_accuracy.png

================================================================================
EVALUATION COMPLETE
================================================================================
Results saved to: models/checkpoints/evaluation
```

---

## How to Interpret Results

### Good Model Performance:

```json
{
  "train": {
    "mae": 0.32,
    "rmse": 0.45,
    "mape": 8.2,
    "r2": 0.78,
    "direction_accuracy": 62.1
  },
  "val": {
    "mae": 0.39,    // ✅ Close to train (not overfitting)
    "rmse": 0.52,
    "mape": 9.9,     // ✅ < 10% error
    "r2": 0.72,      // ✅ > 0.7 (good)
    "direction_accuracy": 58.3  // ✅ > 55% (better than random)
  },
  "test": {
    "mae": 0.41,    // ✅ Similar to val (generalizes well)
    "rmse": 0.57,
    "mape": 10.5,
    "r2": 0.70,     // ✅ Still good on unseen data
    "direction_accuracy": 56.8
  }
}
```

**Interpretation:** Model performs well, generalizes to new data, better than random for trading.

---

### Overfitting Warning:

```json
{
  "train": {
    "mae": 0.15,   // ❌ TOO GOOD on training
    "r2": 0.95     // ❌ Almost perfect on training
  },
  "val": {
    "mae": 0.65,   // ❌ MUCH WORSE on validation
    "r2": 0.45     // ❌ Poor generalization
  }
}
```

**Solution:** Increase dropout, reduce model size, add more regularization.

---

### Underfitting Warning:

```json
{
  "train": {
    "mae": 0.85,   // ❌ Poor even on training
    "r2": 0.35,    // ❌ Can't learn patterns
    "direction_accuracy": 52.1  // ❌ Barely better than random
  }
}
```

**Solution:** Increase model size, reduce dropout, train longer, add more features.

---

## Accessing Metrics Programmatically

### From Python:

```python
import json

# Load metrics
with open('models/checkpoints/evaluation/evaluation_metrics.json') as f:
    metrics = json.load(f)

# Access specific metrics
train_mae = metrics['train']['mae']
val_r2 = metrics['val']['r2']
test_direction_acc = metrics['test']['direction_accuracy']

print(f"Test Direction Accuracy: {test_direction_acc:.2f}%")
```

### From Pandas:

```python
import pandas as pd

# Load as DataFrame
df = pd.read_csv('models/checkpoints/evaluation/evaluation_metrics.csv')

# Compare splits
print(df[['split', 'mae', 'rmse', 'r2', 'direction_accuracy']])

# Visualize
df.plot(x='split', y=['mae', 'rmse'], kind='bar')
```

---

## Checkpoint Metrics

**Every checkpoint saved includes:**
- Model weights
- Optimizer state
- Epoch number
- Validation loss
- **Full evaluation metrics (new!)**

**Location:**
```
models/checkpoints/
├── best_model.ckpt              # Best model by val_loss
├── tft-epoch=XX-val_loss=Y.ckpt # Periodic checkpoints
└── evaluation/                  # Comprehensive evaluation
    ├── evaluation_metrics.json
    ├── evaluation_metrics.csv
    ├── evaluation_metrics.png
    └── horizon_accuracy.png
```

---

## Using Metrics for Model Selection

### Compare Models:

```python
# Model A
metrics_a = json.load(open('models/checkpoints/run_a/evaluation/evaluation_metrics.json'))

# Model B
metrics_b = json.load(open('models/checkpoints/run_b/evaluation/evaluation_metrics.json'))

# Compare test MAE
print(f"Model A Test MAE: {metrics_a['test']['mae']:.4f}")
print(f"Model B Test MAE: {metrics_b['test']['mae']:.4f}")

# Choose better model
best_model = 'A' if metrics_a['test']['mae'] < metrics_b['test']['mae'] else 'B'
print(f"Best model: {best_model}")
```

---

## Summary

**What you get after every training run:**

| Component | Location | Purpose |
|-----------|----------|---------|
| JSON Metrics | `evaluation_metrics.json` | All metrics, machine-readable |
| CSV Metrics | `evaluation_metrics.csv` | Excel-friendly format |
| Comparison Graph | `evaluation_metrics.png` | Visual comparison of train/val/test |
| Horizon Graph | `horizon_accuracy.png` | Forecast accuracy over time |
| Model Checkpoint | `best_model.ckpt` | Best model weights |

**Key Insights:**

1. ✅ **MAE/RMSE** - Overall prediction error
2. ✅ **R² Score** - How well model captures patterns
3. ✅ **Direction Accuracy** - Trading signal quality
4. ✅ **Per-Horizon MAE** - How far ahead model is reliable
5. ✅ **Train vs Val vs Test** - Overfitting detection

**Now you have complete visibility into model performance! 📊📈**
