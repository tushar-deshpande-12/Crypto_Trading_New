# Crypto AI Pipeline Reference Documentation

This document provides a quick reference for understanding and debugging the cryptocurrency AI trading pipeline.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA FLOW ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Binance API ───> Preprocessor ───> Dataset ───> Trainer ───> Predictor    │
│        │              │                │            │            │          │
│        │              │                │            │            │          │
│   dataset/         Features        DataLoaders   Model.ckpt   Predictions   │
│   {SYMBOL}/         + Target       + TimeIdx      + Metrics    + Backtest   │
│   data.json        target_return   train/val     val_loss     results       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Critical Constants (MUST MATCH)

| Constant | Location | Current Value | Description |
|----------|----------|---------------|-------------|
| `PREDICTION_HORIZON` | preprocessor.py | 24 | Hours ahead to predict |
| `max_prediction_length` | model_config.py | 24 | Model prediction horizon |
| `prediction_length` | dataset.py | 24 | Dataset sliding window |
| `max_encoder_length` | model_config.py | 168 | Context window (1 week) |
| `context_length` | dataset.py | 168 | Dataset context window |
| `target_column` | dataset.py | 'target_return' | What we're predicting |

**CRITICAL**: All horizons must match (24h). Mismatches cause training failures.

## Key Files & Their Purpose

### 1. Preprocessing (`src/ml/preprocessing/preprocessor.py`)
- **Purpose**: Load data, create features, normalize, split train/val/test
- **Input**: `dataset/{SYMBOL}/{TIMESTAMP}/data.json`
- **Output**:
  - Train/Val/Test DataFrames
  - Scalers saved to `models/checkpoints/scalers.pkl`
  - Target scaler for denormalization

**Key Methods**:
```python
preprocessor = CryptoPreprocessor("dataset")
train_df, val_df, test_df = preprocessor.process_all(["BTCUSDT"])
# target_scaler available at preprocessor.target_scaler
```

**What Can Go Wrong**:
- NaN/Inf values in features (check after dropna)
- Data leakage (val must come after train temporally)
- Target not normalized (mean should be ~0, std ~1)

### 2. Dataset (`src/ml/training/dataset.py`)
- **Purpose**: Create PyTorch TimeSeriesDataSet and DataLoaders
- **Input**: Preprocessed DataFrames
- **Output**: Train/Val/Test DataLoaders

**Key Methods**:
```python
train_loader, val_loader, test_loader = create_dataloaders(
    train_df, val_df, test_df,
    context_length=168,      # MUST match model config
    prediction_length=24,    # MUST match preprocessor
    target_column='target_return'  # MUST be normalized returns
)
```

**What Can Go Wrong**:
- Wrong target_column (should be 'target_return', NOT 'close')
- Context/prediction length mismatch
- Shuffling validation data (should NOT shuffle)

### 3. Model Training (`src/ml/training/trainer.py`)
- **Purpose**: Train TFT model with PyTorch Lightning
- **Input**: DataLoaders
- **Output**: Model checkpoint (.ckpt)

**Key Methods**:
```python
trainer = TFTTrainer(config=config)
trainer.setup_model(train_loader.dataset)
trainer.setup_trainer(gpus=1)
trainer.train(train_loader, val_loader)
```

**What Can Go Wrong**:
- Model in wrong mode (eval vs train)
- Loss not decreasing (check learning rate)
- Early stopping too aggressive
- Target not normalized

### 4. Prediction (`src/ml/inference/predictor.py`)
- **Purpose**: Make predictions with trained model
- **Input**: Model checkpoint, preprocessor with scalers
- **Output**: Price predictions

**Key Methods**:
```python
predictor = CryptoPredictor()
predictor.load_model("models/checkpoints/best_model.ckpt")
predictions = predictor.predict("BTCUSDT")
```

**What Can Go Wrong**:
- Missing target_scaler for denormalization
- Wrong model checkpoint format
- Preprocessor not loaded with saved scalers

### 5. Backtesting (`src/ml/backtest/backtester.py`)
- **Purpose**: Simulate trading with model predictions
- **Input**: Predicted returns, actual returns, prices
- **Output**: Trading performance metrics

**Key Methods**:
```python
backtester = CryptoBacktester()
backtester.set_target_scaler(preprocessor.target_scaler)  # IMPORTANT!
results = backtester.run_backtest_from_returns(
    predicted_returns, actual_returns, current_prices, timestamps
)
```

**What Can Go Wrong**:
- Not setting target_scaler (returns won't denormalize)
- Using wrong price data
- Timestamps with wrong index (use .iloc)

## Target Variable: `target_return`

The model predicts **normalized 24-hour returns**, NOT raw prices.

```
target_return = (close_t+24 - close_t) / close_t  # Percentage change
target_return_normalized = (target_return - mean) / std  # Scaled
```

**Why returns instead of prices?**
1. Prices are non-stationary (BTC went from $1k to $100k)
2. Returns are stationary (% changes stay similar)
3. Model can generalize across different price levels

## Validation Checks

### 1. Data Format Check
```
Input shape: [batch, context_length, features] = [64, 168, 85]
Target shape: [batch, prediction_length] = [64, 24]
```

### 2. Loss Function Check
- Model loss < Random loss (model better than random)
- Wrong predictions should have HIGHER loss

### 3. Target Normalization
```
Expected: mean ~= 0, std ~= 1
Actual: mean = 0.0000, std = 1.0000 (after normalization)
```

### 4. Baseline Comparison
- Model should beat random predictions
- Model should beat constant (always predict 0) predictions

## Common Issues & Solutions

### Issue: Validation loss not decreasing
**Cause**: Model not learning
**Solutions**:
1. Check target_column is 'target_return' not 'close'
2. Check prediction_length matches PREDICTION_HORIZON (24)
3. Increase learning rate (try 0.001 -> 0.003)
4. Add more training data
5. Check for data leakage

### Issue: Unicode encoding errors
**Cause**: Emoji characters in print statements
**Solution**: Replace emojis with ASCII:
- `❌` -> `[X]`
- `✅` -> `[OK]`
- `⚠️` -> `[!]`

### Issue: Model worse than random
**Cause**: Target leaking into features
**Solution**:
1. Check `target_return` not in time_varying_unknown_reals
2. Check no future data in features
3. Verify temporal splits (val starts AFTER train ends)

### Issue: Predictions all same value
**Cause**: Model collapsed to mean prediction
**Solution**:
1. Reduce learning rate
2. Add gradient clipping
3. Check feature scaling (all features should be ~mean 0, std 1)

### Issue: Denormalization gives wrong prices
**Cause**: Missing or wrong scaler
**Solution**:
1. Ensure preprocessor.target_scaler is saved
2. Load scalers before prediction
3. Use `run_backtest_from_returns` not `run_backtest`

## File Locations Quick Reference

| Component | Path |
|-----------|------|
| Raw Data | `dataset/{SYMBOL}/{TIMESTAMP}/data.json` |
| Preprocessed Scalers | `models/checkpoints/scalers.pkl` |
| Model Checkpoints | `models/checkpoints/tft_{TIMESTAMP}/` |
| Training Logs | `logs/training/tft_{TIMESTAMP}/` |
| TensorBoard Logs | `lightning_logs/` |
| Debug Logs | `debug_logs/debug_{SESSION}.json` |
| Test Scripts | `src/test_scripts/` |

## Running Tests

```bash
# Full pipeline test
python src/test_scripts/test_full_pipeline.py

# Just preprocessing test
python -c "from src.test_scripts.test_full_pipeline import test_preprocessing; test_preprocessing()"

# Just training test
python -c "from src.test_scripts.test_full_pipeline import test_model_training; test_model_training(...)"
```

## Model Checkpoint Formats

### .ckpt (PyTorch Lightning - TFT)
```python
from pytorch_forecasting import TemporalFusionTransformer
model = TemporalFusionTransformer.load_from_checkpoint(path)
```

### .pt (PyTorch - LSTM)
```python
model.load_state_dict(torch.load(path))
```

## Expected Training Behavior

1. **Epoch 1-3**: Loss drops rapidly (learning basic patterns)
2. **Epoch 4-10**: Loss decreases slowly (fine-tuning)
3. **Epoch 10+**: Loss stabilizes (near convergence)

If val_loss increases while train_loss decreases = OVERFITTING
If both stay flat = NOT LEARNING (check configuration)

## Debug Checklist

When something breaks, check in this order:

1. [ ] Target column is 'target_return' (not 'close' or 'direction')
2. [ ] Prediction horizon matches everywhere (24h)
3. [ ] Context length matches everywhere (168h)
4. [ ] No NaN/Inf in features
5. [ ] Target is normalized (mean ~0, std ~1)
6. [ ] Temporal order preserved (val comes after train)
7. [ ] Model is in correct mode (train for training, eval for validation)
8. [ ] Scalers are saved and loaded correctly
9. [ ] No Unicode characters in output (Windows issue)
