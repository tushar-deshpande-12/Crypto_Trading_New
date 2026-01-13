# Quick Start Guide - Crypto AI Predictor v3.0.1

## Getting Started in 5 Minutes

### Step 1: Launch the Application

**Option A - Double Click:**
```
Double-click: run.bat
```

**Option B - Command Line:**
```bash
python main.py
```

The application will open with 3 main tabs:
- 📊 Market Overview
- 📥 Data Pipeline
- 🤖 AI Training

---

## Using the 10-Hour Prediction Feature

### Quick Steps:

**1. Go to AI Training Tab**
   - Click "🤖 AI Training" at the top

**2. Load the Trained Model**
   - In the "Prediction & Inference" section
   - Click "Browse" button
   - Navigate to: `models/checkpoints/`
   - Select: `best_model.ckpt`
   - Click Open

**3. Select a Cryptocurrency**
   - Use the "Predict For:" dropdown
   - Choose: BTCUSDT, ETHUSDT, or XRPUSDT

**4. Generate Prediction**
   - Click the big blue button: "🔮 Generate 10-Hour Prediction"
   - Wait 10-30 seconds while it:
     - Fetches latest market data from Binance
     - Processes the data
     - Generates predictions
     - Displays results

**5. View Results**
   - See the prediction summary above the chart
   - View the interactive price chart
   - Check the logs for detailed information

### Example Output:

```
Current Price: $94,532.15
10h Prediction: $95,123.00 (+0.62%)
Trend: Bullish 📈

95% Confidence Interval: [$93,800 - $96,500]
```

---

## Understanding the Results

### Prediction Chart
- **Blue Line**: Predicted price for next 10 hours
- **Green Dot**: Current price (starting point)
- **Orange Square**: 10-hour target price
- **Y-Axis**: Price in USD
- **X-Axis**: Hours ahead (0 to 10)

### Trend Indicators
- **📈 Bullish**: Price expected to increase (>0.5%)
- **📉 Bearish**: Price expected to decrease (<-0.5%)
- **➡️ Neutral**: Price expected to stay flat (-0.5% to +0.5%)

### Confidence Intervals
- **95% CI**: Very high confidence range (wider)
- **80% CI**: High confidence range (narrower)
- **Median**: Most likely outcome

---

## Data Freshness

**IMPORTANT**: The prediction system automatically fetches the **latest market data** from Binance.

✅ **What this means:**
- No need to manually update datasets
- Always uses current market conditions
- Predictions reflect real-time price movements
- Data is fetched each time you click predict

✅ **Data Retrieved:**
- Latest 500 hours (hourly candles)
- Current price from live market
- All OHLCV data + volume metrics

---

## Training Your Own Model (Optional)

If you want to retrain the model with fresh data:

### 1. Download Market Data
   - Go to "📥 Data Pipeline" tab
   - Select symbols from the list
   - Click download icon next to each symbol
   - Choose number of candles (50,000 recommended)
   - Click "Start Fetch"
   - Wait for download to complete

### 2. Configure Training
   - Go to "🤖 AI Training" tab
   - Select datasets to include
   - Adjust hyperparameters if needed
   - Review train/val/test split ratios

### 3. Start Training
   - Click "🚀 Start Training" button
   - Monitor progress in real-time:
     - Loss curves update each epoch
     - Training logs show detailed progress
     - GPU memory usage (if using GPU)
   - Training takes 30-90 minutes depending on hardware

### 4. Use New Model
   - When training completes, model is automatically saved
   - New model location: `models/checkpoints/best_model.ckpt`
   - Use immediately for predictions

---

## Troubleshooting

### "Model not found" error
**Solution**: Load the model first using the "Browse" button

### "Failed to fetch live data"
**Solution**: Check your internet connection - needs access to Binance API

### Predictions seem wrong
**Possible causes:**
- Extreme market volatility
- Model needs retraining with recent data
- Unusual market conditions

**Solution**: Retrain model with latest data

### GPU not detected
**Note**: GPU is optional. CPU training works fine, just slower.

**To enable GPU:**
1. Install CUDA toolkit
2. Install PyTorch with CUDA support
3. Restart application

---

## System Requirements

### Minimum:
- Windows 10/11
- 8GB RAM
- Internet connection
- Python 3.8+

### Recommended:
- 16GB+ RAM
- NVIDIA GPU with 4GB+ VRAM
- Fast internet connection
- Python 3.9 or 3.10

### Dependencies:
All listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Getting Help

### Log Files
Check logs for detailed information:
- `logs/crypto_ai.log` - Application log
- `logs/training/` - Training logs
- Console output - Real-time messages

### Documentation
- `README.md` - Full documentation
- `CHANGELOG.md` - Recent changes
- `docs/` - Additional guides

### Common Issues
Most issues are related to:
1. Model not loaded
2. Internet connection
3. Missing dependencies

**Quick Fix**: Restart the application and try again.

---

## Best Practices

### For Accurate Predictions:
1. **Use recent models**: Retrain monthly with new data
2. **Multiple symbols**: Train on 3+ cryptocurrencies
3. **Sufficient data**: Use 50,000+ candles per symbol
4. **Monitor trends**: Check if predictions align with market sentiment
5. **Use confidence intervals**: Consider the uncertainty ranges

### For Safe Trading:
⚠️ **WARNING**: Cryptocurrency trading is highly risky

- Predictions are **not financial advice**
- Always do your own research
- Use predictions as one of many inputs
- Never invest more than you can afford to lose
- Past performance doesn't guarantee future results

---

## Quick Reference

| Action | Location | Button/Control |
|--------|----------|---------------|
| Launch App | Root folder | `run.bat` or `main.py` |
| Load Model | AI Training > Prediction | "Browse" button |
| Generate Prediction | AI Training > Prediction | "🔮 Generate 10-Hour Prediction" |
| Download Data | Data Pipeline | Download icons in table |
| Train Model | AI Training > Model Config | "🚀 Start Training" |
| View Logs | AI Training > Training Progress | "Training Logs" section |

---

**Ready to start?**

1. Launch: `run.bat`
2. Click: "🤖 AI Training"
3. Load model: `models/checkpoints/best_model.ckpt`
4. Select symbol: BTCUSDT
5. Click: "🔮 Generate 10-Hour Prediction"
6. View results!

---

**Version**: 3.0.1 | **Date**: 2026-01-13 | **Status**: Production Ready
