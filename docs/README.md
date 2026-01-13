# Crypto AI Predictor v4.0

Professional cryptocurrency analysis and AI-powered prediction platform with Temporal Fusion Transformer (TFT) for multi-asset time series forecasting.

## 🚀 Features

### 📊 Market Monitoring
- **Real-time Market Data**: Live data from Binance public API (2000+ USDT pairs)
- **Comprehensive Metrics**: Price, 24h change %, volume, trades, high/low
- **Smart Filtering**: Instant search by symbol or base asset
- **Column Sorting**: Click any column header to sort
- **Color-Coded**: Green for gains, red for losses
- **Professional UI**: Dark theme optimized for trading

### 📈 Interactive Charts
- **Candlestick Visualization**: Professional OHLCV charts
- **Multiple Timeframes**: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w
- **Volume Analysis**: Color-coded volume bars
- **Chart Tools**: Zoom, pan, save images
- **Flexible History**: 50 to 500 candles

### 📥 Data Pipeline
- **Historical Data Fetching**: Download up to 50,000 hourly candles (~5.7 years)
- **Multi-Symbol Support**: Fetch data for multiple cryptocurrencies
- **Dual Format Storage**: CSV for analysis, JSON for APIs
- **Progress Tracking**: Real-time progress updates
- **Metadata Management**: Automatic dataset versioning

### 🤖 AI Model Training (NEW!)
- **Temporal Fusion Transformer**: State-of-the-art time series prediction
- **Multi-Asset Training**: Single model trained on multiple cryptocurrencies
- **40,000-Hour Context Window**: ~4.5 years of historical data for predictions
- **10-Hour Forecasts**: Predict next 10 hours with confidence intervals
- **Live Training Visualization**: Real-time loss curves and metrics
- **Hyperparameter Configuration**: Customize model architecture
- **TensorBoard Integration**: Advanced training visualization
- **Model Management**: Save, load, and version trained models

### 🔮 AI Predictions (NEW!)
- **10-Hour Price Forecasts**: Probabilistic predictions with confidence intervals
- **Quantile Predictions**: 80% and 95% confidence bounds
- **Trend Analysis**: Bullish/Bearish/Neutral trend indicators
- **Visual Charts**: Actual vs Predicted with shaded confidence regions
- **Multi-Symbol Support**: Generate predictions for any trained cryptocurrency
- **Export Functionality**: Save predictions to CSV

## 📦 Installation

### Basic Installation (Market Monitoring & Charts)

```bash
# Install core dependencies
pip install -r requirements.txt
```

Core dependencies:
- `requests` - Binance API
- `matplotlib` - Charts
- `pandas` - Data manipulation
- `mplfinance` - Candlestick charts

### Full Installation (Including AI Features)

```bash
# Install all dependencies including ML libraries
pip install -r requirements.txt
```

Additional ML dependencies (optional, for AI features):
- `torch` - Deep learning framework (~1.5GB)
- `pytorch-lightning` - Training orchestration
- `pytorch-forecasting` - TFT implementation
- `tensorboard` - Training visualization
- `scikit-learn` - Preprocessing
- `numpy` - Numerical operations
- `tqdm` - Progress bars
- `pyyaml` - Configuration files

**Note**: AI features require PyTorch. The application works without ML dependencies for market monitoring and data fetching.

## 🎯 Quick Start

### Windows

Double-click any launcher:
```
run.bat                      # Main application (v4.0)
run_crypto_tracker.bat       # Legacy launcher
run_data_fetch.bat           # Data pipeline only
```

### Manual Start

```bash
python main.py
```

## 📖 Usage Guide

### 1. Market Monitoring

1. Launch the application
2. Browse 2000+ USDT trading pairs
3. Filter by symbol (e.g., type "BTC")
4. Sort by any column (price, volume, change%)
5. Click [Chart] to view candlestick charts

### 2. Downloading Historical Data

1. Click **📥 Data Pipeline** in the top menu
2. Select a cryptocurrency from the symbol list
3. Choose candle count (1,000 to 50,000)
4. Click **Start Fetching Data**
5. Data saved to `dataset/SYMBOL/YYYY-MM-DD_HH-MM-SS_Ncandles/`

### 3. Training AI Model

1. Click **🤖 AI Training** in the top menu

**Step 1: Select Datasets**
- Check cryptocurrencies to train on (ETHUSDT, XRPUSDT, etc.)
- Adjust train/val/test split (default: 70/15/15)

**Step 2: Configure Model**
- Set hyperparameters:
  - Hidden Size: 128 (model capacity)
  - LSTM Layers: 2 (depth)
  - Attention Heads: 4 (attention mechanism)
  - Batch Size: 32 (training batch)
  - Max Epochs: 100 (training iterations)
  - Learning Rate: 0.001

**Step 3: Start Training**
- Click **🚀 Start Training**
- Monitor live loss curves
- View metrics: MAE, RMSE, MAPE, R²
- Training logs displayed in real-time
- Model saved to `models/checkpoints/`

**Step 4: Save Best Model**
- Best model automatically saved based on validation loss
- Located in `models/checkpoints/tft_TIMESTAMP/`

### 4. Generating Predictions

1. In **🤖 AI Training** view, scroll to Prediction section
2. Click **Browse** to load trained model (.ckpt file)
3. Select cryptocurrency from dropdown
4. Click **🔮 Generate 10-Hour Prediction**
5. View prediction chart with confidence intervals
6. Export to CSV if needed

## 🏗️ Project Structure

```
C:\crypto\ver4\
├── main.py                         # Entry point
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── run.bat                         # Windows launcher
│
├── src/                            # Source code
│   ├── core/                       # Configuration
│   │   └── config.py               # App settings
│   ├── api/                        # Binance API
│   │   └── binance_client.py      # API client
│   ├── data/                       # Data management
│   │   ├── fetcher.py              # Data fetching
│   │   ├── storage.py              # File storage
│   │   └── manager.py              # Data coordination
│   ├── gui/                        # User interface
│   │   ├── app.py                  # Main application
│   │   ├── styles.py               # UI theming
│   │   └── components/             # UI components
│   │       ├── symbol_table.py     # Symbol list
│   │       ├── chart_panel.py      # Chart viewer
│   │       ├── data_panel.py       # Data fetcher
│   │       └── ml_panel.py         # AI training/prediction
│   ├── ml/                         # Machine learning (NEW!)
│   │   ├── models/                 # Model architectures
│   │   │   ├── model_config.py     # TFT configuration
│   │   │   └── tft_model.py        # TFT wrapper
│   │   ├── training/               # Training pipeline
│   │   │   ├── trainer.py          # Training orchestration
│   │   │   ├── dataset.py          # PyTorch Dataset
│   │   │   └── metrics.py          # Evaluation metrics
│   │   ├── preprocessing/          # Data preprocessing
│   │   │   ├── preprocessor.py     # Feature engineering
│   │   │   └── windowing.py        # Sliding windows
│   │   └── inference/              # Prediction
│   │       └── predictor.py        # Inference pipeline
│   └── utils/                      # Utilities
│       └── formatters.py           # Formatting helpers
│
├── dataset/                        # Downloaded data
│   ├── ETHUSDT/                    # Ethereum datasets
│   ├── XRPUSDT/                    # Ripple datasets
│   └── ...
│
├── models/                         # Trained models
│   ├── checkpoints/                # Training checkpoints
│   ├── production/                 # Production models
│   └── experiments/                # Experimental models
│
├── docs/                           # Documentation
├── logs/                           # Application logs
└── legacy/                         # Legacy code
```

## 🧠 AI Model Details

### Temporal Fusion Transformer (TFT)

**What is TFT?**
- State-of-the-art transformer architecture for time series
- Multi-horizon forecasting with interpretable attention
- Handles multiple time series simultaneously
- Variable importance scoring
- Probabilistic predictions with quantiles

**Model Architecture**:
- **Encoder**: 40,000-hour context window (~4.5 years)
- **Decoder**: 10-hour prediction horizon
- **Features**: 30+ engineered features per timestamp
  - Temporal: hour, day, month (cyclical encoding)
  - Technical: returns, volatility, volume metrics
  - Lagged: 1h, 24h, 168h historical values
  - Rolling: 24h moving averages and statistics

**Training Process**:
1. Load historical data (ETHUSDT, XRPUSDT, etc.)
2. Generate 30+ features per candle
3. Create sliding windows (40k input → 10h output)
4. Train with QuantileLoss for confidence intervals
5. Validate on held-out data
6. Early stopping based on validation loss

**Prediction Output**:
- Median forecast (50th percentile)
- 80% confidence interval (10th-90th percentile)
- 95% confidence interval (2nd-98th percentile)
- Trend classification (Bullish/Bearish/Neutral)

## ⚡ Performance

### System Requirements
- **Minimum**: Python 3.8+, 8GB RAM, 2GB free disk
- **Recommended**: Python 3.10+, 16GB RAM, 10GB free disk, GPU (optional)
- **GPU Training**: 5-10x faster with CUDA-compatible GPU

### Training Time Estimates
- **CPU**: ~4-8 hours (100 epochs, 2 symbols)
- **GPU**: ~30-60 minutes (100 epochs, 2 symbols)

### Model Size
- **Checkpoint**: ~50-100MB per model
- **Dataset**: ~27MB per symbol (50,000 candles)

## 🔧 Configuration

### Model Hyperparameters

Edit in GUI or create `model_config.json`:

```json
{
  "hidden_size": 128,
  "lstm_layers": 2,
  "attention_head_size": 4,
  "dropout": 0.1,
  "batch_size": 32,
  "max_epochs": 100,
  "learning_rate": 0.001
}
```

### Training Presets

- **Small** (fast testing): hidden_size=64, context=1000
- **Medium** (balanced): hidden_size=128, context=10000
- **Large** (best accuracy): hidden_size=256, context=40000

## 📊 Metrics Explained

### Training Metrics

- **MAE** (Mean Absolute Error): Average prediction error in dollars
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **MAPE** (Mean Absolute Percentage Error): Error as percentage
- **R²** (Coefficient of Determination): Model fit quality (0-1)
- **Direction Accuracy**: % correct trend prediction (up/down)

### Good Model Performance
- MAPE < 3% (excellent)
- MAPE < 5% (good)
- Direction Accuracy > 60% (better than random)

## 🐛 Troubleshooting

### "PyTorch not available"
```bash
pip install torch pytorch-lightning pytorch-forecasting
```

### "Insufficient data"
- Need at least 40,000 candles for full context
- Download more data via Data Pipeline
- Or reduce context_length in config

### "Out of memory" during training
- Reduce batch_size (try 16, 8, or 4)
- Reduce context_length (try 10000 or 20000)
- Close other applications
- Use CPU instead of GPU if GPU memory limited

### Training very slow
- Consider reducing max_epochs (try 20-50)
- Use smaller context_length
- Enable GPU acceleration if available
- Use "Small" or "Medium" preset

### Predictions unrealistic
- Ensure model trained sufficiently (check validation loss)
- Verify data preprocessing (check logs)
- Try retraining with more epochs
- Check if data is normalized correctly

## 📝 Version History

### v4.0.0 (Current) - AI Integration
- ✨ Added Temporal Fusion Transformer for predictions
- ✨ Multi-asset AI model training
- ✨ 10-hour probabilistic forecasts
- ✨ Live training visualization
- ✨ Comprehensive feature engineering (30+ features)
- ✨ Model management and versioning
- 🎨 New AI Training view in GUI
- 📊 TensorBoard integration
- 📁 Reorganized project structure
- 📚 Complete ML documentation

### v3.0.0 - Unified Application
- Integrated data pipeline into main app
- Unified 3-panel interface

### v2.0.0 - Candlestick Charts
- Added chart visualization
- Multiple timeframes

### v1.0.0 - Initial Release
- Market monitoring
- Basic filtering

## 📄 License

Educational and personal use only.

## ⚠️ Disclaimer

**This software is for educational and informational purposes only.**

- Not financial advice
- Past performance doesn't guarantee future results
- Cryptocurrency trading carries significant risk
- AI predictions are probabilistic and may be incorrect
- Always do your own research (DYOR)
- Never invest more than you can afford to lose

## 🙏 Acknowledgments

- **Binance API**: Market data provider
- **PyTorch Forecasting**: TFT implementation
- **PyTorch Lightning**: Training framework
- **Matplotlib**: Visualization library

---

**Built with ❤️ by a professional trader and machine learning engineer**

*For documentation on specific features, see the `docs/` folder.*
