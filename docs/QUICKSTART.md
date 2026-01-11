# Quick Start Guide - Crypto AI Predictor v3.0

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Application
**Windows:**
```bash
run.bat
```

**Or Python:**
```bash
python app.py
```

### 3. Start Using!

## 📖 Basic Workflow

### View Crypto Prices
1. Application loads → Symbol list appears automatically
2. Browse 2000+ USDT trading pairs
3. See live prices, 24h change%, volume
4. Use search box to filter symbols

### View Charts
1. Find a crypto (e.g., search "BTC")
2. Click **"📊 Chart"** button or double-click the row
3. Chart appears instantly on the right panel
4. View last 500 candlesticks (1-hour interval)

### Download Historical Data
1. Find a crypto (e.g., search "ETH")
2. Click **"📥 Download"** button
3. Data Pipeline view opens automatically
4. Select data amount:
   - 10,000 candles recommended (~13.7 months)
   - Up to 50,000 candles available (~5.7 years)
5. Click **"🚀 Start Fetching Data"**
6. Watch progress in real-time
7. Data saved to: `dataset/{SYMBOL}/{TIMESTAMP}_{CANDLES}candles/`

### Switch Views
**Top menu bar:**
- Click **"📊 Chart View"** → See candlestick charts
- Click **"📥 Data Pipeline"** → Fetch historical data

## 💡 Tips

### For Market Analysis:
- Use the **Chart View** for quick visual analysis
- Sort by Volume to find most active coins
- Search multiple coins quickly

### For AI/ML Training:
- Use the **Data Pipeline** to fetch large datasets
- Start with 10,000 candles for testing
- Fetch 20,000-50,000 for production models
- Data is saved in both CSV and JSON formats

### For Quick Trading:
- Symbol list updates live prices
- Green = price up, Red = price down
- Click chart to see price trends
- Volume sorting shows market activity

## 📁 Where is My Data?

All downloaded data is saved in:
```
dataset/
└── {SYMBOL}/
    └── {DATE}_{TIME}_{CANDLES}candles/
        ├── metadata.json    # Dataset information
        ├── data.csv         # For pandas, Excel
        └── data.json        # For APIs, web apps
```

Example:
```
dataset/
└── BTCUSDT/
    └── 2026-01-10_14-30-00_10000candles/
        ├── metadata.json
        ├── data.csv
        └── data.json
```

## 🔧 Common Tasks

### Fetch Multiple Cryptocurrencies
1. Click **"📥 Download"** on first coin → Fetch
2. Click **"📊 Chart View"** (top menu) to go back
3. Click **"📥 Download"** on next coin → Fetch
4. Repeat as needed

### Load Data in Python
```python
from src.data import DataManager

manager = DataManager()
data = manager.load_latest("BTCUSDT")

# Use with pandas
import pandas as pd
df = pd.DataFrame(data)
```

### Check What Data You Have
```python
from src.data import DataManager

manager = DataManager()

# List all datasets
datasets = manager.list_all_datasets()
for ds in datasets:
    print(f"{ds['symbol']}: {ds['candle_count']} candles")

# Get storage stats
stats = manager.get_storage_statistics()
print(f"Total: {stats['total_datasets']} datasets, {stats['total_size_mb']} MB")
```

## ❓ Troubleshooting

### "Failed to load market data"
- Check internet connection
- Try again in a few seconds
- Binance API might be temporarily down

### "Chart not showing"
- Make sure matplotlib is installed: `pip install matplotlib`
- Try clicking "Chart" again

### Application is slow to start
- Normal! Loading 2000+ symbols takes 2-3 seconds
- Once loaded, everything is fast

### Where are the logs?
- Check `crypto_ai.log` in the application folder
- Contains detailed error information

## 📚 Learn More

- **Full Documentation**: `README_V3.md`
- **Project Structure**: `STRUCTURE.md`
- **Data Pipeline**: `DATA_PIPELINE_README.md`
- **Code Examples**: `example_fetch_data.py`

## 🎯 Next Steps

1. ✅ Get familiar with the interface
2. ✅ View a few charts
3. ✅ Download data for your favorite coins
4. ✅ Start building your AI prediction model!

---

## 📊 Interface Guide

```
┌─────────────────────────────────────────────────────────────────┐
│  📈 Crypto AI Predictor         [📊 Chart View] [📥 Data Pipeline] │  ← Menu bar
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┬─────────────────────────────────────────┐ │
│  │ Symbol List      │ Chart or Data Panel                     │ │
│  │ (Left Panel)     │ (Right Panel - switches views)          │ │
│  │                  │                                          │ │
│  │ [Search: ___]    │  ┌────────────────────────────────┐     │ │
│  │ [↻ Refresh]      │  │  BTCUSDT - Candlestick Chart  │     │ │
│  │                  │  │                                │     │ │
│  │ BTC  $45,234     │  │  📈 Chart displays here        │     │ │
│  │ 📥 Download | 📊 │  │                                │     │ │
│  │                  │  │  (or Data Pipeline controls)   │     │ │
│  │ ETH  $2,345      │  │                                │     │ │
│  │ 📥 Download | 📊 │  │                                │     │ │
│  │                  │  └────────────────────────────────┘     │ │
│  │ SOL  $102        │                                          │ │
│  │ 📥 Download | 📊 │                                          │ │
│  │                  │                                          │ │
│  │ ... 2000+ more   │                                          │ │
│  └──────────────────┴─────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ Ready                                        Symbols: 2043      │  ← Status bar
└─────────────────────────────────────────────────────────────────┘
```

**Actions:**
- **Single click** on Download/Chart buttons
- **Double-click** symbol row → Opens chart
- **Search box** → Type to filter symbols
- **Column headers** → Click to sort

---

**Happy Trading! 📈**

Questions? Check `README_V3.md` for detailed documentation!
