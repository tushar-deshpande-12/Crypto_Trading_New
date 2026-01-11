# Cryptocurrency Data Pipeline for AI Prediction

Professional-grade data fetching and storage system for cryptocurrency price prediction models. This pipeline fetches historical 1-hour OHLCV (Open, High, Low, Close, Volume) data from Binance and organizes it in a structured, time-series folder format.

## 🚀 Quick Start

### Launch the Data Fetch GUI

**Windows:**
```bash
run_data_fetch.bat
```

**Python:**
```bash
python fetch_data.py
```

### Using the GUI

1. **Enter Symbol**: Type the cryptocurrency symbol (e.g., `BTC`, `ETH`, `SOL`)
   - Quick select buttons available for popular coins
   - Automatically appends `USDT` suffix

2. **Select Data Amount**: Choose how much historical data to fetch
   - 1,000 candles ≈ 41 days
   - 5,000 candles ≈ 208 days
   - 10,000 candles ≈ 13.7 months (default)
   - 20,000 candles ≈ 2.3 years
   - 50,000 candles ≈ 5.7 years

3. **Click "Fetch Data"**: Watch real-time progress as data downloads

4. **View Datasets**: Browse all stored datasets with metadata

## 📁 Data Storage Structure

```
dataset/
├── BTCUSDT/
│   ├── 2026-01-10_14-30-00_10000candles/
│   │   ├── metadata.json       # Dataset information
│   │   ├── data.csv            # CSV format (for Excel, pandas)
│   │   └── data.json           # JSON format (for APIs, web apps)
│   └── 2026-01-09_08-15-00_5000candles/
│       └── ...
├── ETHUSDT/
└── SOLUSDT/
```

### Folder Naming Convention

`{YYYY-MM-DD}_{HH-MM-SS}_{N}candles/`

- **Timestamp**: When data was fetched
- **Candle Count**: Number of 1-hour bars
- **Automatic Organization**: Old data coexists with new data for comparison

## 💻 Programmatic Usage

### Basic Example

```python
from src.data import DataManager

# Initialize the data manager
manager = DataManager(storage_dir="dataset", interval="1h")

# Fetch and save maximum historical data
dataset_path = manager.fetch_and_save(
    symbol="BTC",           # Or "BTCUSDT"
    max_candles=10000       # ~13.7 months
)

print(f"Data saved to: {dataset_path}")
```

### Load Existing Dataset

```python
# Load the most recent dataset
data = manager.load_latest("BTCUSDT")

# Access OHLCV data
for candle in data[:5]:
    print(f"{candle['datetime']}: O={candle['open']}, H={candle['high']}, "
          f"L={candle['low']}, C={candle['close']}, V={candle['volume']}")
```

### Check Data Freshness

```python
# Check if data needs updating
freshness = manager.check_data_freshness("BTCUSDT", time_threshold_hours=24)

if freshness['should_fetch_new']:
    print(f"Data is {freshness['fetch_timestamp']} old, fetching new...")
    manager.fetch_and_save("BTCUSDT", max_candles=10000)
else:
    print("Data is fresh, using existing dataset")
    data = manager.load_latest("BTCUSDT")
```

### Fetch Multiple Symbols

```python
symbols = ["BTC", "ETH", "BNB", "SOL", "XRP"]

results = manager.fetch_and_save_multiple(
    symbols=symbols,
    max_candles=5000
)

for symbol, path in results.items():
    if path:
        print(f"✓ {symbol}: {path}")
    else:
        print(f"✗ {symbol}: Failed")
```

### Dataset Management

```python
# List all datasets
datasets = manager.list_all_datasets()
for ds in datasets:
    print(f"{ds['symbol']}: {ds['candle_count']} candles from {ds['fetch_timestamp']}")

# Get storage statistics
stats = manager.get_storage_statistics()
print(f"Total: {stats['total_datasets']} datasets, {stats['total_size_mb']} MB")

# Cleanup old datasets (keep only 5 most recent per symbol)
deleted = manager.cleanup_old_datasets("BTCUSDT", keep_latest=5)
print(f"Deleted {deleted} old datasets")
```

## 🔧 Advanced Configuration

### Custom Configuration

```python
from src.data import DataManager, DataPipelineConfig

# Create custom configuration
config = DataPipelineConfig()
config.default_max_candles = 20000
config.data_freshness_threshold_hours = 12
config.keep_latest_datasets = 10

# Use configuration
manager = DataManager(
    storage_dir="my_custom_dataset",
    interval="1h"
)
```

### Configuration Presets

```python
from src.data import ConfigPresets

# Quick testing (100 candles)
config = ConfigPresets.quick_test()

# Production use (50,000 candles, auto cleanup)
config = ConfigPresets.production()

# ML training optimized (CSV only, 20,000 candles)
config = ConfigPresets.ml_training()

# Real-time prediction (recent data, auto-update)
config = ConfigPresets.real_time_prediction()
```

## 📊 Data Format

### OHLCV Fields

Each candlestick contains:

| Field | Type | Description |
|-------|------|-------------|
| `symbol` | string | Trading pair (e.g., "BTCUSDT") |
| `timestamp` | int | Unix timestamp in milliseconds |
| `datetime` | string | ISO format datetime |
| `open` | float | Opening price |
| `high` | float | Highest price in period |
| `low` | float | Lowest price in period |
| `close` | float | Closing price |
| `volume` | float | Trading volume (base asset) |
| `quote_volume` | float | Trading volume (quote asset, USDT) |
| `trades` | int | Number of trades |
| `taker_buy_volume` | float | Taker buy base asset volume |
| `taker_buy_quote_volume` | float | Taker buy quote asset volume |

### Metadata Format

```json
{
  "symbol": "BTCUSDT",
  "candle_count": 10000,
  "interval": "1h",
  "fetch_timestamp": "2026-01-10T14:30:00",
  "date_range": {
    "start": "2024-10-15T12:00:00",
    "end": "2026-01-10T14:00:00",
    "start_timestamp": 1729000000000,
    "end_timestamp": 1736520000000
  },
  "data_points": 10000,
  "files": ["metadata.json", "data.csv", "data.json"]
}
```

## 🎯 Use Cases

### 1. Initial Data Collection
```python
# Fetch maximum historical data for model training
manager.fetch_and_save("BTC", max_candles=50000)  # ~5.7 years
```

### 2. Regular Updates
```python
# Check and update data daily
if manager.storage.should_fetch_new_data("BTC", time_threshold_hours=24):
    manager.fetch_and_save("BTC", max_candles=10000)
```

### 3. Multiple Cryptocurrency Analysis
```python
# Fetch data for portfolio of coins
portfolio = ["BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOT", "MATIC"]
manager.fetch_and_save_multiple(portfolio, max_candles=10000)
```

### 4. Load Data for ML Training
```python
import pandas as pd

# Load latest dataset as DataFrame
data = manager.load_latest("BTCUSDT", format='json')
df = pd.DataFrame(data)

# Prepare features
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Now ready for model training
X = df[['open', 'high', 'low', 'close', 'volume']]
```

## 🏗️ Architecture

### Modular Design

```
src/data/
├── __init__.py              # Module exports
├── data_fetcher.py          # Binance API integration
├── data_storage.py          # Folder-based storage
├── data_manager.py          # High-level coordinator
└── config.py                # Configuration management
```

### Key Components

1. **CryptoDataFetcher**: Handles Binance API requests
   - Rate limiting (200ms between requests)
   - Automatic retries
   - Progress callbacks
   - Batch fetching for large datasets

2. **DataStorage**: Manages file system organization
   - Automatic folder creation
   - Dual format (CSV + JSON)
   - Metadata tracking
   - Time-based organization

3. **DataManager**: Coordinates operations
   - One-stop fetch-and-save
   - Data freshness checking
   - Multi-symbol fetching
   - Dataset cleanup

## ⚙️ Best Practices

### 1. Start Small, Then Scale
```python
# Test with small dataset first
manager.fetch_and_save("BTC", max_candles=1000)  # ~41 days

# Once verified, fetch full dataset
manager.fetch_and_save("BTC", max_candles=20000)  # ~2.3 years
```

### 2. Regular Data Updates
```python
# Set up daily cron job or scheduled task
import schedule

def update_data():
    manager.fetch_and_save("BTC", max_candles=10000)

schedule.every().day.at("00:00").do(update_data)
```

### 3. Cleanup Old Data
```python
# Keep storage manageable
for symbol in ["BTCUSDT", "ETHUSDT", "BNBUSDT"]:
    manager.cleanup_old_datasets(symbol, keep_latest=5)
```

### 4. Validate Data Quality
```python
data = manager.load_latest("BTC")

# Check for gaps
timestamps = [d['timestamp'] for d in data]
gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
expected_gap = 3600000  # 1 hour in ms

large_gaps = [g for g in gaps if g > expected_gap * 1.5]
if large_gaps:
    print(f"Warning: Found {len(large_gaps)} gaps in data")
```

## 🔍 Troubleshooting

### Issue: "No data fetched"
- **Cause**: Invalid symbol or Binance API down
- **Solution**: Verify symbol exists on Binance, check internet connection

### Issue: "Request timeout"
- **Cause**: Slow network or API rate limiting
- **Solution**: Reduce max_candles or increase timeout in config

### Issue: "Disk space full"
- **Cause**: Too many datasets stored
- **Solution**: Run cleanup: `manager.cleanup_old_datasets(symbol, keep_latest=3)`

### Issue: "Data has gaps"
- **Cause**: Market downtime or delisted trading pairs
- **Solution**: This is normal, handle in your prediction model

## 📈 Performance

- **Fetch Speed**: ~1000 candles per second
- **10,000 candles**: ~20-30 seconds
- **50,000 candles**: ~2-3 minutes
- **Storage**: ~50 KB per 1000 candles (JSON)

## 🛠️ Future Enhancements

- [ ] Support for multiple intervals (4h, 1d, etc.)
- [ ] Data compression for large datasets
- [ ] Gap filling with interpolation
- [ ] Real-time streaming data updates
- [ ] Export to HDF5 for faster loading
- [ ] Integration with ML frameworks (TensorFlow, PyTorch)

## 📝 License

Part of the Crypto Market Tracker project.

---

**Ready to build your AI prediction model?** Start by fetching maximum historical data!

```bash
python fetch_data.py
```
