# Cryptocurrency Dataset Storage

This directory contains historical OHLCV (Open, High, Low, Close, Volume) data for cryptocurrency trading pairs.

## Directory Structure

```
dataset/
├── BTCUSDT/
│   ├── 2026-01-10_14-30-00_10000candles/
│   │   ├── metadata.json          # Dataset metadata and information
│   │   ├── data.csv               # OHLCV data in CSV format
│   │   └── data.json              # OHLCV data in JSON format
│   └── 2026-01-09_08-15-00_5000candles/
│       └── ...
├── ETHUSDT/
│   └── ...
└── README.md                      # This file
```

## Folder Naming Convention

Each dataset is stored in a folder with the following format:
```
{YYYY-MM-DD}_{HH-MM-SS}_{N}candles/
```

- **Date & Time**: When the data was fetched
- **Candle Count**: Number of 1-hour candlesticks in the dataset

Example: `2026-01-10_14-30-00_10000candles` means:
- Fetched on January 10, 2026 at 14:30:00
- Contains 10,000 one-hour candlesticks (~416 days or ~13.7 months)

## Data Files

### metadata.json
Contains information about the dataset:
- Symbol and interval
- Number of candlesticks
- Date range (start and end)
- Fetch timestamp
- Custom metadata

Example:
```json
{
  "symbol": "BTCUSDT",
  "candle_count": 10000,
  "interval": "1h",
  "fetch_timestamp": "2026-01-10T14:30:00",
  "date_range": {
    "start": "2024-10-15T12:00:00",
    "end": "2026-01-10T14:00:00"
  }
}
```

### data.csv
OHLCV data in CSV format, optimized for:
- Data analysis tools (Excel, pandas)
- Machine learning pipelines
- Quick data inspection

Columns:
- symbol, timestamp, datetime, open, high, low, close, volume, quote_volume, trades, etc.

### data.json
OHLCV data in JSON format, optimized for:
- Web applications
- API integrations
- Programmatic access

## Time-Based Data Management

The folder-based approach allows for:

1. **Version Control**: Multiple snapshots of the same symbol at different times
2. **Data Freshness Tracking**: Easy to identify when data was last updated
3. **Incremental Updates**: New data fetches create new folders without overwriting old data
4. **Time Comparison**: Compare market conditions across different time periods
5. **Automatic Cleanup**: Old datasets can be removed while keeping recent ones

## Usage with Data Manager

```python
from src.data import DataManager

# Initialize manager
manager = DataManager(storage_dir="dataset", interval="1h")

# Fetch and save data
manager.fetch_and_save("BTCUSDT", max_candles=10000)

# Load latest dataset
data = manager.load_latest("BTCUSDT")

# Check data freshness
freshness = manager.check_data_freshness("BTCUSDT", time_threshold_hours=24)

# List all datasets
datasets = manager.list_all_datasets("BTCUSDT")

# Cleanup old datasets (keep only 5 most recent)
manager.cleanup_old_datasets("BTCUSDT", keep_latest=5)
```

## Data Freshness Strategy

When new data is fetched:
- If time difference is small (< 24 hours by default), it's considered a refresh
- If time difference is large, it's considered a new dataset
- Old datasets are automatically managed based on configuration

## Best Practices

1. **Regular Updates**: Fetch new data periodically to keep predictions current
2. **Backup Important Datasets**: Keep copies of datasets used for model training
3. **Clean Old Data**: Remove outdated datasets to save disk space
4. **Document Custom Metadata**: Use metadata field to track data provenance
5. **Monitor Storage**: Check storage statistics regularly

## Storage Statistics

Use the Data Manager to get storage statistics:

```python
stats = manager.get_storage_statistics()
print(f"Total datasets: {stats['total_datasets']}")
print(f"Total symbols: {stats['total_symbols']}")
print(f"Total size: {stats['total_size_mb']} MB")
```

---

**Note**: This directory is automatically managed by the Crypto Data Pipeline. Manual modifications should be done carefully to maintain data integrity.
