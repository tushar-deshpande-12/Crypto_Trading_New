# Complete Usage Guide - Crypto AI Predictor v3.1

## 🎯 Three Ways to Interact with Symbols

### Method 1: Right-Click Context Menu (⭐ Recommended)

**Most Reliable Method**

1. **Right-click** on any cryptocurrency row
2. A context menu appears with options:
   ```
   📊 View Chart for BTCUSDT
   📥 Download Data for BTCUSDT
   ```
3. Click your desired action

**Why Recommended**: Clear, reliable, works every time!

---

### Method 2: Click Actions Column

**Quick Access Method**

Each row has an "Actions" column showing: `[Download] | [Chart]`

- **Click left side** (`[Download]`) → Download historical data
- **Click right side** (`[Chart]`) → View candlestick chart

**Tip**: If unsure where to click, use the right-click menu instead!

---

### Method 3: Double-Click Row

**Fastest Chart Access**

- **Double-click** any cryptocurrency row
- Instantly opens the chart view
- Perfect for quick price analysis

---

## 📊 Viewing Charts

### Steps:
1. Find your cryptocurrency (use search if needed)
2. Choose one of these methods:
   - **Right-click** → "View Chart"
   - **Click** right half of Actions column
   - **Double-click** the row
3. Chart view opens automatically
4. See 500 candlesticks (1-hour each)

### Chart Features:
- **Green candles**: Price went up
- **Red candles**: Price went down
- **Zoom**: Use matplotlib toolbar (if available)
- **Date labels**: Auto-formatted on X-axis
- **Professional styling**: Dark theme

### Chart Controls:
- Switch symbols: Just click/right-click another symbol
- Back to list: Already visible on the left panel
- Different view: Click "Data Pipeline" in top menu

---

## 📥 Downloading Historical Data

### Quick Start:
1. Find your cryptocurrency
2. **Right-click** → "Download Data" (recommended)
   - Or click left side of Actions column
3. Data Pipeline view opens
4. Select data amount:
   ```
   1,000 candles   = ~41 days     (quick test)
   5,000 candles   = ~208 days    (moderate)
   10,000 candles  = ~13.7 months (recommended) ⭐
   20,000 candles  = ~2.3 years   (ML training)
   50,000 candles  = ~5.7 years   (maximum)
   ```
5. Click **"🚀 Start Fetching Data"**
6. Wait for completion (10k candles ≈ 30 seconds)

### What Happens:
- Real-time progress bar updates
- Detailed log messages show progress
- Data saved to: `dataset/{SYMBOL}/{DATE}_{TIME}_{N}candles/`

### Data Files Created:
```
dataset/
└── BTCUSDT/
    └── 2026-01-10_14-30-00_10000candles/
        ├── metadata.json    # Info about dataset
        ├── data.csv         # For Excel, pandas
        └── data.json        # For APIs, code
```

### After Download:
- ✅ Data ready for AI/ML training
- ✅ Can load with Python/pandas
- ✅ Both CSV and JSON formats
- ✅ Metadata tracks when/what was fetched

---

## 🔍 Searching and Filtering

### Search Box Usage:
1. Type cryptocurrency symbol or name
2. Results filter automatically
3. Examples:
   - Type: `BTC` → Shows BTCUSDT
   - Type: `ETH` → Shows ETHUSDT
   - Type: `SOL` → Shows SOLUSDT

### Sorting Columns:
1. Click any column header to sort
2. Click again to reverse order
3. Columns you can sort:
   - **Symbol**: Alphabetical
   - **Price**: Price value
   - **Change%**: 24h price change
   - **Volume**: Trading volume

---

## 🔄 Switching Views

### Top Menu Bar:
- **📊 Chart View**: See candlestick charts
- **📥 Data Pipeline**: Download historical data

### When to Use Each:
- **Chart View**: Quick analysis, price trends, trading decisions
- **Data Pipeline**: Getting data for AI models, backtesting, research

### Quick Switch:
1. Click symbol's Download → Auto-switches to Data Pipeline
2. Click symbol's Chart → Auto-switches to Chart View
3. Manual switch: Click buttons in top menu bar

---

## 💡 Common Workflows

### Workflow 1: Quick Price Check
```
1. Launch app
2. Type symbol in search (e.g., "BTC")
3. Double-click row
4. View chart
```
⏱️ Time: 5 seconds

---

### Workflow 2: Download Data for ML
```
1. Launch app
2. Right-click desired symbol
3. Click "Download Data"
4. Select 10,000-20,000 candles
5. Click "Start Fetching Data"
6. Wait for completion
7. Data saved to dataset/
```
⏱️ Time: 30-60 seconds

---

### Workflow 3: Compare Multiple Coins
```
1. Search "BTC" → View chart
2. Search "ETH" → View chart
3. Search "SOL" → View chart
4. Analyze patterns across coins
```
⏱️ Time: 1-2 minutes

---

### Workflow 4: Build Dataset for Portfolio
```
1. Right-click BTC → Download (10k candles)
2. Go back to Chart View (top menu)
3. Right-click ETH → Download (10k candles)
4. Repeat for all coins in portfolio
5. All data saved to dataset/
```
⏱️ Time: 3-5 minutes for 5 coins

---

## ⌨️ Keyboard & Mouse Reference

### Mouse Actions:
| Action | Result |
|--------|--------|
| **Single click** (Actions column) | Download or Chart |
| **Right-click** (anywhere on row) | Context menu |
| **Double-click** (anywhere on row) | Open chart |
| **Scroll wheel** | Scroll symbol list |

### Search Box:
| Key | Action |
|-----|--------|
| **Type** | Filter symbols |
| **Enter** | (No effect, filters live) |
| **Escape** | Clear search |
| **Backspace** | Delete characters |

### Navigation:
- **Top menu buttons**: Switch views
- **Column headers**: Sort data
- **Scrollbar**: Navigate long lists

---

## 🎨 Visual Guide

### Symbol List Layout:
```
┌─────────────────────────────────────────┐
│ Search: [____]  [↻ Refresh]            │
│ Tip: Right-click for menu | Double-click│
├─────┬────────┬──────┬────────┬──────────┤
│ BTC │ $45234 │ +2.5%│ $1.2B  │[D] | [C] │ ← Click left for Download
│ ETH │ $2345  │ -1.2%│ $800M  │[D] | [C] │ ← Click right for Chart
│ SOL │ $102   │ +5.3%│ $200M  │[D] | [C] │ ← Or right-click row
└─────┴────────┴──────┴────────┴──────────┘
         ↑        ↑       ↑
      Price   Change%  Volume
```

### Chart View:
```
┌─────────────────────────────────────────┐
│ BTCUSDT - Candlestick Chart            │
├─────────────────────────────────────────┤
│         📈 Chart Display               │
│     ┌──┐  ┌──┐                         │
│  ┌──┤  ├──┤  ├──┐                      │
│  │  └──┘  └──┘  │                      │
│  │              │                      │
│  │ Price Action │                      │
└─────────────────────────────────────────┘
```

### Data Pipeline View:
```
┌─────────────────────────────────────────┐
│ 📥 Data Pipeline                       │
├─────────────────────────────────────────┤
│ Target: BTCUSDT                        │
│ Max Candles: [10000 (~13.7 months) ▼] │
│ [🚀 Start Fetching Data]               │
│                                         │
│ Progress: [████████░░] 80%             │
│ Status: Fetching 8000/10000 candles    │
│                                         │
│ Log:                                    │
│ [12:34:56] Starting fetch...           │
│ [12:35:02] Batch 1/10 complete         │
│ [12:35:08] Batch 2/10 complete         │
└─────────────────────────────────────────┘
```

---

## 🚨 Tips & Best Practices

### ✅ DO:
- Use **right-click menu** for most reliable interaction
- **Download 10,000 candles** for good balance (size vs history)
- **Search before scrolling** to find symbols quickly
- **Double-click** for fastest chart access
- **Wait for downloads** to complete (don't close app)

### ❌ DON'T:
- Don't download 50k candles unless you really need years of data
- Don't spam-click Download (it queues up)
- Don't close app during downloads (data may be incomplete)
- Don't manually edit dataset files (use the app)

---

## 📈 Data Usage Examples

### Load Data in Python:
```python
from src.data import DataManager

# Initialize manager
manager = DataManager()

# Load latest BTC data
data = manager.load_latest("BTCUSDT")

# Convert to pandas DataFrame
import pandas as pd
df = pd.DataFrame(data)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Now ready for analysis!
print(df.head())
print(df.describe())
```

### Check Available Data:
```python
from src.data import DataManager

manager = DataManager()

# List all datasets
datasets = manager.list_all_datasets()
for ds in datasets:
    print(f"{ds['symbol']}: {ds['candle_count']} candles")

# Get storage stats
stats = manager.get_storage_statistics()
print(f"Total: {stats['total_datasets']} datasets")
print(f"Size: {stats['total_size_mb']} MB")
```

### Load Specific Dataset:
```python
# When you have multiple versions
datasets = manager.list_all_datasets("BTCUSDT")

# Load most recent (index 0)
latest_data = manager.load_dataset_by_path(
    datasets[0]['path'],
    format='json'
)

# Or load older version (index 1, 2, etc.)
older_data = manager.load_dataset_by_path(
    datasets[1]['path'],
    format='csv'
)
```

---

## 🐛 Troubleshooting

### Issue: "Can't click Download/Chart"
**Solution**: Use right-click menu instead
1. Right-click the symbol row
2. Select from menu

### Issue: "Chart not showing"
**Solution**:
1. Check if matplotlib is installed: `pip install matplotlib`
2. Try another symbol
3. Check logs: `crypto_ai.log`

### Issue: "Download seems stuck"
**Solution**:
- It's working! Large downloads take time
- 10,000 candles ≈ 30 seconds
- 50,000 candles ≈ 2-3 minutes
- Watch progress bar and log

### Issue: "No symbols showing"
**Solution**:
- Check internet connection
- Click "Refresh" button
- Restart application
- Check logs for API errors

---

## 📚 Additional Resources

- **Quick Start**: `QUICKSTART.md`
- **Full Documentation**: `README_V3.md`
- **Code Structure**: `STRUCTURE.md`
- **Bug Fixes**: `FIXES_v3.1.md`
- **Data Pipeline**: `DATA_PIPELINE_README.md`
- **Examples**: `example_fetch_data.py`

---

## 🎓 Learning Path

### Beginner:
1. Launch app
2. Browse symbols
3. View a few charts
4. Download small dataset (1000 candles)

### Intermediate:
1. Download multiple symbols
2. Load data in Python
3. Basic pandas analysis
4. Understand data structure

### Advanced:
1. Download large datasets (20k+ candles)
2. Build ML prediction models
3. Programmatic data fetching
4. Custom analysis pipelines

---

**Ready to explore crypto markets?** 🚀

```bash
python app.py
```

**Happy Trading!** 📈
