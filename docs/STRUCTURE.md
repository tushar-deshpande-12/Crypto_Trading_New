# Project Structure - Crypto AI Predictor v3.0

## 📁 Folder Organization

```
crypto/ver3/
│
├── 📂 src/                          # Source code (modular architecture)
│   │
│   ├── 📂 core/                     # Core configuration & constants
│   │   ├── __init__.py             # Module exports
│   │   └── config.py               # AppConfig class - all settings
│   │
│   ├── 📂 api/                      # External API integrations
│   │   ├── __init__.py             # Module exports
│   │   └── binance_client.py       # Binance public API client
│   │
│   ├── 📂 data/                     # Data pipeline (fetch, store, manage)
│   │   ├── __init__.py             # Module exports
│   │   ├── fetcher.py              # CryptoDataFetcher - API data fetching
│   │   ├── storage.py              # DataStorage - file system operations
│   │   ├── manager.py              # DataManager - high-level coordinator
│   │   └── config.py               # DataPipelineConfig & presets
│   │
│   ├── 📂 gui/                      # Graphical user interface
│   │   ├── __init__.py             # Module exports
│   │   ├── app.py                  # CryptoAIPredictorApp - main window
│   │   ├── styles.py               # setup_styles() - centralized theming
│   │   │
│   │   ├── 📂 components/           # Reusable UI components
│   │   │   ├── __init__.py         # Component exports
│   │   │   ├── symbol_table.py     # SymbolTable - crypto list widget
│   │   │   ├── chart_panel.py      # ChartPanel - candlestick display
│   │   │   └── data_panel.py       # DataPanel - data fetch controls
│   │   │
│   │   ├── main_window.py          # (Legacy) Old tracker window
│   │   ├── chart_window.py         # (Legacy) Old chart window
│   │   └── data_fetch_window.py    # (Legacy) Old data fetch window
│   │
│   └── 📂 utils/                    # Utility functions
│       ├── __init__.py             # Module exports
│       └── formatters.py           # Data formatting helpers
│
├── 📂 dataset/                      # Downloaded cryptocurrency data
│   ├── README.md                   # Dataset documentation
│   ├── BTCUSDT/                    # Bitcoin datasets
│   │   └── 2026-01-10_14-30-00_10000candles/
│   │       ├── metadata.json       # Dataset information
│   │       ├── data.csv            # CSV format (pandas/Excel)
│   │       └── data.json           # JSON format (APIs/web)
│   ├── ETHUSDT/                    # Ethereum datasets
│   └── ...                         # Other cryptocurrencies
│
├── 📂 .claude/                      # Claude Code artifacts (ignore)
├── 📂 .git/                         # Git repository (ignore)
├── 📂 __pycache__/                  # Python cache (ignore)
│
├── 📄 app.py                        # ⭐ Main application launcher
├── 📄 run.bat                       # ⭐ Windows quick launcher
├── 📄 main.py                       # Legacy launcher (redirects to app.py)
│
├── 📄 requirements.txt              # Python dependencies
├── 📄 README_V3.md                  # ⭐ Main documentation
├── 📄 STRUCTURE.md                  # ⭐ This file - project structure
├── 📄 DATA_PIPELINE_README.md       # Data pipeline documentation
│
├── 📄 fetch_data.py                 # Standalone data fetch script
├── 📄 run_data_fetch.bat            # Data fetch launcher
├── 📄 example_fetch_data.py         # Usage examples
│
└── 📄 *.log                         # Application logs (created at runtime)
```

## 🏗️ Architecture Layers

### 1. Core Layer (`src/core/`)

**Purpose**: Application-wide configuration and constants

**Files**:
- `config.py`: `AppConfig` class with all settings
  - Application metadata (name, version)
  - Data configuration (dataset dir, intervals)
  - UI configuration (window size, colors)
  - API configuration (URLs, timeouts)

**Usage**:
```python
from src.core.config import AppConfig

window_width = AppConfig.WINDOW_WIDTH
dataset_dir = AppConfig.DATASET_DIR
```

---

### 2. API Layer (`src/api/`)

**Purpose**: External API integration (Binance)

**Files**:
- `binance_client.py`: `BinanceAPIClient` class
  - `get_24h_ticker()`: Market data for all symbols
  - `get_klines()`: Candlestick/OHLCV data
  - `get_usdt_pairs_detailed()`: Formatted USDT pairs

**Usage**:
```python
from src.api.binance_client import BinanceAPIClient

client = BinanceAPIClient()
market_data = client.get_usdt_pairs_detailed()
candles = client.get_klines_formatted("BTCUSDT", "1h", 500)
```

---

### 3. Data Layer (`src/data/`)

**Purpose**: Data pipeline (fetch, store, manage)

**Files**:

1. **`fetcher.py`**: `CryptoDataFetcher`
   - Fetches OHLCV data from Binance
   - `fetch_max_historical_data()`: Fetch up to 50k candles
   - Progress callbacks, rate limiting

2. **`storage.py`**: `DataStorage`
   - File system operations
   - Folder-based organization
   - CSV + JSON dual format
   - Metadata management

3. **`manager.py`**: `DataManager`
   - High-level coordinator
   - `fetch_and_save()`: One-stop data acquisition
   - `load_latest()`: Load most recent dataset
   - `check_data_freshness()`: Determine if new fetch needed

4. **`config.py`**: `DataPipelineConfig`
   - Pipeline configuration
   - Presets (quick_test, production, ml_training)

**Usage**:
```python
from src.data import DataManager

manager = DataManager(storage_dir="dataset", interval="1h")
dataset_path = manager.fetch_and_save("BTC", max_candles=10000)
data = manager.load_latest("BTCUSDT")
```

---

### 4. GUI Layer (`src/gui/`)

**Purpose**: User interface components

**Structure**:

```
gui/
├── app.py              # Main application window
├── styles.py           # Centralized theming
└── components/         # Modular UI widgets
    ├── symbol_table.py # Symbol list component
    ├── chart_panel.py  # Chart display component
    └── data_panel.py   # Data fetch component
```

**Main Application** (`app.py`):
- `CryptoAIPredictorApp`: Unified window
- Integrates all components
- Handles view switching
- Manages callbacks

**Components** (`components/`):

1. **`symbol_table.py`**: `SymbolTable`
   - Displays cryptocurrency list
   - Search/filter functionality
   - Download and Chart action buttons
   - Callbacks: `on_download_click`, `on_chart_click`

2. **`chart_panel.py`**: `ChartPanel`
   - Displays candlestick charts
   - Matplotlib integration
   - Fallback text mode

3. **`data_panel.py`**: `DataPanel`
   - Data fetch controls
   - Progress tracking
   - Log output
   - Callback: `on_fetch_click`

**Styles** (`styles.py`):
- `setup_styles()`: Configure ttk styles
- `get_color()`: Color palette helper
- Centralized dark theme

**Usage**:
```python
from src.gui.app import CryptoAIPredictorApp
import tkinter as tk

root = tk.Tk()
app = CryptoAIPredictorApp(root)
app.run()
```

---

### 5. Utils Layer (`src/utils/`)

**Purpose**: Utility functions and helpers

**Files**:
- `formatters.py`: Data formatting functions
  - Price formatting
  - Volume formatting
  - Timestamp conversions

---

## 📊 Data Flow

### Symbol List → Chart

```
User clicks "Chart" on symbol
    ↓
SymbolTable calls on_chart_click(symbol_data)
    ↓
App._on_chart_click() triggered
    ↓
Fetch candlestick data via BinanceAPIClient
    ↓
Switch to chart view
    ↓
ChartPanel.show_chart(symbol_data, chart_data)
    ↓
Chart rendered with matplotlib
```

### Symbol List → Data Download

```
User clicks "Download" on symbol
    ↓
SymbolTable calls on_download_click(symbol_data)
    ↓
App._on_download_click() triggered
    ↓
Switch to data view
    ↓
DataPanel.set_symbol(symbol_data)
    ↓
User configures max candles
    ↓
User clicks "Start Fetching"
    ↓
DataPanel calls on_fetch_click(symbol_data, max_candles)
    ↓
App._on_fetch_data() triggered
    ↓
DataManager.fetch_and_save() in background thread
    ↓
Progress updates via callbacks
    ↓
Data saved to dataset/{SYMBOL}/{TIMESTAMP}/
```

---

## 🎯 Component Communication

### Callback Pattern

Components use callbacks for loose coupling:

```python
# Parent sets up component with callback
symbol_table = SymbolTable(
    parent=frame,
    on_download_click=self._handle_download,
    on_chart_click=self._handle_chart
)

# Component calls callback when action occurs
def _on_tree_click(self, event):
    # ... determine action ...
    if self.on_download_click:
        self.on_download_click(symbol_data)
```

Benefits:
- **Decoupled**: Components don't know about parent
- **Reusable**: Same component works in different contexts
- **Testable**: Easy to mock callbacks

---

## 🔧 Configuration Hierarchy

```
src/core/config.py (AppConfig)
    ↓ Global settings
    ├── Application (name, version)
    ├── Data (dataset dir, interval)
    ├── UI (window size, colors)
    └── API (URLs, timeouts)

src/data/config.py (DataPipelineConfig)
    ↓ Data-specific settings
    ├── Storage (formats, directories)
    ├── Fetching (max candles, intervals)
    ├── Rate limiting (delays, retries)
    └── Data quality (validation, dedup)
```

**Usage**:
```python
# Application-wide settings
from src.core.config import AppConfig
window_size = (AppConfig.WINDOW_WIDTH, AppConfig.WINDOW_HEIGHT)

# Data pipeline settings
from src.data.config import DataPipelineConfig
config = DataPipelineConfig()
max_candles = config.default_max_candles
```

---

## 📦 Module Dependencies

```
app.py
  ├── src.gui.app
  │   ├── src.core.config
  │   ├── src.gui.styles
  │   ├── src.gui.components
  │   │   ├── symbol_table
  │   │   ├── chart_panel
  │   │   └── data_panel
  │   ├── src.api.binance_client
  │   └── src.data.manager
  │       ├── src.data.fetcher
  │       └── src.data.storage
  └── logging, tkinter (stdlib)
```

**Dependency Rules**:
1. **Core** has no dependencies (except stdlib)
2. **API** depends on: core
3. **Data** depends on: core, api (optional)
4. **GUI** depends on: core, api, data
5. **Utils** depends on: core

---

## 🚀 Entry Points

### Primary Entry Point
```
app.py → CryptoAIPredictorApp
```
⭐ **Recommended**: Unified GUI with all features

### Legacy Entry Points
```
main.py → Redirects to app.py
fetch_data.py → DataFetchWindow (standalone)
example_fetch_data.py → CLI examples
```

### Quick Launchers
```
run.bat → Launches app.py
run_data_fetch.bat → Launches fetch_data.py
```

---

## 📝 File Naming Conventions

### Python Files
- `snake_case.py` for all Python files
- Component names match class names
  - `symbol_table.py` → `class SymbolTable`
  - `chart_panel.py` → `class ChartPanel`

### Folders
- `lowercase` for all folders
- Semantic names: `core`, `api`, `data`, `gui`, `utils`

### Datasets
- `SYMBOL/YYYY-MM-DD_HH-MM-SS_Ncandles/`
- Example: `BTCUSDT/2026-01-10_14-30-00_10000candles/`

---

## 🎨 Code Organization Principles

### 1. Separation of Concerns
- Each module has one clear responsibility
- API ≠ Data ≠ GUI

### 2. Component-Based GUI
- Reusable, self-contained widgets
- Callbacks for communication
- No tight coupling

### 3. Centralized Configuration
- Single source of truth (`src/core/config.py`)
- Easy to modify settings
- Type-safe with dataclasses

### 4. Layer Architecture
- Core → API → Data → GUI
- Lower layers don't import from higher layers
- Clear dependency flow

### 5. Modular Folders
- Related files grouped together
- Easy to navigate
- Scalable structure

---

## 🔍 Finding Code

**Want to modify colors?**
→ `src/core/config.py` (AppConfig.COLOR_*)

**Want to add a new API endpoint?**
→ `src/api/binance_client.py` (add method)

**Want to change data storage format?**
→ `src/data/storage.py` (DataStorage class)

**Want to add a UI component?**
→ `src/gui/components/` (create new component)

**Want to adjust window layout?**
→ `src/gui/app.py` (CryptoAIPredictorApp._create_ui)

---

## ✅ Best Practices Used

1. **Type Hints**: All functions have type annotations
2. **Docstrings**: Every class and method documented
3. **Logging**: Comprehensive logging throughout
4. **Error Handling**: Try-except blocks with logging
5. **Threading**: Background tasks don't block GUI
6. **Callbacks**: Loose coupling via callbacks
7. **Configuration**: Centralized, not hardcoded
8. **Naming**: Clear, descriptive names
9. **Comments**: Explain "why", not "what"
10. **Modularity**: Small, focused files

---

**This structure ensures the code is:**
- ✅ Easy to understand
- ✅ Easy to maintain
- ✅ Easy to extend
- ✅ Easy to test
- ✅ Professional quality

---

Ready to explore the code? Start with:
1. `README_V3.md` - User guide
2. `src/gui/app.py` - Main application
3. `src/data/manager.py` - Data pipeline
4. `src/core/config.py` - Configuration
