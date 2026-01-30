"""
Chart Panel Component - Matplotlib candlestick chart with indicator overlays

Displays OHLCV candlestick charts with selectable technical indicators.
Supports subplot indicators (RSI, MACD, Stochastic) and overlay indicators
(Bollinger Bands, Moving Averages, Ichimoku).

Features:
- Multi-timescale analysis (1h, 4h, 1d, 1w)
- Dropdown to select which timescale's graph to render
- Strategy signal markers (MACD crossover, Bollinger Bands, MA crossover)
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QScrollArea, QGroupBox, QPushButton, QFrame, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from src.gui_pyqt.styles import get_chart_colors, COLORS

# Available timescales for multi-timescale analysis
TIMESCALES = {
    '1h': {'label': '1 Hour (Raw)', 'resample': None, 'description': 'Original hourly data'},
    '4h': {'label': '4 Hours', 'resample': '4h', 'description': '4-hour aggregated candles'},
    '1d': {'label': '1 Day', 'resample': '1D', 'description': 'Daily aggregated candles'},
    '1w': {'label': '1 Week', 'resample': '1W', 'description': 'Weekly aggregated candles'}
}


class ChartPanel(QWidget):
    """
    Chart Panel with Technical Indicator Overlays

    Features:
    - Candlestick chart with volume subplot
    - Selectable overlay indicators (MA, Bollinger, Ichimoku)
    - Selectable subplot indicators (RSI, MACD, Stochastic)
    - Strategy signal markers (buy/sell)

    Signals:
        indicator_changed(list): List of selected indicator names
    """

    indicator_changed = pyqtSignal(list)

    # Available indicators grouped by type
    OVERLAY_INDICATORS = {
        'sma_10': 'SMA 10',
        'sma_20': 'SMA 20',
        'ema_12': 'EMA 12',
        'bollinger': 'Bollinger Bands',
        'keltner': 'Keltner Channels',
        'ichimoku': 'Ichimoku Cloud'
    }

    SUBPLOT_INDICATORS = {
        'rsi': 'RSI (14)',
        'macd': 'MACD',
        'stochastic': 'Stochastic',
        'volume': 'Volume',
        'adx': 'ADX',
        'mfi': 'MFI'
    }

    # Available trading strategies
    STRATEGY_OPTIONS = {
        'none': 'None',
        'rsi': 'RSI Overbought/Oversold',
        'macd': 'MACD Crossover',
        'bollinger': 'Bollinger Bands',
        'ma_crossover': 'MA Crossover',
        'stochastic': 'Stochastic',
        'ensemble': 'Ensemble (Multi-Strategy)',
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._raw_chart_data = None  # Original 1h data
        self._chart_data = None  # Currently displayed data (may be resampled)
        self._symbol_data = None
        self._selected_overlays = set()
        self._selected_subplots = {'volume'}  # Volume on by default
        self._selected_strategy = 'none'
        self._strategy_signals = []
        self._selected_timescale = '1h'  # Default timescale
        self._timescale_data = {}  # Cache for resampled data per timescale
        self._setup_ui()

    def _setup_ui(self):
        """Initialize the UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Main chart area
        chart_layout = QVBoxLayout()

        # Title
        self.title_label = QLabel("Select a symbol to view chart")
        self.title_label.setProperty("class", "header")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chart_layout.addWidget(self.title_label)

        # Matplotlib figure
        colors = get_chart_colors()
        self.figure = Figure(figsize=(10, 8), facecolor=colors['background'])
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(400)

        # Navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        chart_layout.addWidget(self.toolbar)
        chart_layout.addWidget(self.canvas, stretch=1)

        layout.addLayout(chart_layout, stretch=4)

        # Indicator selector panel (right side)
        indicator_panel = self._create_indicator_panel()
        layout.addWidget(indicator_panel, stretch=1)

    def _create_indicator_panel(self) -> QWidget:
        """Create the indicator selection panel"""
        panel = QFrame()
        panel.setMaximumWidth(220)
        panel.setMinimumWidth(180)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        # Title
        title = QLabel("Chart Settings")
        title.setProperty("class", "header")
        layout.addWidget(title)

        # Timescale selection group
        timescale_group = QGroupBox("Timescale Analysis")
        timescale_layout = QVBoxLayout(timescale_group)

        self.timescale_combo = QComboBox()
        for key, info in TIMESCALES.items():
            self.timescale_combo.addItem(info['label'], key)
        self.timescale_combo.currentIndexChanged.connect(self._on_timescale_changed)
        timescale_layout.addWidget(self.timescale_combo)

        # Timescale description label
        self.timescale_desc_label = QLabel(TIMESCALES['1h']['description'])
        self.timescale_desc_label.setWordWrap(True)
        self.timescale_desc_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 9px;")
        timescale_layout.addWidget(self.timescale_desc_label)

        layout.addWidget(timescale_group)

        # Strategy signals group
        strategy_group = QGroupBox("Strategy Signals")
        strategy_layout = QVBoxLayout(strategy_group)

        self.strategy_combo = QComboBox()
        for key, label in self.STRATEGY_OPTIONS.items():
            self.strategy_combo.addItem(label, key)
        self.strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        strategy_layout.addWidget(self.strategy_combo)

        # Signal stats label
        self.signal_stats_label = QLabel("Select strategy to see signals")
        self.signal_stats_label.setWordWrap(True)
        self.signal_stats_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 9px;")
        strategy_layout.addWidget(self.signal_stats_label)

        layout.addWidget(strategy_group)

        # Overlay indicators group
        overlay_group = QGroupBox("Overlays")
        overlay_layout = QVBoxLayout(overlay_group)

        self.overlay_checkboxes = {}
        for key, label in self.OVERLAY_INDICATORS.items():
            cb = QCheckBox(label)
            cb.stateChanged.connect(lambda state, k=key: self._on_overlay_changed(k, state))
            self.overlay_checkboxes[key] = cb
            overlay_layout.addWidget(cb)

        layout.addWidget(overlay_group)

        # Subplot indicators group
        subplot_group = QGroupBox("Subplots")
        subplot_layout = QVBoxLayout(subplot_group)

        self.subplot_checkboxes = {}
        for key, label in self.SUBPLOT_INDICATORS.items():
            cb = QCheckBox(label)
            if key == 'volume':
                cb.setChecked(True)
            cb.stateChanged.connect(lambda state, k=key: self._on_subplot_changed(k, state))
            self.subplot_checkboxes[key] = cb
            subplot_layout.addWidget(cb)

        layout.addWidget(subplot_group)

        # Apply button
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._redraw_chart)
        layout.addWidget(apply_btn)

        # Spacer
        layout.addStretch()

        return panel

    def _on_timescale_changed(self, index: int):
        """Handle timescale selection change"""
        self._selected_timescale = self.timescale_combo.currentData()
        timescale_info = TIMESCALES.get(self._selected_timescale, TIMESCALES['1h'])
        self.timescale_desc_label.setText(timescale_info['description'])

        # Switch to the selected timescale data
        if self._raw_chart_data is not None:
            self._switch_timescale(self._selected_timescale)
            self._generate_strategy_signals()
            self._redraw_chart()

    def _on_strategy_changed(self, index: int):
        """Handle strategy selection change"""
        self._selected_strategy = self.strategy_combo.currentData()

        # Auto-enable relevant indicators for each strategy
        strategy_indicators = {
            'rsi': {'subplots': ['rsi']},
            'macd': {'subplots': ['macd']},
            'bollinger': {'overlays': ['bollinger']},
            'ma_crossover': {'overlays': ['sma_10', 'sma_20']},
            'stochastic': {'subplots': ['stochastic']},
            'ensemble': {
                'overlays': ['bollinger', 'sma_10', 'sma_20'],
                'subplots': ['rsi', 'macd', 'stochastic']
            },
        }

        if self._selected_strategy in strategy_indicators:
            indicators = strategy_indicators[self._selected_strategy]

            # Auto-check overlay checkboxes
            for key in indicators.get('overlays', []):
                if key in self.overlay_checkboxes:
                    self.overlay_checkboxes[key].setChecked(True)
                    self._selected_overlays.add(key)

            # Auto-check subplot checkboxes
            for key in indicators.get('subplots', []):
                if key in self.subplot_checkboxes:
                    self.subplot_checkboxes[key].setChecked(True)
                    self._selected_subplots.add(key)

        self._generate_strategy_signals()
        self._redraw_chart()

    def _generate_strategy_signals(self):
        """Generate signals for the selected strategy"""
        self._strategy_signals = []

        if self._selected_strategy == 'none' or self._chart_data is None:
            self.signal_stats_label.setText("Select strategy to see signals")
            return

        try:
            from src.ml.strategies import get_strategy

            strategy = get_strategy(self._selected_strategy)

            # Prepare data for strategy - ensure datetime is available as a column
            # Strategies access row.get('datetime', None) so we need it as a column
            strategy_data = self._chart_data.copy()
            if isinstance(strategy_data.index, pd.DatetimeIndex):
                strategy_data['datetime'] = strategy_data.index

            # Validate strategy has required features
            is_valid, missing = strategy.validate_data(strategy_data)
            if not is_valid:
                self.signal_stats_label.setText(f"Missing: {', '.join(missing[:3])}")
                print(f"[CHART] Strategy '{self._selected_strategy}' missing features: {missing}")
                return

            signals = strategy.generate_signals(strategy_data)

            # Filter to only BUY and SELL signals with confidence > 0
            self._strategy_signals = [
                s for s in signals if s.action in ('BUY', 'SELL') and s.confidence > 0
            ]

            # Update stats
            buy_count = sum(1 for s in self._strategy_signals if s.action == 'BUY')
            sell_count = sum(1 for s in self._strategy_signals if s.action == 'SELL')
            timescale_label = TIMESCALES[self._selected_timescale]['label']

            # For ensemble, show additional vote breakdown
            if self._selected_strategy == 'ensemble' and self._strategy_signals:
                # Get voting info from most recent signal
                last_signal = self._strategy_signals[-1] if self._strategy_signals else None
                if last_signal and last_signal.metadata:
                    votes = last_signal.metadata.get('votes', {})
                    strategy_signals = last_signal.metadata.get('strategy_signals', {})

                    # Build vote summary
                    vote_summary = []
                    for strat_name, strat_data in strategy_signals.items():
                        action = strat_data.get('action', 'HOLD')[:1]  # B/S/H
                        conf = strat_data.get('confidence', 0) * 100
                        vote_summary.append(f"{strat_name}:{action}")

                    self.signal_stats_label.setText(
                        f"Signals ({timescale_label}): {len(self._strategy_signals)}\n"
                        f"BUY: {buy_count} | SELL: {sell_count}\n"
                        f"Latest: {' | '.join(vote_summary)}"
                    )
                else:
                    self.signal_stats_label.setText(
                        f"Signals ({timescale_label}): {len(self._strategy_signals)}\n"
                        f"BUY: {buy_count} | SELL: {sell_count}"
                    )
            else:
                self.signal_stats_label.setText(
                    f"Signals ({timescale_label}): {len(self._strategy_signals)}\n"
                    f"BUY: {buy_count} | SELL: {sell_count}"
                )

        except Exception as e:
            self.signal_stats_label.setText(f"Error: {str(e)[:50]}")
            print(f"[CHART] Strategy signal error: {e}")
            import traceback
            traceback.print_exc()

    def _switch_timescale(self, timescale: str):
        """Switch to a different timescale view"""
        if timescale not in self._timescale_data:
            # Generate resampled data if not cached
            self._timescale_data[timescale] = self._resample_data(self._raw_chart_data, timescale)

        self._chart_data = self._timescale_data[timescale].copy()

    def _resample_data(self, df: pd.DataFrame, timescale: str) -> pd.DataFrame:
        """
        Resample OHLCV data to a different timescale.

        Args:
            df: DataFrame with datetime index and OHLCV columns
            timescale: Target timescale key (1h, 4h, 1d, 1w)

        Returns:
            Resampled DataFrame with recalculated indicators
        """
        timescale_info = TIMESCALES.get(timescale, TIMESCALES['1h'])
        resample_rule = timescale_info['resample']

        # If no resampling needed (1h raw data), just return copy with indicators
        if resample_rule is None:
            result = df.copy()
            self._calculate_indicators(result)
            return result

        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'datetime' in df.columns:
                df = df.set_index('datetime')
            else:
                raise ValueError("DataFrame must have datetime index or datetime column")

        # Resample OHLCV data
        resampled = df.resample(resample_rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Reset index to have datetime as column for consistency
        resampled = resampled.reset_index()
        resampled = resampled.set_index('datetime')

        # Recalculate all indicators for the new timescale
        self._calculate_indicators(resampled)

        return resampled

    def _on_overlay_changed(self, key: str, state: int):
        """Handle overlay checkbox change"""
        if state == Qt.CheckState.Checked.value:
            self._selected_overlays.add(key)
        else:
            self._selected_overlays.discard(key)

    def _on_subplot_changed(self, key: str, state: int):
        """Handle subplot checkbox change"""
        if state == Qt.CheckState.Checked.value:
            self._selected_subplots.add(key)
        else:
            self._selected_subplots.discard(key)

    def show_chart(self, symbol_data: Dict, chart_data: List[Dict]):
        """
        Display candlestick chart for symbol with multi-timescale support.

        Args:
            symbol_data: Dict with symbol info (symbol, price, etc.)
            chart_data: List of OHLCV dicts with datetime/timestamp, open, high, low, close, volume
        """
        self._symbol_data = symbol_data

        # Store raw data
        raw_df = pd.DataFrame(chart_data)

        # Handle both 'datetime' and 'timestamp' column names
        if 'timestamp' in raw_df.columns:
            raw_df['datetime'] = pd.to_datetime(raw_df['timestamp'])
        elif 'datetime' in raw_df.columns:
            raw_df['datetime'] = pd.to_datetime(raw_df['datetime'])

        # Set datetime as index
        raw_df.set_index('datetime', inplace=True)

        # Store raw data for multi-timescale analysis
        self._raw_chart_data = raw_df.copy()

        # Clear timescale cache when new data arrives
        self._timescale_data = {}

        # Generate data for selected timescale (triggers indicator calculation)
        self._switch_timescale(self._selected_timescale)

        # Regenerate strategy signals with new data
        self._generate_strategy_signals()

        # Update title with timescale info
        symbol = symbol_data.get('symbol', 'Unknown')
        price = symbol_data.get('price', 0)
        change = symbol_data.get('price_change_pct', 0)
        change_color = COLORS['chart_green'] if change >= 0 else COLORS['chart_red']
        timescale_label = TIMESCALES[self._selected_timescale]['label']

        self.title_label.setText(
            f"{symbol} | ${price:,.2f} | "
            f"<span style='color:{change_color}'>{change:+.2f}%</span> | "
            f"<span style='color:{COLORS['text_secondary']}'>{timescale_label}</span>"
        )

        self._redraw_chart()

    def _calculate_indicators(self, df: pd.DataFrame = None):
        """
        Calculate all technical indicators from OHLCV data.

        This method calculates indicators needed for:
        - Chart overlays (MA, Bollinger, Keltner, Ichimoku)
        - Chart subplots (RSI, MACD, Stochastic, ADX, MFI)
        - Trading strategies (MACD crossover, Bollinger, MA crossover)

        Args:
            df: DataFrame to calculate indicators on. If None, uses self._chart_data
        """
        if df is None:
            df = self._chart_data
        if df is None or len(df) == 0:
            return

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # Moving Averages
        df['sma_10'] = close.rolling(window=10).mean()
        df['sma_20'] = close.rolling(window=20).mean()
        df['ema_12'] = close.ewm(span=12, adjust=False).mean()
        df['ema_26'] = close.ewm(span=26, adjust=False).mean()

        # === Strategy-required ratio columns (for MA Crossover strategy) ===
        # These ratios are needed by MACrossoverStrategy
        df['sma10_sma20_ratio'] = (df['sma_10'] / df['sma_20'].replace(0, np.nan)) - 1.0  # Normalized around 0
        df['price_sma10_ratio'] = (close / df['sma_10'].replace(0, np.nan)) - 1.0  # Price vs SMA10
        df['price_sma20_ratio'] = (close / df['sma_20'].replace(0, np.nan)) - 1.0  # Price vs SMA20

        # Bollinger Bands (20-period, 2 std dev)
        df['bb_middle'] = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # === Strategy-required Bollinger columns (for Bollinger strategy) ===
        # bb_position: -1 (at lower band) to +1 (at upper band)
        bb_range = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
        df['bb_position'] = ((close - df['bb_lower']) / bb_range) * 2 - 1
        # bb_width: Width of bands relative to middle band
        df['bb_width'] = bb_range / df['bb_middle'].replace(0, np.nan)

        # Keltner Channels (20-period, 2 ATR)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=20).mean()
        df['keltner_middle'] = close.ewm(span=20, adjust=False).mean()
        df['keltner_upper'] = df['keltner_middle'] + (atr * 2)
        df['keltner_lower'] = df['keltner_middle'] - (atr * 2)

        # RSI (14-period)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD (for MACD Crossover strategy)
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Stochastic Oscillator (14-period)
        lowest_low = low.rolling(window=14).min()
        highest_high = high.rolling(window=14).max()
        stoch_range = (highest_high - lowest_low).replace(0, np.nan)
        df['stoch_k'] = 100 * (close - lowest_low) / stoch_range
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        # ADX (14-period)
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr.replace(0, np.nan))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df['adx'] = dx.rolling(window=14).mean()

        # MFI (14-period)
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        positive_mf = positive_flow.rolling(window=14).sum()
        negative_mf = negative_flow.rolling(window=14).sum()
        mfi_ratio = positive_mf / negative_mf.replace(0, np.nan)
        df['mfi'] = 100 - (100 / (1 + mfi_ratio))

        # Ichimoku Cloud components
        df['tenkan_sen'] = (high.rolling(9).max() + low.rolling(9).min()) / 2
        df['kijun_sen'] = (high.rolling(26).max() + low.rolling(26).min()) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        df['senkou_span_b'] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    def _redraw_chart(self):
        """Redraw the chart with current settings"""
        if self._chart_data is None or len(self._chart_data) == 0:
            return

        self.figure.clear()
        colors = get_chart_colors()
        df = self._chart_data.copy()

        # Calculate number of subplots needed
        n_subplots = 1  # Main price chart
        if self._selected_subplots:
            n_subplots += len(self._selected_subplots)

        # Create subplots with varying heights
        height_ratios = [3]  # Main chart takes 3 parts
        height_ratios.extend([1] * (n_subplots - 1))  # Subplots take 1 part each

        axes = self.figure.subplots(
            n_subplots, 1,
            gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.05},
            sharex=True
        )

        if n_subplots == 1:
            axes = [axes]

        main_ax = axes[0]
        subplot_idx = 1

        # Style main axis
        main_ax.set_facecolor(colors['background'])
        main_ax.tick_params(colors=colors['text'])
        main_ax.spines['bottom'].set_color(colors['grid'])
        main_ax.spines['top'].set_color(colors['grid'])
        main_ax.spines['left'].set_color(colors['grid'])
        main_ax.spines['right'].set_color(colors['grid'])

        # Draw candlesticks
        self._draw_candlesticks(main_ax, df, colors)

        # Draw overlay indicators
        self._draw_overlays(main_ax, df, colors)

        # Draw strategy signals on main chart
        self._draw_strategy_signals(main_ax, df, colors)

        # Draw subplot indicators
        for key in sorted(self._selected_subplots):
            if subplot_idx < len(axes):
                ax = axes[subplot_idx]
                ax.set_facecolor(colors['background'])
                ax.tick_params(colors=colors['text'])
                self._draw_subplot_indicator(ax, df, key, colors)
                subplot_idx += 1

        # Format x-axis
        main_ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        main_ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        self.figure.tight_layout()
        self.canvas.draw()

    def _draw_strategy_signals(self, ax, df: pd.DataFrame, colors: Dict):
        """Draw strategy buy/sell signals on chart"""
        if not self._strategy_signals:
            return

        dates = df.index
        buy_dates = []
        buy_prices = []
        sell_dates = []
        sell_prices = []

        for signal in self._strategy_signals:
            # Get timestamp from signal
            ts = signal.timestamp
            if ts is None:
                continue

            # Find matching date in index
            try:
                if ts in dates:
                    idx = dates.get_loc(ts)
                    price = signal.price if signal.price > 0 else df.iloc[idx]['close']

                    if signal.action == 'BUY':
                        buy_dates.append(ts)
                        buy_prices.append(price)
                    elif signal.action == 'SELL':
                        sell_dates.append(ts)
                        sell_prices.append(price)
            except (KeyError, IndexError):
                continue

        # Draw buy signals (green triangles pointing up)
        if buy_dates:
            ax.scatter(buy_dates, buy_prices, marker='^', color=colors['positive'],
                      s=100, zorder=5, label=f'BUY ({len(buy_dates)})', edgecolors='white', linewidths=0.5)

        # Draw sell signals (red triangles pointing down)
        if sell_dates:
            ax.scatter(sell_dates, sell_prices, marker='v', color=colors['negative'],
                      s=100, zorder=5, label=f'SELL ({len(sell_dates)})', edgecolors='white', linewidths=0.5)

        # Add legend if signals exist
        if buy_dates or sell_dates:
            ax.legend(loc='upper right', fontsize=8, facecolor=colors['background'],
                     edgecolor=colors['grid'], labelcolor=colors['text'])

    def _draw_candlesticks(self, ax, df: pd.DataFrame, colors: Dict):
        """Draw enhanced candlestick chart with improved visuals"""
        dates = df.index
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        # Determine candle colors
        up = closes >= opens
        down = ~up

        # Width of candles (in days for datetime index)
        if len(dates) > 1:
            diff = dates[1] - dates[0]
            if hasattr(diff, 'total_seconds'):
                width = 0.7 * diff.total_seconds() / 86400
            else:
                width = 0.7
        else:
            width = 0.01

        # Enhanced colors for better visibility
        up_color = '#26a69a'  # Teal green
        down_color = '#ef5350'  # Coral red
        up_wick = '#4db6ac'  # Lighter teal for wicks
        down_wick = '#e57373'  # Lighter red for wicks

        # Draw wicks with gradient effect
        for i in range(len(dates)):
            wick_color = up_wick if up[i] else down_wick
            ax.plot([dates[i], dates[i]], [lows[i], highs[i]],
                   color=wick_color, linewidth=1.2, solid_capstyle='round')

        # Draw bodies with slight transparency for depth
        ax.bar(dates[up], closes[up] - opens[up], width, bottom=opens[up],
               color=up_color, edgecolor=up_color, alpha=0.95, linewidth=0.5)
        ax.bar(dates[down], opens[down] - closes[down], width, bottom=closes[down],
               color=down_color, edgecolor=down_color, alpha=0.95, linewidth=0.5)

        # Add current price annotation
        if len(closes) > 0:
            current_price = closes[-1]
            last_date = dates[-1]
            price_color = up_color if closes[-1] >= opens[-1] else down_color

            # Price line across chart
            ax.axhline(y=current_price, color=price_color, linestyle='--',
                      alpha=0.5, linewidth=1)

            # Price label on right side
            ax.annotate(f'${current_price:,.2f}',
                       xy=(last_date, current_price),
                       xytext=(5, 0), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       color=price_color,
                       bbox=dict(boxstyle='round,pad=0.3',
                                facecolor=colors['background'],
                                edgecolor=price_color, alpha=0.9))

        # Add high/low markers for visible range
        visible_high = highs.max()
        visible_low = lows.min()
        high_idx = np.argmax(highs)
        low_idx = np.argmin(lows)

        # High marker
        ax.annotate(f'H: ${visible_high:,.2f}',
                   xy=(dates[high_idx], visible_high),
                   xytext=(0, 8), textcoords='offset points',
                   fontsize=8, color=up_color, ha='center',
                   fontweight='bold')

        # Low marker
        ax.annotate(f'L: ${visible_low:,.2f}',
                   xy=(dates[low_idx], visible_low),
                   xytext=(0, -12), textcoords='offset points',
                   fontsize=8, color=down_color, ha='center',
                   fontweight='bold')

        ax.set_ylabel('Price (USD)', color=colors['text'], fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2, color=colors['grid'], linestyle='-', linewidth=0.5)

        # Add subtle background gradient effect
        ax.set_facecolor(colors['background'])

    def _draw_overlays(self, ax, df: pd.DataFrame, colors: Dict):
        """Draw overlay indicators on main chart"""
        dates = df.index

        # Moving Averages
        if 'sma_10' in self._selected_overlays and 'sma_10' in df.columns:
            ax.plot(dates, df['sma_10'], label='SMA 10', color='#ff9800', linewidth=1.5, alpha=0.9)
        if 'sma_20' in self._selected_overlays and 'sma_20' in df.columns:
            ax.plot(dates, df['sma_20'], label='SMA 20', color='#2196f3', linewidth=1.5, alpha=0.9)
        if 'ema_12' in self._selected_overlays and 'ema_12' in df.columns:
            ax.plot(dates, df['ema_12'], label='EMA 12', color='#9c27b0', linewidth=1.5, alpha=0.9)

        # Bollinger Bands
        if 'bollinger' in self._selected_overlays:
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                ax.plot(dates, df['bb_upper'], color='#4fc3f7', linewidth=1, alpha=0.8, label='BB Upper')
                ax.plot(dates, df['bb_lower'], color='#4fc3f7', linewidth=1, alpha=0.8, label='BB Lower')
                if 'bb_middle' in df.columns:
                    ax.plot(dates, df['bb_middle'], color='#4fc3f7', linewidth=1, linestyle='--', alpha=0.6)
                ax.fill_between(dates, df['bb_lower'], df['bb_upper'], alpha=0.1, color='#4fc3f7')

        # Keltner Channels
        if 'keltner' in self._selected_overlays:
            if 'keltner_upper' in df.columns and 'keltner_lower' in df.columns:
                ax.plot(dates, df['keltner_upper'], color='#ffab00', linewidth=1, alpha=0.8, label='Keltner Upper')
                ax.plot(dates, df['keltner_lower'], color='#ffab00', linewidth=1, alpha=0.8, label='Keltner Lower')
                if 'keltner_middle' in df.columns:
                    ax.plot(dates, df['keltner_middle'], color='#ffab00', linewidth=1, linestyle='--', alpha=0.6)
                ax.fill_between(dates, df['keltner_lower'], df['keltner_upper'], alpha=0.08, color='#ffab00')

        # Ichimoku Cloud
        if 'ichimoku' in self._selected_overlays:
            if 'tenkan_sen' in df.columns and 'kijun_sen' in df.columns:
                ax.plot(dates, df['tenkan_sen'], label='Tenkan-sen', color='#e91e63', linewidth=1)
                ax.plot(dates, df['kijun_sen'], label='Kijun-sen', color='#3f51b5', linewidth=1)

                # Draw cloud (Kumo)
                if 'senkou_span_a' in df.columns and 'senkou_span_b' in df.columns:
                    span_a = df['senkou_span_a']
                    span_b = df['senkou_span_b']
                    # Green cloud when span_a > span_b, red when span_b > span_a
                    ax.fill_between(dates, span_a, span_b,
                                   where=(span_a >= span_b),
                                   alpha=0.15, color=colors['positive'], interpolate=True)
                    ax.fill_between(dates, span_a, span_b,
                                   where=(span_a < span_b),
                                   alpha=0.15, color=colors['negative'], interpolate=True)

        if self._selected_overlays:
            ax.legend(loc='upper left', fontsize=8, facecolor=colors['background'],
                     edgecolor=colors['grid'], labelcolor=colors['text'])

    def _draw_subplot_indicator(self, ax, df: pd.DataFrame, indicator: str, colors: Dict):
        """Draw subplot indicator"""
        dates = df.index
        ax.set_facecolor(colors['background'])
        ax.grid(True, alpha=0.3, color=colors['grid'])
        ax.tick_params(colors=colors['text'])
        for spine in ax.spines.values():
            spine.set_color(colors['grid'])

        if indicator == 'volume':
            if 'volume' in df.columns:
                up = df['close'] >= df['open']
                volumes = df['volume'].values

                # Enhanced volume colors with gradient effect
                up_color = '#26a69a'  # Teal green
                down_color = '#ef5350'  # Coral red

                # Calculate relative volume for color intensity
                vol_ma = df['volume'].rolling(20).mean().fillna(df['volume'].mean())
                vol_ratio = df['volume'] / vol_ma

                # Create color array with intensity based on relative volume
                vol_colors = []
                vol_alphas = []
                for i, (is_up, ratio) in enumerate(zip(up, vol_ratio)):
                    base_color = up_color if is_up else down_color
                    # Higher volume = more opaque
                    alpha = min(0.9, max(0.3, 0.4 + ratio * 0.2))
                    vol_colors.append(base_color)
                    vol_alphas.append(alpha)

                # Calculate bar width based on time interval
                if len(dates) > 1:
                    diff = dates[1] - dates[0]
                    if hasattr(diff, 'total_seconds'):
                        width = 0.8 * diff.total_seconds() / 86400
                    else:
                        width = 0.8
                else:
                    width = 0.8

                # Draw volume bars
                for i, (date, vol, color, alpha) in enumerate(zip(dates, volumes, vol_colors, vol_alphas)):
                    ax.bar(date, vol, color=color, alpha=alpha, width=width, edgecolor='none')

                # Add volume moving average line
                vol_ma_values = vol_ma.values
                ax.plot(dates, vol_ma_values, color=colors['accent'], linewidth=1.2,
                       alpha=0.7, label='Vol MA(20)')

                # Add current volume annotation
                if len(volumes) > 0:
                    current_vol = volumes[-1]
                    avg_vol = vol_ma_values[-1] if len(vol_ma_values) > 0 else volumes.mean()
                    vol_change = ((current_vol / avg_vol) - 1) * 100 if avg_vol > 0 else 0

                    vol_text = f'Vol: {current_vol:,.0f}'
                    if vol_change > 20:
                        vol_text += f' (+{vol_change:.0f}%)'
                    elif vol_change < -20:
                        vol_text += f' ({vol_change:.0f}%)'

                    ax.annotate(vol_text,
                               xy=(dates[-1], current_vol),
                               xytext=(5, 0), textcoords='offset points',
                               fontsize=8, color=colors['text'],
                               bbox=dict(boxstyle='round,pad=0.2',
                                        facecolor=colors['background'],
                                        edgecolor=colors['grid'], alpha=0.8))

                ax.set_ylabel('Volume', color=colors['text'], fontsize=9, fontweight='bold')
                # Format volume axis with better readability
                ax.ticklabel_format(axis='y', style='scientific', scilimits=(6, 6))
                ax.legend(loc='upper left', fontsize=7, facecolor=colors['background'],
                         edgecolor=colors['grid'], labelcolor=colors['text'])

        elif indicator == 'rsi':
            if 'rsi' in df.columns:
                rsi = df['rsi'].dropna()
                ax.plot(dates[-len(rsi):], rsi, color=colors['primary'], linewidth=1.5)
                ax.axhline(70, color=colors['negative'], linestyle='--', alpha=0.7, linewidth=1)
                ax.axhline(30, color=colors['positive'], linestyle='--', alpha=0.7, linewidth=1)
                ax.axhline(50, color=colors['text'], linestyle=':', alpha=0.4)
                ax.set_ylim(0, 100)
                ax.set_ylabel('RSI', color=colors['text'], fontsize=9)
                # Fill overbought/oversold zones
                ax.fill_between(dates[-len(rsi):], 70, rsi.clip(lower=70), alpha=0.2, color=colors['negative'])
                ax.fill_between(dates[-len(rsi):], 30, rsi.clip(upper=30), alpha=0.2, color=colors['positive'])

        elif indicator == 'macd':
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                macd = df['macd'].dropna()
                signal = df['macd_signal'].dropna()
                # Align lengths
                min_len = min(len(macd), len(signal))
                plot_dates = dates[-min_len:]

                ax.plot(plot_dates, macd[-min_len:], color=colors['primary'], linewidth=1.5, label='MACD')
                ax.plot(plot_dates, signal[-min_len:], color='#ff5722', linewidth=1.5, label='Signal')

                # Histogram
                if 'macd_histogram' in df.columns:
                    hist = df['macd_histogram'].dropna()[-min_len:]
                    hist_colors = [colors['positive'] if h >= 0 else colors['negative'] for h in hist]
                    if len(plot_dates) > 1:
                        diff = plot_dates[1] - plot_dates[0]
                        if hasattr(diff, 'total_seconds'):
                            width = 0.8 * diff.total_seconds() / 86400
                        else:
                            width = 0.8
                    else:
                        width = 0.8
                    ax.bar(plot_dates, hist, color=hist_colors, alpha=0.5, width=width)

                ax.axhline(0, color=colors['text'], linestyle='-', alpha=0.4)
                ax.set_ylabel('MACD', color=colors['text'], fontsize=9)
                ax.legend(loc='upper left', fontsize=7, facecolor=colors['background'],
                         edgecolor=colors['grid'], labelcolor=colors['text'])

        elif indicator == 'stochastic':
            if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
                k = df['stoch_k'].dropna()
                d = df['stoch_d'].dropna()
                min_len = min(len(k), len(d))
                plot_dates = dates[-min_len:]

                ax.plot(plot_dates, k[-min_len:], color=colors['primary'], linewidth=1.5, label='%K')
                ax.plot(plot_dates, d[-min_len:], color=colors['accent'], linewidth=1.5, label='%D')
                ax.axhline(80, color=colors['negative'], linestyle='--', alpha=0.7, linewidth=1)
                ax.axhline(20, color=colors['positive'], linestyle='--', alpha=0.7, linewidth=1)
                ax.set_ylim(0, 100)
                ax.set_ylabel('Stoch', color=colors['text'], fontsize=9)
                ax.legend(loc='upper left', fontsize=7, facecolor=colors['background'],
                         edgecolor=colors['grid'], labelcolor=colors['text'])

        elif indicator == 'adx':
            if 'adx' in df.columns:
                adx = df['adx'].dropna()
                ax.plot(dates[-len(adx):], adx, color=colors['primary'], linewidth=1.5)
                ax.axhline(25, color=colors['accent'], linestyle='--', alpha=0.7, linewidth=1)
                ax.axhline(50, color=colors['negative'], linestyle='--', alpha=0.5, linewidth=1)
                ax.set_ylim(0, 100)
                ax.set_ylabel('ADX', color=colors['text'], fontsize=9)
                # Fill strong trend zone
                ax.fill_between(dates[-len(adx):], 25, adx.clip(lower=25), alpha=0.15, color=colors['accent'])

        elif indicator == 'mfi':
            if 'mfi' in df.columns:
                mfi = df['mfi'].dropna()
                ax.plot(dates[-len(mfi):], mfi, color=colors['primary'], linewidth=1.5)
                ax.axhline(80, color=colors['negative'], linestyle='--', alpha=0.7, linewidth=1)
                ax.axhline(20, color=colors['positive'], linestyle='--', alpha=0.7, linewidth=1)
                ax.set_ylim(0, 100)
                ax.set_ylabel('MFI', color=colors['text'], fontsize=9)
                # Fill overbought/oversold zones
                ax.fill_between(dates[-len(mfi):], 80, mfi.clip(lower=80), alpha=0.2, color=colors['negative'])
                ax.fill_between(dates[-len(mfi):], 20, mfi.clip(upper=20), alpha=0.2, color=colors['positive'])

    def add_trade_signals(self, signals: List[Dict]):
        """
        Add buy/sell markers to the chart.

        Args:
            signals: List of dicts with 'datetime', 'action' (BUY/SELL), 'price'
        """
        if self._chart_data is None or len(signals) == 0:
            return

        # Get main axis (first one)
        if len(self.figure.axes) == 0:
            return

        ax = self.figure.axes[0]
        colors = get_chart_colors()

        for signal in signals:
            dt = signal.get('datetime')
            action = signal.get('action')
            price = signal.get('price')

            if action == 'BUY':
                ax.scatter(dt, price, marker='^', color=colors['positive'], s=100, zorder=5)
            elif action == 'SELL':
                ax.scatter(dt, price, marker='v', color=colors['negative'], s=100, zorder=5)

        self.canvas.draw()

    def clear(self):
        """Clear the chart and reset all data"""
        self._raw_chart_data = None
        self._chart_data = None
        self._symbol_data = None
        self._timescale_data = {}
        self._strategy_signals = []
        self.figure.clear()
        self.canvas.draw()
        self.title_label.setText("Select a symbol to view chart")
        self.signal_stats_label.setText("Select strategy to see signals")
