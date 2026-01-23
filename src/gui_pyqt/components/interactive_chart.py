"""
Interactive Chart Panel using Plotly

Features:
- Interactive candlestick charts with zoom/pan
- Multiple timeframes (1m to 1w)
- Pivot point levels (Classic, Fibonacci, Woodie, Camarilla)
- Support/Resistance detection
- Volume profile
- Technical indicators overlay
- Strategy signals
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QFrame, QGroupBox, QCheckBox, QSpinBox,
    QSplitter, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QUrl
from PyQt6.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
import tempfile
import os

from src.gui_pyqt.styles import COLORS, get_chart_colors
from src.gui_pyqt.utils.chart_calculations import (
    PivotType, calculate_pivot_points_from_df,
    find_support_resistance_levels, find_volume_profile_levels,
    auto_detect_fibonacci_levels
)


# All available timeframes
TIMEFRAMES = {
    '1m': {'label': '1 Min', 'ccxt': '1m'},
    '3m': {'label': '3 Min', 'ccxt': '3m'},
    '5m': {'label': '5 Min', 'ccxt': '5m'},
    '15m': {'label': '15 Min', 'ccxt': '15m'},
    '30m': {'label': '30 Min', 'ccxt': '30m'},
    '1h': {'label': '1 Hour', 'ccxt': '1h'},
    '2h': {'label': '2 Hour', 'ccxt': '2h'},
    '4h': {'label': '4 Hour', 'ccxt': '4h'},
    '6h': {'label': '6 Hour', 'ccxt': '6h'},
    '8h': {'label': '8 Hour', 'ccxt': '8h'},
    '12h': {'label': '12 Hour', 'ccxt': '12h'},
    '1d': {'label': '1 Day', 'ccxt': '1d'},
    '3d': {'label': '3 Day', 'ccxt': '3d'},
    '1w': {'label': '1 Week', 'ccxt': '1w'},
}


class InteractiveChartPanel(QWidget):
    """
    Interactive Chart Panel with Plotly

    Features:
    - Candlestick chart with volume
    - Multiple timeframes
    - Pivot points (multiple types)
    - Support/Resistance levels
    - Fibonacci retracements
    - Technical indicators
    """

    # Signals
    timeframe_changed = pyqtSignal(str)
    data_requested = pyqtSignal(str, str, int)  # symbol, timeframe, limit

    def __init__(self, parent=None):
        super().__init__(parent)

        self._chart_data: Optional[pd.DataFrame] = None
        self._symbol: str = ""
        self._current_timeframe = '1h'
        self._pivot_type = PivotType.CLASSIC
        self._show_pivots = True
        self._show_sr_levels = True
        self._show_fibonacci = False
        self._show_volume_profile = False
        self._strategy_signals = []
        self._selected_strategy = 'none'

        self._temp_html_path = None

        # Strategy options with descriptions
        self.STRATEGY_OPTIONS = {
            'none': 'None',
            'rsi': 'RSI Overbought/Oversold',
            'macd': 'MACD Crossover',
            'bollinger': 'Bollinger Bands',
            'ma_crossover': 'MA Crossover',
            'stochastic': 'Stochastic',
            'ensemble': 'Ensemble (Multi-Strategy)',
            # New combo strategies
            'rsi_macd': 'RSI + MACD Combo',
            'bb_rsi': 'Bollinger + RSI Combo',
            'macd_ma': 'MACD + MA Combo',
            'stoch_rsi': 'Stochastic + RSI Combo',
            'triple_ema': 'Triple EMA Crossover',
            'adx_macd': 'ADX + MACD Trend',
        }

        self._setup_ui()

    def _setup_ui(self):
        """Initialize the UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create splitter for chart and controls
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Main chart area
        chart_container = QWidget()
        chart_layout = QVBoxLayout(chart_container)
        chart_layout.setContentsMargins(8, 8, 8, 8)

        # Header with symbol info
        header_layout = QHBoxLayout()

        self.symbol_label = QLabel("Select a symbol")
        self.symbol_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['text_primary']};")
        header_layout.addWidget(self.symbol_label)

        self.price_label = QLabel("")
        self.price_label.setStyleSheet(f"font-size: 14px; color: {COLORS['text_secondary']};")
        header_layout.addWidget(self.price_label)

        header_layout.addStretch()

        # Timeframe selector
        header_layout.addWidget(QLabel("Timeframe:"))
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.setMinimumWidth(80)
        for key, info in TIMEFRAMES.items():
            self.timeframe_combo.addItem(info['label'], key)
        self.timeframe_combo.setCurrentText('1 Hour')
        self.timeframe_combo.currentIndexChanged.connect(self._on_timeframe_changed)
        header_layout.addWidget(self.timeframe_combo)

        # Candle limit
        header_layout.addWidget(QLabel("Candles:"))
        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(100, 5000)
        self.limit_spin.setValue(500)
        self.limit_spin.setSingleStep(100)
        self.limit_spin.setMinimumWidth(70)
        header_layout.addWidget(self.limit_spin)

        # Refresh button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._on_refresh_clicked)
        header_layout.addWidget(self.refresh_btn)

        chart_layout.addLayout(header_layout)

        # Web view for Plotly chart
        self.web_view = QWebEngineView()
        self.web_view.setMinimumHeight(500)
        chart_layout.addWidget(self.web_view, stretch=1)

        splitter.addWidget(chart_container)

        # Control panel
        control_panel = self._create_control_panel()
        splitter.addWidget(control_panel)

        splitter.setSizes([800, 200])
        layout.addWidget(splitter)

    def _create_control_panel(self) -> QWidget:
        """Create the control panel with chart options."""
        panel = QFrame()
        panel.setMaximumWidth(220)
        panel.setMinimumWidth(180)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)

        # Title
        title = QLabel("Chart Settings")
        title.setStyleSheet(f"font-weight: bold; color: {COLORS['text_primary']};")
        layout.addWidget(title)

        # Pivot Points Group
        pivot_group = QGroupBox("Pivot Points")
        pivot_layout = QVBoxLayout(pivot_group)

        self.show_pivots_cb = QCheckBox("Show Pivot Lines")
        self.show_pivots_cb.setChecked(True)
        self.show_pivots_cb.stateChanged.connect(self._on_option_changed)
        pivot_layout.addWidget(self.show_pivots_cb)

        pivot_layout.addWidget(QLabel("Pivot Type:"))
        self.pivot_type_combo = QComboBox()
        self.pivot_type_combo.addItem("Classic", PivotType.CLASSIC)
        self.pivot_type_combo.addItem("Fibonacci", PivotType.FIBONACCI)
        self.pivot_type_combo.addItem("Woodie", PivotType.WOODIE)
        self.pivot_type_combo.addItem("Camarilla", PivotType.CAMARILLA)
        self.pivot_type_combo.addItem("DeMark", PivotType.DEMARK)
        self.pivot_type_combo.currentIndexChanged.connect(self._on_option_changed)
        pivot_layout.addWidget(self.pivot_type_combo)

        layout.addWidget(pivot_group)

        # Support/Resistance Group
        sr_group = QGroupBox("Support/Resistance")
        sr_layout = QVBoxLayout(sr_group)

        self.show_sr_cb = QCheckBox("Show S/R Levels")
        self.show_sr_cb.setChecked(True)
        self.show_sr_cb.stateChanged.connect(self._on_option_changed)
        sr_layout.addWidget(self.show_sr_cb)

        self.show_volume_profile_cb = QCheckBox("Volume Profile Levels")
        self.show_volume_profile_cb.setChecked(False)
        self.show_volume_profile_cb.stateChanged.connect(self._on_option_changed)
        sr_layout.addWidget(self.show_volume_profile_cb)

        layout.addWidget(sr_group)

        # Fibonacci Group
        fib_group = QGroupBox("Fibonacci")
        fib_layout = QVBoxLayout(fib_group)

        self.show_fib_cb = QCheckBox("Show Fib Retracement")
        self.show_fib_cb.setChecked(False)
        self.show_fib_cb.stateChanged.connect(self._on_option_changed)
        fib_layout.addWidget(self.show_fib_cb)

        layout.addWidget(fib_group)

        # Indicators Group
        indicator_group = QGroupBox("Indicators")
        indicator_layout = QVBoxLayout(indicator_group)

        self.show_ma_cb = QCheckBox("Moving Averages")
        self.show_ma_cb.setChecked(False)
        self.show_ma_cb.stateChanged.connect(self._on_option_changed)
        indicator_layout.addWidget(self.show_ma_cb)

        self.show_bb_cb = QCheckBox("Bollinger Bands")
        self.show_bb_cb.setChecked(False)
        self.show_bb_cb.stateChanged.connect(self._on_option_changed)
        indicator_layout.addWidget(self.show_bb_cb)

        self.show_rsi_cb = QCheckBox("RSI (subplot)")
        self.show_rsi_cb.setChecked(False)
        self.show_rsi_cb.stateChanged.connect(self._on_option_changed)
        indicator_layout.addWidget(self.show_rsi_cb)

        self.show_macd_cb = QCheckBox("MACD (subplot)")
        self.show_macd_cb.setChecked(False)
        self.show_macd_cb.stateChanged.connect(self._on_option_changed)
        indicator_layout.addWidget(self.show_macd_cb)

        layout.addWidget(indicator_group)

        # Strategy Signals Group
        strategy_group = QGroupBox("Strategy Signals")
        strategy_layout = QVBoxLayout(strategy_group)

        self.strategy_combo = QComboBox()
        for key, label in self.STRATEGY_OPTIONS.items():
            self.strategy_combo.addItem(label, key)
        self.strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        strategy_layout.addWidget(self.strategy_combo)

        # Signal stats
        self.signal_stats_label = QLabel("Select strategy to see signals")
        self.signal_stats_label.setWordWrap(True)
        self.signal_stats_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 9px;")
        strategy_layout.addWidget(self.signal_stats_label)

        layout.addWidget(strategy_group)

        # Info labels
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 10px;")
        layout.addWidget(self.info_label)

        layout.addStretch()

        scroll.setWidget(content)

        # Wrap in layout
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(scroll)

        return panel

    def _on_timeframe_changed(self, index: int):
        """Handle timeframe change."""
        self._current_timeframe = self.timeframe_combo.currentData()
        self.timeframe_changed.emit(self._current_timeframe)

        if self._symbol:
            self._on_refresh_clicked()

    def _on_refresh_clicked(self):
        """Handle refresh button click."""
        if self._symbol:
            timeframe = self.timeframe_combo.currentData()
            limit = self.limit_spin.value()
            self.data_requested.emit(self._symbol, timeframe, limit)

    def _on_option_changed(self, state=None):
        """Handle option checkbox changes."""
        self._show_pivots = self.show_pivots_cb.isChecked()
        self._pivot_type = self.pivot_type_combo.currentData()
        self._show_sr_levels = self.show_sr_cb.isChecked()
        self._show_volume_profile = self.show_volume_profile_cb.isChecked()
        self._show_fibonacci = self.show_fib_cb.isChecked()

        if self._chart_data is not None:
            self._render_chart()

    def _on_strategy_changed(self, index: int):
        """Handle strategy selection change."""
        self._selected_strategy = self.strategy_combo.currentData()

        # Auto-enable relevant indicators for each strategy
        strategy_indicators = {
            'rsi': {'rsi': True},
            'macd': {'macd': True},
            'bollinger': {'bb': True},
            'ma_crossover': {'ma': True},
            'stochastic': {},
            'ensemble': {'rsi': True, 'macd': True, 'bb': True, 'ma': True},
            'rsi_macd': {'rsi': True, 'macd': True},
            'bb_rsi': {'bb': True, 'rsi': True},
            'macd_ma': {'macd': True, 'ma': True},
            'stoch_rsi': {'rsi': True},
            'triple_ema': {'ma': True},
            'adx_macd': {'macd': True},
        }

        if self._selected_strategy in strategy_indicators:
            indicators = strategy_indicators[self._selected_strategy]
            if indicators.get('rsi'):
                self.show_rsi_cb.setChecked(True)
            if indicators.get('macd'):
                self.show_macd_cb.setChecked(True)
            if indicators.get('bb'):
                self.show_bb_cb.setChecked(True)
            if indicators.get('ma'):
                self.show_ma_cb.setChecked(True)

        # Generate signals
        self._generate_strategy_signals()

        if self._chart_data is not None:
            self._render_chart()

    def _generate_strategy_signals(self):
        """Generate signals for the selected strategy."""
        self._strategy_signals = []

        if self._selected_strategy == 'none' or self._chart_data is None:
            self.signal_stats_label.setText("Select strategy to see signals")
            return

        try:
            from src.ml.strategies import get_strategy

            strategy = get_strategy(self._selected_strategy)

            # Prepare data - ensure datetime column exists
            strategy_data = self._chart_data.copy()
            if isinstance(strategy_data.index, pd.DatetimeIndex):
                strategy_data['datetime'] = strategy_data.index

            # Validate data
            is_valid, missing = strategy.validate_data(strategy_data)
            if not is_valid:
                self.signal_stats_label.setText(f"Missing: {', '.join(missing[:3])}")
                return

            # Generate signals
            signals = strategy.generate_signals(strategy_data)

            # Filter to BUY and SELL with confidence > 0
            self._strategy_signals = [
                s for s in signals if s.action in ('BUY', 'SELL') and s.confidence > 0
            ]

            # Update stats
            buy_count = sum(1 for s in self._strategy_signals if s.action == 'BUY')
            sell_count = sum(1 for s in self._strategy_signals if s.action == 'SELL')

            self.signal_stats_label.setText(
                f"Signals: {len(self._strategy_signals)}\n"
                f"BUY: {buy_count} | SELL: {sell_count}"
            )

        except Exception as e:
            self.signal_stats_label.setText(f"Error: {str(e)[:40]}")
            print(f"[CHART] Strategy error: {e}")
            import traceback
            traceback.print_exc()

    def set_symbol(self, symbol: str, price: float = 0, change_pct: float = 0):
        """Set the current symbol."""
        self._symbol = symbol

        # Update header
        self.symbol_label.setText(symbol)

        if price > 0:
            change_color = COLORS['chart_green'] if change_pct >= 0 else COLORS['chart_red']
            self.price_label.setText(
                f"${price:,.2f} "
                f"<span style='color:{change_color}'>{change_pct:+.2f}%</span>"
            )

    def set_data(self, df: pd.DataFrame):
        """
        Set chart data.

        Args:
            df: DataFrame with datetime index and OHLCV columns
        """
        if df is None or len(df) == 0:
            return

        self._chart_data = df.copy()

        # Calculate indicators
        self._calculate_indicators()

        # Regenerate strategy signals with new data
        self._generate_strategy_signals()

        # Render chart
        self._render_chart()

        # Update info
        self.info_label.setText(
            f"Loaded {len(df)} candles\n"
            f"From: {df.index[0].strftime('%Y-%m-%d %H:%M')}\n"
            f"To: {df.index[-1].strftime('%Y-%m-%d %H:%M')}"
        )

    def _calculate_indicators(self):
        """Calculate technical indicators on the data."""
        df = self._chart_data
        if df is None:
            return

        close = df['close']
        high = df['high']
        low = df['low']

        # Moving Averages
        df['sma_10'] = close.rolling(window=10).mean()
        df['sma_20'] = close.rolling(window=20).mean()
        df['sma_50'] = close.rolling(window=50).mean()
        df['ema_12'] = close.ewm(span=12, adjust=False).mean()
        df['ema_26'] = close.ewm(span=26, adjust=False).mean()

        # Bollinger Bands
        df['bb_middle'] = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # Bollinger Band position (-1 to 1 for strategies)
        bb_range = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
        df['bb_position'] = ((close - df['bb_lower']) / bb_range) * 2 - 1

        # Bollinger Band width (for strategy signals)
        df['bb_width'] = bb_range / df['bb_middle'].replace(0, np.nan)

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Stochastic Oscillator
        lowest_low = low.rolling(window=14).min()
        highest_high = high.rolling(window=14).max()
        stoch_range = (highest_high - lowest_low).replace(0, np.nan)
        df['stoch_k'] = 100 * (close - lowest_low) / stoch_range
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        # ADX (Average Directional Index)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()

        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr.replace(0, np.nan))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df['adx'] = dx.rolling(window=14).mean()

        # MA ratios for MA strategies
        df['sma10_sma20_ratio'] = (df['sma_10'] / df['sma_20'].replace(0, np.nan)) - 1.0
        df['price_sma10_ratio'] = (close / df['sma_10'].replace(0, np.nan)) - 1.0
        df['price_sma20_ratio'] = (close / df['sma_20'].replace(0, np.nan)) - 1.0

    def _render_chart(self):
        """Render the Plotly chart."""
        if self._chart_data is None or len(self._chart_data) == 0:
            return

        df = self._chart_data
        colors = get_chart_colors()

        # Determine subplot configuration
        rows = 1
        row_heights = [0.7]

        if self.show_rsi_cb.isChecked():
            rows += 1
            row_heights.append(0.15)

        if self.show_macd_cb.isChecked():
            rows += 1
            row_heights.append(0.15)

        # Always add volume as bottom subplot
        rows += 1
        row_heights.append(0.15 if rows > 2 else 0.3)

        # Create subplots
        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=row_heights,
            subplot_titles=None
        )

        # Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                increasing_line_color=colors['positive'],
                decreasing_line_color=colors['negative'],
                increasing_fillcolor=colors['positive'],
                decreasing_fillcolor=colors['negative'],
                name='Price',
                showlegend=False
            ),
            row=1, col=1
        )

        # Add pivot lines
        if self._show_pivots:
            self._add_pivot_lines(fig, df, colors)

        # Add support/resistance levels
        if self._show_sr_levels:
            self._add_sr_levels(fig, df, colors)

        # Add volume profile levels
        if self._show_volume_profile:
            self._add_volume_profile_levels(fig, df, colors)

        # Add Fibonacci levels
        if self._show_fibonacci:
            self._add_fibonacci_levels(fig, df, colors)

        # Add moving averages
        if self.show_ma_cb.isChecked():
            self._add_moving_averages(fig, df, colors)

        # Add Bollinger Bands
        if self.show_bb_cb.isChecked():
            self._add_bollinger_bands(fig, df, colors)

        # Add strategy signals
        if self._strategy_signals:
            self._add_strategy_signals(fig, df, colors)

        # Add RSI subplot
        current_row = 2
        if self.show_rsi_cb.isChecked():
            self._add_rsi_subplot(fig, df, colors, current_row)
            current_row += 1

        # Add MACD subplot
        if self.show_macd_cb.isChecked():
            self._add_macd_subplot(fig, df, colors, current_row)
            current_row += 1

        # Add volume subplot (always last)
        self._add_volume_subplot(fig, df, colors, current_row)

        # Update layout
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=colors['background'],
            plot_bgcolor=colors['background'],
            font_color=colors['text'],
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10)
            ),
            margin=dict(l=50, r=50, t=30, b=30),
            hovermode='x unified',
        )

        # Update all y-axes
        for i in range(1, rows + 1):
            fig.update_yaxes(
                gridcolor=colors['grid'],
                showgrid=True,
                zeroline=False,
                row=i, col=1
            )

        # Update x-axis
        fig.update_xaxes(
            gridcolor=colors['grid'],
            showgrid=True,
            rangeslider_visible=False
        )

        # Save to temporary HTML and load in web view
        # Use include_plotlyjs=True to embed the full library (avoids CDN issues)
        html_content = fig.to_html(
            include_plotlyjs=True,
            full_html=True,
            config={
                'scrollZoom': True,
                'displayModeBar': True,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'displaylogo': False
            }
        )

        # Write to temp file
        if self._temp_html_path is None:
            fd, self._temp_html_path = tempfile.mkstemp(suffix='.html')
            os.close(fd)

        with open(self._temp_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.web_view.setUrl(QUrl.fromLocalFile(self._temp_html_path))

    def _add_pivot_lines(self, fig, df: pd.DataFrame, colors: Dict):
        """Add pivot point lines to the chart."""
        pivots = calculate_pivot_points_from_df(df, self._pivot_type)

        pivot_colors = {
            'P': colors['text'],
            'R1': '#ef5350',
            'R2': '#f44336',
            'R3': '#d32f2f',
            'R4': '#b71c1c',
            'S1': '#66bb6a',
            'S2': '#4caf50',
            'S3': '#388e3c',
            'S4': '#1b5e20',
        }

        for level_name, price in pivots.to_dict().items():
            if price and price > 0:
                color = pivot_colors.get(level_name, colors['text'])
                dash = 'dash' if level_name == 'P' else 'dot'

                fig.add_hline(
                    y=price,
                    line_dash=dash,
                    line_color=color,
                    line_width=1,
                    annotation_text=f"{level_name}: ${price:,.2f}",
                    annotation_position="right",
                    annotation_font_size=9,
                    annotation_font_color=color,
                    row=1, col=1
                )

    def _add_sr_levels(self, fig, df: pd.DataFrame, colors: Dict):
        """Add support/resistance levels to the chart."""
        levels = find_support_resistance_levels(df, lookback=10, min_touches=2)

        for level in levels[:6]:  # Top 6 levels
            color = colors['negative'] if level.level_type == 'resistance' else colors['positive']
            opacity = 0.3 + (level.strength * 0.5)

            # Add shaded zone
            zone_height = level.price * 0.002  # 0.2% zone

            fig.add_hrect(
                y0=level.price - zone_height,
                y1=level.price + zone_height,
                fillcolor=color,
                opacity=opacity * 0.3,
                line_width=0,
                row=1, col=1
            )

            # Add line
            fig.add_hline(
                y=level.price,
                line_dash='dot',
                line_color=color,
                line_width=1,
                opacity=opacity,
                annotation_text=f"{level.level_type[0].upper()}: ${level.price:,.2f} ({level.touches}x)",
                annotation_position="left",
                annotation_font_size=8,
                annotation_font_color=color,
                row=1, col=1
            )

    def _add_volume_profile_levels(self, fig, df: pd.DataFrame, colors: Dict):
        """Add volume profile levels to the chart."""
        levels = find_volume_profile_levels(df, num_bins=30, significance_threshold=1.3)

        for level in levels[:5]:  # Top 5 volume nodes
            color = '#ff9800'  # Orange for volume levels
            opacity = 0.3 + (level['strength'] * 0.4)

            fig.add_hline(
                y=level['price'],
                line_dash='dashdot',
                line_color=color,
                line_width=1,
                opacity=opacity,
                annotation_text=f"VP: ${level['price']:,.2f}",
                annotation_position="right",
                annotation_font_size=8,
                annotation_font_color=color,
                row=1, col=1
            )

    def _add_fibonacci_levels(self, fig, df: pd.DataFrame, colors: Dict):
        """Add Fibonacci retracement levels to the chart."""
        fib_levels = auto_detect_fibonacci_levels(df, lookback=min(100, len(df)))

        fib_colors = {
            '0%': '#9e9e9e',
            '23.6%': '#4fc3f7',
            '38.2%': '#29b6f6',
            '50%': '#ffb74d',
            '61.8%': '#ffa726',
            '78.6%': '#ab47bc',
            '100%': '#9e9e9e',
        }

        for level_name, price in fib_levels.items():
            color = fib_colors.get(level_name, colors['accent'])

            fig.add_hline(
                y=price,
                line_dash='dot',
                line_color=color,
                line_width=1,
                opacity=0.7,
                annotation_text=f"Fib {level_name}: ${price:,.2f}",
                annotation_position="left",
                annotation_font_size=8,
                annotation_font_color=color,
                row=1, col=1
            )

    def _add_moving_averages(self, fig, df: pd.DataFrame, colors: Dict):
        """Add moving average lines to the chart."""
        if 'sma_10' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['sma_10'],
                    mode='lines',
                    name='SMA 10',
                    line=dict(color='#ff9800', width=1),
                    opacity=0.8
                ),
                row=1, col=1
            )

        if 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['sma_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='#2196f3', width=1),
                    opacity=0.8
                ),
                row=1, col=1
            )

        if 'sma_50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['sma_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='#9c27b0', width=1),
                    opacity=0.8
                ),
                row=1, col=1
            )

    def _add_bollinger_bands(self, fig, df: pd.DataFrame, colors: Dict):
        """Add Bollinger Bands to the chart."""
        if 'bb_upper' not in df.columns:
            return

        # Upper band
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['bb_upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='#4fc3f7', width=1),
                opacity=0.6
            ),
            row=1, col=1
        )

        # Lower band
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['bb_lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='#4fc3f7', width=1),
                fill='tonexty',
                fillcolor='rgba(79, 195, 247, 0.1)',
                opacity=0.6
            ),
            row=1, col=1
        )

        # Middle band
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['bb_middle'],
                mode='lines',
                name='BB Mid',
                line=dict(color='#4fc3f7', width=1, dash='dash'),
                opacity=0.4
            ),
            row=1, col=1
        )

    def _add_strategy_signals(self, fig, df: pd.DataFrame, colors: Dict):
        """Add strategy buy/sell signals to the chart."""
        if not self._strategy_signals:
            return

        buy_dates = []
        buy_prices = []
        buy_texts = []
        sell_dates = []
        sell_prices = []
        sell_texts = []

        for signal in self._strategy_signals:
            ts = signal.timestamp
            if ts is None:
                continue

            # Find price at signal time
            try:
                if ts in df.index:
                    price = signal.price if signal.price > 0 else df.loc[ts, 'close']
                else:
                    # Find nearest timestamp
                    idx = df.index.get_indexer([ts], method='nearest')[0]
                    if idx >= 0 and idx < len(df):
                        price = signal.price if signal.price > 0 else df.iloc[idx]['close']
                        ts = df.index[idx]
                    else:
                        continue

                conf = signal.confidence * 100
                if signal.action == 'BUY':
                    buy_dates.append(ts)
                    buy_prices.append(price)
                    buy_texts.append(f"BUY @ ${price:.2f}<br>Conf: {conf:.0f}%")
                elif signal.action == 'SELL':
                    sell_dates.append(ts)
                    sell_prices.append(price)
                    sell_texts.append(f"SELL @ ${price:.2f}<br>Conf: {conf:.0f}%")

            except (KeyError, IndexError):
                continue

        # Add buy signals (green triangles pointing up)
        if buy_dates:
            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_prices,
                    mode='markers',
                    name=f'BUY ({len(buy_dates)})',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color=colors['positive'],
                        line=dict(width=1, color='white')
                    ),
                    text=buy_texts,
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=1, col=1
            )

        # Add sell signals (red triangles pointing down)
        if sell_dates:
            fig.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode='markers',
                    name=f'SELL ({len(sell_dates)})',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color=colors['negative'],
                        line=dict(width=1, color='white')
                    ),
                    text=sell_texts,
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=1, col=1
            )

    def _add_rsi_subplot(self, fig, df: pd.DataFrame, colors: Dict, row: int):
        """Add RSI subplot."""
        if 'rsi' not in df.columns:
            return

        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['rsi'],
                mode='lines',
                name='RSI',
                line=dict(color=colors['primary'], width=1.5)
            ),
            row=row, col=1
        )

        # Overbought/oversold lines
        fig.add_hline(y=70, line_dash='dash', line_color=colors['negative'],
                     line_width=1, opacity=0.5, row=row, col=1)
        fig.add_hline(y=30, line_dash='dash', line_color=colors['positive'],
                     line_width=1, opacity=0.5, row=row, col=1)
        fig.add_hline(y=50, line_dash='dot', line_color=colors['text'],
                     line_width=1, opacity=0.3, row=row, col=1)

        fig.update_yaxes(title_text='RSI', range=[0, 100], row=row, col=1)

    def _add_macd_subplot(self, fig, df: pd.DataFrame, colors: Dict, row: int):
        """Add MACD subplot."""
        if 'macd' not in df.columns:
            return

        # MACD line
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['macd'],
                mode='lines',
                name='MACD',
                line=dict(color=colors['primary'], width=1.5)
            ),
            row=row, col=1
        )

        # Signal line
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['macd_signal'],
                mode='lines',
                name='Signal',
                line=dict(color='#ff5722', width=1.5)
            ),
            row=row, col=1
        )

        # Histogram
        hist_colors = [colors['positive'] if v >= 0 else colors['negative']
                      for v in df['macd_histogram'].fillna(0)]

        fig.add_trace(
            go.Bar(
                x=df.index, y=df['macd_histogram'],
                name='Histogram',
                marker_color=hist_colors,
                opacity=0.5
            ),
            row=row, col=1
        )

        fig.add_hline(y=0, line_color=colors['text'], line_width=1,
                     opacity=0.3, row=row, col=1)

        fig.update_yaxes(title_text='MACD', row=row, col=1)

    def _add_volume_subplot(self, fig, df: pd.DataFrame, colors: Dict, row: int):
        """Add volume subplot."""
        if 'volume' not in df.columns:
            return

        # Color bars based on price direction
        vol_colors = [colors['positive'] if df['close'].iloc[i] >= df['open'].iloc[i]
                     else colors['negative'] for i in range(len(df))]

        fig.add_trace(
            go.Bar(
                x=df.index, y=df['volume'],
                name='Volume',
                marker_color=vol_colors,
                opacity=0.7
            ),
            row=row, col=1
        )

        fig.update_yaxes(title_text='Volume', row=row, col=1)

    def add_strategy_signals(self, signals: List[Dict]):
        """
        Add strategy buy/sell signals to the chart.

        Args:
            signals: List of dicts with 'datetime', 'action', 'price', 'confidence'
        """
        self._strategy_signals = signals
        if self._chart_data is not None:
            self._render_chart()

    def clear(self):
        """Clear the chart."""
        self._chart_data = None
        self._symbol = ""
        self._strategy_signals = []
        self.symbol_label.setText("Select a symbol")
        self.price_label.setText("")
        self.info_label.setText("")
        self.web_view.setHtml("<html><body style='background-color: #1e1e1e;'></body></html>")

    def cleanup(self):
        """Clean up temporary files."""
        if self._temp_html_path and os.path.exists(self._temp_html_path):
            try:
                os.remove(self._temp_html_path)
            except:
                pass
