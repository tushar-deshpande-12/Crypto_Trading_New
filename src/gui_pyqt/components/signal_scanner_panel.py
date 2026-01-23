"""
Signal Scanner Panel Component - Scan cryptos for trading signals

Scans cryptocurrencies for recent buy/sell signals based on selected strategies
within a user-specified timeframe.

Features:
- Timeframe selection (1m to 1w)
- Combo strategy selection (RSI+MACD, BB+RSI, MACD+MA, etc.)
- Past candles filter for signal recency
- AND/OR logic for combining strategy signals
- Results table with signal details
- Click to view chart functionality
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QProgressBar, QGroupBox, QFrame, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QSplitter, QTextEdit, QSpinBox, QButtonGroup, QRadioButton
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from typing import Dict, List, Optional
from src.gui_pyqt.styles import COLORS


class SignalScannerPanel(QWidget):
    """
    Signal Scanner Panel

    Scans cryptocurrencies for trading signals based on selected strategies
    and timeframes, with AND/OR logic support.

    Signals:
        scan_requested(dict): Emitted when scan is requested with config
        symbol_selected(dict): Emitted when a symbol is selected from results
    """

    scan_requested = pyqtSignal(dict)
    symbol_selected = pyqtSignal(dict)

    # Timeframe options (for CCXT)
    TIMEFRAMES = {
        '1m': '1 Minute',
        '3m': '3 Minutes',
        '5m': '5 Minutes',
        '15m': '15 Minutes',
        '30m': '30 Minutes',
        '1h': '1 Hour',
        '2h': '2 Hours',
        '4h': '4 Hours',
        '6h': '6 Hours',
        '8h': '8 Hours',
        '12h': '12 Hours',
        '1d': '1 Day',
        '3d': '3 Days',
        '1w': '1 Week',
    }

    # Available strategies (combo strategies + stochastic)
    STRATEGIES = {
        'rsi_macd': 'RSI + MACD Combo',
        'bb_rsi': 'Bollinger + RSI Combo',
        'macd_ma': 'MACD + MA Combo',
        'stoch_rsi': 'Stochastic + RSI Combo',
        'triple_ema': 'Triple EMA Crossover',
        'adx_macd': 'ADX + MACD Trend',
        'stochastic': 'Stochastic',
        'ensemble': 'Ensemble (All Combined)',
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_scanning = False
        self._scan_results = []
        self._market_data = []
        self._setup_ui()

    def _setup_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Header
        header = QLabel("Signal Scanner")
        header.setProperty("class", "title")
        header.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {COLORS['text_primary']};")
        layout.addWidget(header)

        # Description
        desc = QLabel(
            "Scan cryptocurrencies for trading signals using combo strategies. "
            "Use AND to require all strategies to agree, or OR to match any strategy."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {COLORS['text_secondary']}; margin-bottom: 10px;")
        layout.addWidget(desc)

        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Configuration
        config_panel = self._create_config_panel()
        splitter.addWidget(config_panel)

        # Right panel - Results
        results_panel = self._create_results_panel()
        splitter.addWidget(results_panel)

        splitter.setSizes([350, 850])
        layout.addWidget(splitter, stretch=1)

    def _create_config_panel(self) -> QWidget:
        """Create the configuration panel"""
        panel = QFrame()
        panel.setMaximumWidth(400)
        panel.setMinimumWidth(300)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 16, 0)

        # Timeframe Selection
        timeframe_group = QGroupBox("Time Frame")
        timeframe_layout = QVBoxLayout(timeframe_group)

        self.timeframe_combo = QComboBox()
        for key, label in self.TIMEFRAMES.items():
            self.timeframe_combo.addItem(label, key)
        self.timeframe_combo.setCurrentText('1 Hour')  # Default to 1h
        timeframe_layout.addWidget(self.timeframe_combo)

        timeframe_desc = QLabel("Candle timeframe for analysis")
        timeframe_desc.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 10px;")
        timeframe_layout.addWidget(timeframe_desc)

        layout.addWidget(timeframe_group)

        # Past Candles Filter
        candles_group = QGroupBox("Signal Recency Filter")
        candles_layout = QVBoxLayout(candles_group)

        candles_input_layout = QHBoxLayout()
        candles_input_layout.addWidget(QLabel("Accept signals from last"))

        self.past_candles_spin = QSpinBox()
        self.past_candles_spin.setRange(1, 100)
        self.past_candles_spin.setValue(5)
        self.past_candles_spin.setMinimumWidth(60)
        candles_input_layout.addWidget(self.past_candles_spin)

        candles_input_layout.addWidget(QLabel("candles"))
        candles_input_layout.addStretch()
        candles_layout.addLayout(candles_input_layout)

        candles_desc = QLabel("Only show signals that occurred within the last N candles")
        candles_desc.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 10px;")
        candles_desc.setWordWrap(True)
        candles_layout.addWidget(candles_desc)

        layout.addWidget(candles_group)

        # Strategy Selection
        strategy_group = QGroupBox("Strategies to Scan")
        strategy_layout = QVBoxLayout(strategy_group)

        self.strategy_checkboxes = {}
        for key, label in self.STRATEGIES.items():
            cb = QCheckBox(label)
            cb.setChecked(True)  # All strategies enabled by default
            self.strategy_checkboxes[key] = cb
            strategy_layout.addWidget(cb)

        # Select/Deselect all buttons
        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(lambda: self._set_all_strategies(True))
        btn_layout.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(lambda: self._set_all_strategies(False))
        btn_layout.addWidget(deselect_all_btn)

        strategy_layout.addLayout(btn_layout)
        layout.addWidget(strategy_group)

        # Strategy Logic (AND/OR)
        logic_group = QGroupBox("Strategy Logic")
        logic_layout = QVBoxLayout(logic_group)

        self.logic_button_group = QButtonGroup(self)

        self.or_radio = QRadioButton("OR - Match ANY selected strategy")
        self.or_radio.setChecked(True)
        self.logic_button_group.addButton(self.or_radio)
        logic_layout.addWidget(self.or_radio)

        self.and_radio = QRadioButton("AND - Match ALL selected strategies")
        self.logic_button_group.addButton(self.and_radio)
        logic_layout.addWidget(self.and_radio)

        logic_desc = QLabel("OR: Signal if any strategy triggers\nAND: Signal only if all strategies agree")
        logic_desc.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 10px;")
        logic_layout.addWidget(logic_desc)

        layout.addWidget(logic_group)

        # Signal Type Filter
        signal_group = QGroupBox("Signal Type")
        signal_layout = QVBoxLayout(signal_group)

        self.buy_checkbox = QCheckBox("BUY Signals")
        self.buy_checkbox.setChecked(True)
        signal_layout.addWidget(self.buy_checkbox)

        self.sell_checkbox = QCheckBox("SELL Signals")
        self.sell_checkbox.setChecked(True)
        signal_layout.addWidget(self.sell_checkbox)

        layout.addWidget(signal_group)

        # Scan Button
        self.scan_btn = QPushButton("Start Scanning")
        self.scan_btn.setMinimumHeight(50)
        self.scan_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_hover']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_secondary']};
            }}
        """)
        self.scan_btn.clicked.connect(self._on_scan_click)
        layout.addWidget(self.scan_btn)

        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready to scan")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        progress_layout.addWidget(self.status_label)

        layout.addWidget(progress_group)

        # Spacer
        layout.addStretch()

        return panel

    def _create_results_panel(self) -> QWidget:
        """Create the results panel"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Results header
        header_layout = QHBoxLayout()

        self.results_label = QLabel("Scan Results")
        self.results_label.setProperty("class", "header")
        self.results_label.setStyleSheet(f"font-size: 16px; font-weight: bold;")
        header_layout.addWidget(self.results_label)

        header_layout.addStretch()

        self.results_count_label = QLabel("0 signals found")
        self.results_count_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        header_layout.addWidget(self.results_count_label)

        layout.addLayout(header_layout)

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            "Symbol", "Signal", "Strategy", "Confidence",
            "Price", "24h Change", "Time"
        ])

        # Table styling
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.results_table.setSortingEnabled(True)
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        # Column sizing
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)

        # Double-click to view chart
        self.results_table.doubleClicked.connect(self._on_result_double_click)

        layout.addWidget(self.results_table, stretch=1)

        # Log output
        log_group = QGroupBox("Scan Log")
        log_layout = QVBoxLayout(log_group)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(120)
        self.log_output.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_secondary']};
                font-family: monospace;
                font-size: 11px;
            }}
        """)
        log_layout.addWidget(self.log_output)

        layout.addWidget(log_group)

        return panel

    def _set_all_strategies(self, checked: bool):
        """Set all strategy checkboxes to checked/unchecked"""
        for cb in self.strategy_checkboxes.values():
            cb.setChecked(checked)

    def _on_scan_click(self):
        """Handle scan button click"""
        if self._is_scanning:
            return

        # Get selected strategies
        selected_strategies = [
            key for key, cb in self.strategy_checkboxes.items()
            if cb.isChecked()
        ]

        if not selected_strategies:
            self.log("ERROR: Please select at least one strategy")
            return

        if not self.buy_checkbox.isChecked() and not self.sell_checkbox.isChecked():
            self.log("ERROR: Please select at least one signal type (BUY/SELL)")
            return

        # Get logic mode
        logic_mode = 'or' if self.or_radio.isChecked() else 'and'

        # Build scan config
        config = {
            'timeframe': self.timeframe_combo.currentData(),
            'past_candles': self.past_candles_spin.value(),
            'strategies': selected_strategies,
            'logic_mode': logic_mode,
            'include_buy': self.buy_checkbox.isChecked(),
            'include_sell': self.sell_checkbox.isChecked(),
        }

        self.log(f"Starting scan: {self.timeframe_combo.currentText()} timeframe, "
                f"last {config['past_candles']} candles, {logic_mode.upper()} logic")
        self.log(f"Strategies: {', '.join(selected_strategies)}")
        self.scan_requested.emit(config)

    def _on_result_double_click(self, index):
        """Handle double-click on result row"""
        row = index.row()
        if row < len(self._scan_results):
            result = self._scan_results[row]
            symbol_data = {
                'symbol': result.get('symbol', ''),
                'price': result.get('price', 0),
                'price_change_pct': result.get('change_pct', 0),
            }
            self.symbol_selected.emit(symbol_data)

    def set_market_data(self, data: List[Dict]):
        """Set market data for filtering"""
        self._market_data = data

    def set_scanning(self, is_scanning: bool):
        """Set scanning state"""
        self._is_scanning = is_scanning
        self.scan_btn.setEnabled(not is_scanning)

        if is_scanning:
            self.scan_btn.setText("Scanning...")
            self.results_table.setRowCount(0)
            self._scan_results = []
        else:
            self.scan_btn.setText("Start Scanning")

    def update_progress(self, current: int, total: int, message: str):
        """Update progress display"""
        if total > 0:
            pct = int((current / total) * 100)
            self.progress_bar.setValue(pct)
        self.status_label.setText(message)

    def log(self, message: str):
        """Add message to log output"""
        self.log_output.append(message)
        # Auto-scroll
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def display_results(self, results: List[Dict]):
        """
        Display scan results in the table.

        Args:
            results: List of dicts with keys:
                - symbol: Trading pair symbol
                - signal: BUY or SELL
                - strategy: Strategy name
                - confidence: Signal confidence (0-1)
                - price: Current price
                - change_pct: 24h price change %
                - timestamp: Signal timestamp
        """
        self._scan_results = results
        self.results_table.setRowCount(len(results))

        for row, result in enumerate(results):
            # Symbol
            symbol_item = QTableWidgetItem(result.get('symbol', ''))
            symbol_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setItem(row, 0, symbol_item)

            # Signal (colored)
            signal = result.get('signal', 'HOLD')
            signal_item = QTableWidgetItem(signal)
            signal_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if signal == 'BUY':
                signal_item.setForeground(QColor(COLORS['chart_green']))
            elif signal == 'SELL':
                signal_item.setForeground(QColor(COLORS['chart_red']))
            self.results_table.setItem(row, 1, signal_item)

            # Strategy
            strategy_item = QTableWidgetItem(result.get('strategy', ''))
            self.results_table.setItem(row, 2, strategy_item)

            # Confidence
            confidence = result.get('confidence', 0)
            conf_item = QTableWidgetItem(f"{confidence:.1%}")
            conf_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            # Color based on confidence
            if confidence >= 0.7:
                conf_item.setForeground(QColor(COLORS['chart_green']))
            elif confidence >= 0.4:
                conf_item.setForeground(QColor(COLORS['warning']))
            self.results_table.setItem(row, 3, conf_item)

            # Price
            price = result.get('price', 0)
            price_item = QTableWidgetItem(f"${price:,.4f}" if price < 1 else f"${price:,.2f}")
            price_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.results_table.setItem(row, 4, price_item)

            # 24h Change
            change = result.get('change_pct', 0)
            change_item = QTableWidgetItem(f"{change:+.2f}%")
            change_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if change >= 0:
                change_item.setForeground(QColor(COLORS['chart_green']))
            else:
                change_item.setForeground(QColor(COLORS['chart_red']))
            self.results_table.setItem(row, 5, change_item)

            # Time
            timestamp = result.get('timestamp', '')
            time_item = QTableWidgetItem(str(timestamp)[-8:] if timestamp else '')  # Show time portion
            time_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setItem(row, 6, time_item)

        # Update count label
        buy_count = sum(1 for r in results if r.get('signal') == 'BUY')
        sell_count = sum(1 for r in results if r.get('signal') == 'SELL')
        self.results_count_label.setText(
            f"{len(results)} signals found (BUY: {buy_count}, SELL: {sell_count})"
        )

        # Log summary
        self.log(f"Scan complete: {len(results)} signals found")
        self.log(f"  BUY signals: {buy_count}")
        self.log(f"  SELL signals: {sell_count}")

    def clear_results(self):
        """Clear all results"""
        self.results_table.setRowCount(0)
        self._scan_results = []
        self.results_count_label.setText("0 signals found")
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready to scan")
