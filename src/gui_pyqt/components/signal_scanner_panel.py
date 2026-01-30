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
    QSplitter, QTextEdit, QSpinBox, QButtonGroup, QRadioButton,
    QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from src.gui_pyqt.styles import COLORS


class SignalScannerPanel(QWidget):
    """
    Signal Scanner Panel

    Scans cryptocurrencies for trading signals based on selected strategies
    and timeframes, with AND/OR logic support.

    Also provides AI confidence-based scanning using saved directionality models.

    Signals:
        scan_requested(dict): Emitted when scan is requested with config
        ai_scan_requested(dict): Emitted when AI scan is requested with config
        symbol_selected(dict): Emitted when a symbol is selected from results
    """

    scan_requested = pyqtSignal(dict)
    ai_scan_requested = pyqtSignal(dict)
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
        self._ai_scanning = False
        self._scan_results = []
        self._market_data = []
        self._ai_scan_worker = None
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
        """Create the configuration panel wrapped in a scroll area"""
        # Outer container holds the scroll area
        outer = QFrame()
        outer.setMaximumWidth(400)
        outer.setMinimumWidth(300)
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panel = QWidget()
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

        # AI Confidence Scanner Section
        ai_group = QGroupBox("AI Model Scanner")
        ai_layout = QVBoxLayout(ai_group)

        ai_desc = QLabel(
            "Scan using saved directionality models.\n"
            "Train models first in the Prediction tab."
        )
        ai_desc.setWordWrap(True)
        ai_desc.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 10px;")
        ai_layout.addWidget(ai_desc)

        # AI Timeframe
        ai_tf_layout = QHBoxLayout()
        ai_tf_layout.addWidget(QLabel("Timeframe:"))
        self.ai_timeframe_combo = QComboBox()
        for key, label in self.TIMEFRAMES.items():
            self.ai_timeframe_combo.addItem(label, key)
        self.ai_timeframe_combo.setCurrentText('1 Hour')
        ai_tf_layout.addWidget(self.ai_timeframe_combo)
        ai_layout.addLayout(ai_tf_layout)

        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Min confidence:"))
        self.ai_confidence_spin = QSpinBox()
        self.ai_confidence_spin.setRange(0, 100)
        self.ai_confidence_spin.setValue(30)
        self.ai_confidence_spin.setSuffix("%")
        self.ai_confidence_spin.setMinimumWidth(70)
        conf_layout.addWidget(self.ai_confidence_spin)
        conf_layout.addStretch()
        ai_layout.addLayout(conf_layout)

        # Signal filter
        ai_signal_layout = QHBoxLayout()
        ai_signal_layout.addWidget(QLabel("Show:"))
        self.ai_signal_combo = QComboBox()
        self.ai_signal_combo.addItem("All Signals", "all")
        self.ai_signal_combo.addItem("BUY Only", "buy")
        self.ai_signal_combo.addItem("SELL Only", "sell")
        ai_signal_layout.addWidget(self.ai_signal_combo)
        ai_layout.addLayout(ai_signal_layout)

        # AI Scan button
        self.ai_scan_btn = QPushButton("Scan with AI Models")
        self.ai_scan_btn.setMinimumHeight(45)
        self.ai_scan_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                font-size: 13px;
                font-weight: bold;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['success_hover']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_secondary']};
            }}
        """)
        self.ai_scan_btn.clicked.connect(self._on_ai_scan_click)
        ai_layout.addWidget(self.ai_scan_btn)

        layout.addWidget(ai_group)

        # Spacer
        layout.addStretch()

        scroll.setWidget(panel)
        outer_layout.addWidget(scroll)

        return outer

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

        # Calculate and display the time window
        timeframe_key = config['timeframe']
        past_candles = config['past_candles']
        time_window = self._calculate_time_window(timeframe_key, past_candles)

        self.log(f"Starting scan: {self.timeframe_combo.currentText()} timeframe, "
                f"last {past_candles} candles, {logic_mode.upper()} logic")
        self.log(f"Time window: signals from the last {time_window}")
        self.log(f"Strategies: {', '.join(selected_strategies)}")
        self.scan_requested.emit(config)

    def _calculate_time_window(self, timeframe: str, num_candles: int) -> str:
        """Calculate human-readable time window string."""
        # Timeframe to minutes mapping
        tf_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080,
        }

        minutes = tf_minutes.get(timeframe, 60) * num_candles

        if minutes < 60:
            return f"{minutes} minutes"
        elif minutes < 1440:
            hours = minutes / 60
            if hours == int(hours):
                return f"{int(hours)} hour{'s' if hours > 1 else ''}"
            return f"{hours:.1f} hours"
        else:
            days = minutes / 1440
            if days == int(days):
                return f"{int(days)} day{'s' if days > 1 else ''}"
            return f"{days:.1f} days"

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

            # Time - format the timestamp properly
            timestamp = result.get('timestamp', None)
            time_str = self._format_signal_time(timestamp)
            time_item = QTableWidgetItem(time_str)
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

    def _format_signal_time(self, timestamp) -> str:
        """
        Format the signal timestamp for display.

        Shows the candle time (converted to local) and how long ago it occurred.
        """
        if timestamp is None:
            return '--'

        try:
            # Convert to datetime if it's a pandas Timestamp
            if isinstance(timestamp, pd.Timestamp):
                # Timestamps from CCXT are in UTC - convert to local time
                if timestamp.tzinfo is not None:
                    # Has timezone - convert to local
                    dt_utc = timestamp.to_pydatetime()
                    dt = dt_utc.replace(tzinfo=None)  # Remove tz for local comparison
                else:
                    # No timezone - assume UTC, convert to local
                    dt = timestamp.to_pydatetime()
            elif isinstance(timestamp, datetime):
                dt = timestamp
            elif isinstance(timestamp, str):
                # Try to parse string timestamp
                try:
                    dt = pd.to_datetime(timestamp).to_pydatetime()
                except:
                    return timestamp[:19] if len(timestamp) > 19 else timestamp
            else:
                return str(timestamp)

            # Calculate time ago using UTC for comparison
            # Since CCXT returns UTC timestamps, use utcnow for accurate difference
            from datetime import timezone
            now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
            diff = now_utc - dt
            total_minutes = int(diff.total_seconds() / 60)

            # Handle negative or very small differences (signal from current candle)
            if total_minutes < 0:
                total_minutes = 0

            if total_minutes < 60:
                ago_str = f"{total_minutes}m ago"
            elif total_minutes < 1440:  # Less than 24 hours
                hours = total_minutes // 60
                ago_str = f"{hours}h ago"
            else:
                days = total_minutes // 1440
                ago_str = f"{days}d ago"

            # Format as "14:00 (1h ago)" - show UTC time
            time_str = dt.strftime('%H:%M')
            return f"{time_str} ({ago_str})"

        except Exception as e:
            return str(timestamp)[:16] if timestamp else '--'

    def _on_ai_scan_click(self):
        """Handle AI scan button click."""
        if self._ai_scanning:
            return

        timeframe = self.ai_timeframe_combo.currentData()
        confidence = self.ai_confidence_spin.value() / 100.0
        signal_filter = self.ai_signal_combo.currentData()

        config = {
            'timeframe': timeframe,
            'confidence_threshold': confidence,
            'signal_filter': signal_filter,
        }

        self.log(f"Starting AI scan: {self.ai_timeframe_combo.currentText()}, "
                 f"min confidence {self.ai_confidence_spin.value()}%, filter: {signal_filter}")

        self._ai_scanning = True
        self.ai_scan_btn.setEnabled(False)
        self.ai_scan_btn.setText("Scanning...")
        self.results_table.setRowCount(0)
        self._scan_results = []

        from src.gui_pyqt.workers import AISignalScannerWorker

        self._ai_scan_worker = AISignalScannerWorker(config)
        self._ai_scan_worker.progress.connect(self._on_ai_scan_progress)
        self._ai_scan_worker.result.connect(self._on_ai_scan_complete)
        self._ai_scan_worker.error.connect(self._on_ai_scan_error)
        self._ai_scan_worker.finished.connect(self._on_ai_scan_finished)
        self._ai_scan_worker.start()

    def _on_ai_scan_progress(self, current: int, total: int, message: str):
        """Handle AI scan progress."""
        if total > 0:
            pct = int((current / total) * 100)
            self.progress_bar.setValue(pct)
        self.status_label.setText(message)

    def _on_ai_scan_complete(self, results: list):
        """Handle AI scan results."""
        self._ai_scanning = False
        self._scan_results = results
        self.display_ai_results(results)

    def _on_ai_scan_error(self, error: str):
        """Handle AI scan error."""
        self.log(f"AI SCAN ERROR: {error}")
        self.status_label.setText(f"Error: {error}")

    def _on_ai_scan_finished(self):
        """Handle AI scan worker finished."""
        self._ai_scanning = False
        self.ai_scan_btn.setEnabled(True)
        self.ai_scan_btn.setText("Scan with AI Models")

    def display_ai_results(self, results: list):
        """Display AI scan results in the table."""
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
            if 'BUY' in signal:
                signal_item.setForeground(QColor(COLORS['chart_green']))
            elif 'SELL' in signal:
                signal_item.setForeground(QColor(COLORS['chart_red']))
            self.results_table.setItem(row, 1, signal_item)

            # Strategy column - show AI model info
            strategy_text = f"AI ({result.get('direction', '--')})"
            strategy_item = QTableWidgetItem(strategy_text)
            self.results_table.setItem(row, 2, strategy_item)

            # Confidence
            confidence = result.get('confidence', 0)
            conf_item = QTableWidgetItem(f"{confidence:.1%}")
            conf_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
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

            # Time column - show win rate and IC instead
            wr = result.get('win_rate', 0)
            ic = result.get('ic', 0)
            info_item = QTableWidgetItem(f"WR:{wr:.0%} IC:{ic:.3f}")
            info_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setItem(row, 6, info_item)

        # Update count
        buy_count = sum(1 for r in results if 'BUY' in r.get('signal', ''))
        sell_count = sum(1 for r in results if 'SELL' in r.get('signal', ''))
        self.results_count_label.setText(
            f"{len(results)} AI signals (BUY: {buy_count}, SELL: {sell_count})"
        )

        self.log(f"AI scan complete: {len(results)} signals found")
        self.log(f"  BUY signals: {buy_count}")
        self.log(f"  SELL signals: {sell_count}")

    def clear_results(self):
        """Clear all results"""
        self.results_table.setRowCount(0)
        self._scan_results = []
        self.results_count_label.setText("0 signals found")
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready to scan")
