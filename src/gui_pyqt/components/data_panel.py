"""
Data Panel Component - Historical data fetching interface

Provides UI for downloading historical cryptocurrency data with progress tracking.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QProgressBar, QTextEdit, QGroupBox, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from typing import Dict, Optional
from src.gui_pyqt.styles import COLORS


class DataPanel(QWidget):
    """
    Data Fetching Panel

    Provides controls for downloading historical OHLCV data.

    Signals:
        fetch_requested(dict, int): (symbol_data, max_candles) when fetch requested
    """

    fetch_requested = pyqtSignal(dict, int)

    CANDLE_OPTIONS = [
        ("1,000 candles (~6 weeks)", 1000),
        ("5,000 candles (~7 months)", 5000),
        ("10,000 candles (~14 months)", 10000),
        ("25,000 candles (~3 years)", 25000),
        ("50,000 candles (~6 years)", 50000),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_symbol = None
        self._is_fetching = False
        self._setup_ui()

    def _setup_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Header
        header = QLabel("Data Pipeline")
        header.setProperty("class", "title")
        layout.addWidget(header)

        # Symbol display
        symbol_group = QGroupBox("Selected Symbol")
        symbol_layout = QVBoxLayout(symbol_group)

        self.symbol_label = QLabel("No symbol selected")
        self.symbol_label.setProperty("class", "header")
        self.symbol_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        symbol_layout.addWidget(self.symbol_label)

        self.symbol_info = QLabel("")
        self.symbol_info.setProperty("class", "secondary")
        self.symbol_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        symbol_layout.addWidget(self.symbol_info)

        layout.addWidget(symbol_group)

        # Configuration
        config_group = QGroupBox("Download Configuration")
        config_layout = QVBoxLayout(config_group)

        # Candle count selector
        count_layout = QHBoxLayout()
        count_layout.addWidget(QLabel("Data Amount:"))

        self.candle_combo = QComboBox()
        for label, _ in self.CANDLE_OPTIONS:
            self.candle_combo.addItem(label)
        self.candle_combo.setCurrentIndex(2)  # Default to 10,000
        count_layout.addWidget(self.candle_combo, stretch=1)

        config_layout.addLayout(count_layout)

        # Fetch button
        self.fetch_btn = QPushButton("Start Fetching Data")
        self.fetch_btn.setEnabled(False)
        self.fetch_btn.clicked.connect(self._on_fetch_click)
        config_layout.addWidget(self.fetch_btn)

        layout.addWidget(config_group)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setProperty("class", "secondary")
        progress_layout.addWidget(self.status_label)

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(200)
        progress_layout.addWidget(self.log_output)

        layout.addWidget(progress_group)

        # Spacer
        layout.addStretch()

    def set_symbol(self, symbol_data: Dict):
        """
        Set the current symbol for fetching.

        Args:
            symbol_data: Dict with symbol info
        """
        self._current_symbol = symbol_data
        symbol = symbol_data.get('symbol', 'Unknown')
        price = symbol_data.get('price', 0)
        change = symbol_data.get('price_change_pct', 0)

        self.symbol_label.setText(symbol)

        change_color = COLORS['chart_green'] if change >= 0 else COLORS['chart_red']
        self.symbol_info.setText(
            f"Price: ${price:,.2f} | Change: "
            f"<span style='color:{change_color}'>{change:+.2f}%</span>"
        )

        self.fetch_btn.setEnabled(True)
        self.log_output.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready to fetch")

    def _on_fetch_click(self):
        """Handle fetch button click"""
        if self._current_symbol and not self._is_fetching:
            idx = self.candle_combo.currentIndex()
            _, max_candles = self.CANDLE_OPTIONS[idx]
            self.fetch_requested.emit(self._current_symbol, max_candles)

    def update_progress(self, current: int, total: int, message: str):
        """
        Update progress display.

        Args:
            current: Current progress value
            total: Total expected value
            message: Status message
        """
        if total > 0:
            pct = int((current / total) * 100)
            self.progress_bar.setValue(pct)
        self.status_label.setText(message)

    def log(self, message: str):
        """Append message to log output"""
        self.log_output.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def set_fetching(self, is_fetching: bool):
        """Set fetching state"""
        self._is_fetching = is_fetching
        self.fetch_btn.setEnabled(not is_fetching and self._current_symbol is not None)
        self.candle_combo.setEnabled(not is_fetching)

        if is_fetching:
            self.fetch_btn.setText("Fetching...")
        else:
            self.fetch_btn.setText("Start Fetching Data")

    def complete_fetch(self, success: bool, message: str):
        """
        Handle fetch completion.

        Args:
            success: Whether fetch was successful
            message: Completion message
        """
        self.set_fetching(False)
        self.progress_bar.setValue(100 if success else 0)
        self.status_label.setText(message)
        self.log(f"\n{'[OK]' if success else '[ERROR]'} {message}")

    def clear(self):
        """Clear the panel"""
        self._current_symbol = None
        self.symbol_label.setText("No symbol selected")
        self.symbol_info.setText("")
        self.fetch_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")
        self.log_output.clear()
