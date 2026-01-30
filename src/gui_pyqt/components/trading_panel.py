"""
Mock Trading Panel - Book and track trades with real-time P&L

Features:
1. Book long/short trades with leverage
2. Set stop loss and take profit levels
3. Real-time price tracking and P&L calculation
4. Trade history with performance metrics
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QFrame, QGroupBox, QGridLayout, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSpinBox,
    QMessageBox, QSplitter, QCheckBox, QTextEdit, QLineEdit
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path

from src.gui_pyqt.styles import COLORS


@dataclass
class Trade:
    """Represents a single trade."""
    id: str
    symbol: str
    position_type: str  # 'LONG' or 'SHORT'
    entry_time: datetime
    entry_price: float
    amount: float  # In USD
    leverage: int
    trading_fee: float  # Percentage
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    status: str = 'OPEN'  # 'OPEN', 'CLOSED', 'STOPPED', 'TARGET'
    notes: str = ''
    signals: str = ''  # Signals that favored the trade
    strategy: str = ''  # Strategy used for the trade

    def calculate_pnl(self, current_price: float) -> tuple:
        """
        Calculate P&L for this trade.

        Returns:
            (pnl_dollars, pnl_percent, pnl_with_leverage)
        """
        if self.position_type == 'LONG':
            price_change_pct = (current_price - self.entry_price) / self.entry_price
        else:  # SHORT
            price_change_pct = (self.entry_price - current_price) / self.entry_price

        # P&L without leverage
        pnl_base = self.amount * price_change_pct

        # P&L with leverage
        pnl_leveraged = pnl_base * self.leverage

        # Subtract trading fees (entry + exit)
        total_fees = self.amount * (self.trading_fee / 100) * 2
        pnl_after_fees = pnl_leveraged - total_fees

        # Calculate percentage return on investment
        roi_pct = (pnl_after_fees / self.amount) * 100

        return pnl_after_fees, roi_pct, price_change_pct * 100

    def check_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss is triggered."""
        if self.stop_loss is None:
            return False
        if self.position_type == 'LONG':
            return current_price <= self.stop_loss
        else:  # SHORT
            return current_price >= self.stop_loss

    def check_take_profit(self, current_price: float) -> bool:
        """Check if take profit is triggered."""
        if self.take_profit is None:
            return False
        if self.position_type == 'LONG':
            return current_price >= self.take_profit
        else:  # SHORT
            return current_price <= self.take_profit

    def to_dict(self) -> dict:
        """Convert to dictionary for saving."""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'position_type': self.position_type,
            'entry_time': self.entry_time.isoformat(),
            'entry_price': self.entry_price,
            'amount': self.amount,
            'leverage': self.leverage,
            'trading_fee': self.trading_fee,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_price': self.exit_price,
            'status': self.status,
            'notes': self.notes,
            'signals': self.signals,
            'strategy': self.strategy,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Trade':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            symbol=data['symbol'],
            position_type=data['position_type'],
            entry_time=datetime.fromisoformat(data['entry_time']),
            entry_price=data['entry_price'],
            amount=data['amount'],
            leverage=data['leverage'],
            trading_fee=data['trading_fee'],
            stop_loss=data.get('stop_loss'),
            take_profit=data.get('take_profit'),
            exit_time=datetime.fromisoformat(data['exit_time']) if data.get('exit_time') else None,
            exit_price=data.get('exit_price'),
            status=data.get('status', 'OPEN'),
            notes=data.get('notes', ''),
            signals=data.get('signals', ''),
            strategy=data.get('strategy', ''),
        )


class TradeLogger:
    """
    Comprehensive trade logger for debugging and analysis.

    Logs all trade events to timestamped files with full details.
    """

    LOG_DIR = Path("data/trades/logs")

    def __init__(self):
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._current_log_file = None
        self._init_session_log()

    def _init_session_log(self):
        """Initialize a new session log file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._current_log_file = self.LOG_DIR / f"trade_session_{timestamp}.log"

        # Write session header
        with open(self._current_log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"MOCK TRADING SESSION LOG\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    def _write_log(self, message: str):
        """Write message to current log file."""
        if self._current_log_file:
            with open(self._current_log_file, 'a') as f:
                f.write(message + "\n")

    def log_trade_opened(self, trade: 'Trade'):
        """Log when a trade is opened."""
        sl_str = f"${trade.stop_loss:,.6f}" if trade.stop_loss else "Not Set"
        tp_str = f"${trade.take_profit:,.6f}" if trade.take_profit else "Not Set"
        strategy_str = trade.strategy if trade.strategy else "Manual Entry"
        signals_str = trade.signals if trade.signals else "No signals recorded"
        notes_str = trade.notes if trade.notes else "None"

        log_entry = f"""
{'=' * 80}
TRADE OPENED: {trade.id}
{'=' * 80}
Timestamp:      {trade.entry_time.strftime('%Y-%m-%d %H:%M:%S')}
Symbol:         {trade.symbol}
Position:       {trade.position_type}
Entry Price:    ${trade.entry_price:,.6f}
Amount:         ${trade.amount:,.2f}
Leverage:       {trade.leverage}x
Position Size:  ${trade.amount * trade.leverage:,.2f}
Trading Fee:    {trade.trading_fee}%
Stop Loss:      {sl_str}
Take Profit:    {tp_str}

SIGNALS & STRATEGY:
{'-' * 40}
Strategy:       {strategy_str}
Signals:        {signals_str}
Notes:          {notes_str}

RISK ANALYSIS:
{'-' * 40}"""

        # Calculate risk metrics
        if trade.stop_loss:
            if trade.position_type == 'LONG':
                sl_distance_pct = ((trade.entry_price - trade.stop_loss) / trade.entry_price) * 100
            else:
                sl_distance_pct = ((trade.stop_loss - trade.entry_price) / trade.entry_price) * 100
            max_loss = trade.amount * (sl_distance_pct / 100) * trade.leverage
            log_entry += f"\nSL Distance:    {sl_distance_pct:.2f}%"
            log_entry += f"\nMax Loss:       ${max_loss:,.2f}"

        if trade.take_profit:
            if trade.position_type == 'LONG':
                tp_distance_pct = ((trade.take_profit - trade.entry_price) / trade.entry_price) * 100
            else:
                tp_distance_pct = ((trade.entry_price - trade.take_profit) / trade.entry_price) * 100
            potential_profit = trade.amount * (tp_distance_pct / 100) * trade.leverage
            log_entry += f"\nTP Distance:    {tp_distance_pct:.2f}%"
            log_entry += f"\nMax Profit:     ${potential_profit:,.2f}"

            if trade.stop_loss:
                risk_reward = tp_distance_pct / sl_distance_pct if sl_distance_pct > 0 else 0
                log_entry += f"\nRisk/Reward:    1:{risk_reward:.2f}"

        log_entry += f"\n{'=' * 80}\n"

        self._write_log(log_entry)

        # Also write to individual trade log file
        self._write_individual_trade_log(trade, "OPENED")

    def log_trade_closed(self, trade: 'Trade', close_reason: str = "Manual"):
        """Log when a trade is closed."""
        pnl, roi, price_change = trade.calculate_pnl(trade.exit_price or trade.entry_price)
        duration = (trade.exit_time - trade.entry_time) if trade.exit_time else None

        close_time_str = trade.exit_time.strftime('%Y-%m-%d %H:%M:%S') if trade.exit_time else 'N/A'
        exit_price_str = f"${trade.exit_price:,.6f}" if trade.exit_price else 'N/A'
        duration_str = self._format_duration(duration) if duration else 'N/A'
        result_str = 'WIN' if pnl > 0 else ('LOSS' if pnl < 0 else 'BREAKEVEN')
        sl_str = f"${trade.stop_loss:,.6f}" if trade.stop_loss else "Not Set"
        tp_str = f"${trade.take_profit:,.6f}" if trade.take_profit else "Not Set"
        strategy_str = trade.strategy if trade.strategy else "Manual Entry"
        signals_str = trade.signals if trade.signals else "No signals recorded"

        log_entry = f"""
{'=' * 80}
TRADE CLOSED: {trade.id} ({close_reason})
{'=' * 80}
Close Time:     {close_time_str}
Symbol:         {trade.symbol}
Position:       {trade.position_type}

ENTRY:
{'-' * 40}
Entry Time:     {trade.entry_time.strftime('%Y-%m-%d %H:%M:%S')}
Entry Price:    ${trade.entry_price:,.6f}

EXIT:
{'-' * 40}
Exit Price:     {exit_price_str}
Price Change:   {price_change:+.2f}%

PERFORMANCE:
{'-' * 40}
P&L:            ${pnl:+,.2f}
ROI:            {roi:+.2f}%
Duration:       {duration_str}
Result:         {result_str}

TRADE DETAILS:
{'-' * 40}
Amount:         ${trade.amount:,.2f}
Leverage:       {trade.leverage}x
Stop Loss:      {sl_str}
Take Profit:    {tp_str}
Strategy:       {strategy_str}
Signals:        {signals_str}
{'=' * 80}
"""

        self._write_log(log_entry)
        self._write_individual_trade_log(trade, "CLOSED", pnl, roi)

    def log_sl_triggered(self, trade: 'Trade', trigger_price: float):
        """Log when stop loss is triggered."""
        pnl, roi, _ = trade.calculate_pnl(trigger_price)

        log_entry = f"""
{'!' * 80}
STOP LOSS TRIGGERED: {trade.id}
{'!' * 80}
Trigger Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Symbol:         {trade.symbol}
Position:       {trade.position_type}
Entry Price:    ${trade.entry_price:,.6f}
Stop Loss:      ${trade.stop_loss:,.6f}
Trigger Price:  ${trigger_price:,.6f}
P&L:            ${pnl:+,.2f}
ROI:            {roi:+.2f}%
{'!' * 80}
"""
        self._write_log(log_entry)

    def log_tp_triggered(self, trade: 'Trade', trigger_price: float):
        """Log when take profit is triggered."""
        pnl, roi, _ = trade.calculate_pnl(trigger_price)

        log_entry = f"""
{'*' * 80}
TAKE PROFIT TRIGGERED: {trade.id}
{'*' * 80}
Trigger Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Symbol:         {trade.symbol}
Position:       {trade.position_type}
Entry Price:    ${trade.entry_price:,.6f}
Take Profit:    ${trade.take_profit:,.6f}
Trigger Price:  ${trigger_price:,.6f}
P&L:            ${pnl:+,.2f}
ROI:            {roi:+.2f}%
{'*' * 80}
"""
        self._write_log(log_entry)

    def _write_individual_trade_log(self, trade: 'Trade', event: str,
                                     pnl: float = None, roi: float = None):
        """Write individual trade log file for detailed debugging."""
        trade_log_dir = self.LOG_DIR / "individual_trades"
        trade_log_dir.mkdir(parents=True, exist_ok=True)

        trade_file = trade_log_dir / f"{trade.id}_{trade.symbol}.log"

        with open(trade_file, 'a') as f:
            f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {event}\n")
            f.write(f"  Status: {trade.status}\n")
            if event == "CLOSED" and pnl is not None:
                f.write(f"  P&L: ${pnl:+,.2f} ({roi:+.2f}%)\n")
            f.write(f"  Entry: ${trade.entry_price:,.6f} @ {trade.entry_time}\n")
            if trade.exit_price:
                f.write(f"  Exit: ${trade.exit_price:,.6f} @ {trade.exit_time}\n")
            f.write(f"  Signals: {trade.signals}\n")
            f.write(f"  Strategy: {trade.strategy}\n")

    def _format_duration(self, duration) -> str:
        """Format duration for display."""
        total_seconds = int(duration.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 24:
            days = hours // 24
            hours = hours % 24
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def get_log_file_path(self) -> str:
        """Get current log file path."""
        return str(self._current_log_file) if self._current_log_file else ""


class TradingPanel(QWidget):
    """
    Mock Trading Panel for booking and tracking trades.
    """

    trade_booked = pyqtSignal(dict)  # Emitted when a trade is booked

    def __init__(self, parent=None):
        super().__init__(parent)
        self.trades: List[Trade] = []
        self.current_prices: Dict[str, float] = {}
        self._trade_counter = 0
        self._trade_logger = TradeLogger()
        self._setup_ui()
        self._load_trades()
        self._setup_price_timer()

    def _setup_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header
        header = QLabel("Mock Trading")
        header.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {COLORS['text_primary']};")
        layout.addWidget(header)

        subtitle = QLabel("Book trades and track real-time P&L to test strategies")
        subtitle.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(subtitle)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Top section - Trade booking
        booking_widget = self._create_booking_section()
        splitter.addWidget(booking_widget)

        # Bottom section - Trades table
        table_widget = self._create_trades_table()
        splitter.addWidget(table_widget)

        splitter.setSizes([300, 400])
        layout.addWidget(splitter)

        # Summary bar
        self._create_summary_bar(layout)

    def _create_booking_section(self) -> QWidget:
        """Create the trade booking section."""
        widget = QFrame()
        layout = QHBoxLayout(widget)

        # Left - Trade parameters
        params_group = QGroupBox("Book New Trade")
        params_layout = QGridLayout(params_group)

        row = 0

        # Symbol selection
        params_layout.addWidget(QLabel("Symbol:"), row, 0)
        self.symbol_combo = QComboBox()
        self.symbol_combo.setEditable(True)
        self.symbol_combo.setMinimumWidth(150)
        # Load symbols
        try:
            from src.gui_pyqt.utils.symbols import get_all_usdt_symbols
            self.symbol_combo.addItems(get_all_usdt_symbols())
        except:
            self.symbol_combo.addItems(['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'])
        self.symbol_combo.currentTextChanged.connect(self._on_symbol_changed)
        params_layout.addWidget(self.symbol_combo, row, 1)

        # Current price display
        params_layout.addWidget(QLabel("Current Price:"), row, 2)
        self.current_price_label = QLabel("--")
        self.current_price_label.setStyleSheet(f"font-weight: bold; color: {COLORS['text_primary']};")
        params_layout.addWidget(self.current_price_label, row, 3)

        row += 1

        # Position type
        params_layout.addWidget(QLabel("Position:"), row, 0)
        self.position_combo = QComboBox()
        self.position_combo.addItems(['LONG', 'SHORT'])
        self.position_combo.setStyleSheet(f"""
            QComboBox {{
                font-weight: bold;
            }}
        """)
        self.position_combo.currentTextChanged.connect(self._update_position_style)
        params_layout.addWidget(self.position_combo, row, 1)

        # Amount
        params_layout.addWidget(QLabel("Amount ($):"), row, 2)
        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(10, 1000000)
        self.amount_spin.setValue(100)
        self.amount_spin.setPrefix("$")
        self.amount_spin.setDecimals(2)
        params_layout.addWidget(self.amount_spin, row, 3)

        row += 1

        # Leverage
        params_layout.addWidget(QLabel("Leverage:"), row, 0)
        self.leverage_spin = QSpinBox()
        self.leverage_spin.setRange(1, 125)
        self.leverage_spin.setValue(1)
        self.leverage_spin.setSuffix("x")
        params_layout.addWidget(self.leverage_spin, row, 1)

        # Trading fee
        params_layout.addWidget(QLabel("Trading Fee:"), row, 2)
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setRange(0, 1)
        self.fee_spin.setValue(0.1)
        self.fee_spin.setSuffix("%")
        self.fee_spin.setDecimals(3)
        params_layout.addWidget(self.fee_spin, row, 3)

        row += 1

        # Stop Loss
        params_layout.addWidget(QLabel("Stop Loss:"), row, 0)
        sl_layout = QHBoxLayout()
        self.sl_enabled = QCheckBox()
        self.sl_enabled.toggled.connect(lambda x: self.sl_spin.setEnabled(x))
        sl_layout.addWidget(self.sl_enabled)
        self.sl_spin = QDoubleSpinBox()
        self.sl_spin.setRange(0, 10000000)
        self.sl_spin.setDecimals(4)
        self.sl_spin.setEnabled(False)
        self.sl_spin.setPrefix("$")
        sl_layout.addWidget(self.sl_spin)
        params_layout.addLayout(sl_layout, row, 1)

        # Take Profit
        params_layout.addWidget(QLabel("Take Profit:"), row, 2)
        tp_layout = QHBoxLayout()
        self.tp_enabled = QCheckBox()
        self.tp_enabled.toggled.connect(lambda x: self.tp_spin.setEnabled(x))
        tp_layout.addWidget(self.tp_enabled)
        self.tp_spin = QDoubleSpinBox()
        self.tp_spin.setRange(0, 10000000)
        self.tp_spin.setDecimals(4)
        self.tp_spin.setEnabled(False)
        self.tp_spin.setPrefix("$")
        tp_layout.addWidget(self.tp_spin)
        params_layout.addLayout(tp_layout, row, 3)

        row += 1

        # Quick SL/TP buttons
        params_layout.addWidget(QLabel("Quick SL:"), row, 0)
        sl_btn_layout = QHBoxLayout()
        for pct in [1, 2, 5, 10]:
            btn = QPushButton(f"-{pct}%")
            btn.setFixedWidth(45)
            btn.clicked.connect(lambda _, p=pct: self._set_quick_sl(p))
            sl_btn_layout.addWidget(btn)
        params_layout.addLayout(sl_btn_layout, row, 1)

        params_layout.addWidget(QLabel("Quick TP:"), row, 2)
        tp_btn_layout = QHBoxLayout()
        for pct in [2, 5, 10, 20]:
            btn = QPushButton(f"+{pct}%")
            btn.setFixedWidth(45)
            btn.clicked.connect(lambda _, p=pct: self._set_quick_tp(p))
            tp_btn_layout.addWidget(btn)
        params_layout.addLayout(tp_btn_layout, row, 3)

        row += 1

        # Strategy selection
        params_layout.addWidget(QLabel("Strategy:"), row, 0)
        self.strategy_combo = QComboBox()
        self.strategy_combo.setEditable(True)
        self.strategy_combo.addItems([
            'Manual Entry',
            'RSI + MACD',
            'Bollinger + RSI',
            'MACD + MA',
            'Stochastic + RSI',
            'Triple EMA',
            'ADX + MACD',
            'Stochastic',
            'Prophet AI',
            'Ensemble',
        ])
        params_layout.addWidget(self.strategy_combo, row, 1)

        # Signals input
        params_layout.addWidget(QLabel("Signals:"), row, 2)
        self.signals_input = QLineEdit()
        self.signals_input.setPlaceholderText("e.g., RSI oversold, MACD crossover")
        params_layout.addWidget(self.signals_input, row, 3)

        row += 1

        # Notes input (spans full row)
        params_layout.addWidget(QLabel("Notes:"), row, 0)
        self.notes_input = QLineEdit()
        self.notes_input.setPlaceholderText("Additional notes for this trade...")
        notes_layout = QHBoxLayout()
        notes_layout.addWidget(self.notes_input)
        params_layout.addLayout(notes_layout, row, 1, 1, 3)  # Span 3 columns

        layout.addWidget(params_group)

        # Right - Book button and info
        action_group = QGroupBox("Action")
        action_layout = QVBoxLayout(action_group)

        # Estimated info
        self.est_info_label = QLabel("Select a symbol to see estimated trade info")
        self.est_info_label.setWordWrap(True)
        self.est_info_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        action_layout.addWidget(self.est_info_label)

        action_layout.addStretch()

        # Book trade button
        self.book_btn = QPushButton("BOOK TRADE")
        self.book_btn.setMinimumHeight(50)
        self.book_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent']};
            }}
        """)
        self.book_btn.clicked.connect(self._book_trade)
        action_layout.addWidget(self.book_btn)

        # Refresh prices button
        refresh_btn = QPushButton("Refresh Prices")
        refresh_btn.clicked.connect(self._refresh_all_prices)
        action_layout.addWidget(refresh_btn)

        layout.addWidget(action_group)

        return widget

    def _create_trades_table(self) -> QWidget:
        """Create the trades table section."""
        widget = QFrame()
        layout = QVBoxLayout(widget)

        # Table header
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Open Trades"))

        # Filter buttons
        self.show_open_btn = QPushButton("Open")
        self.show_open_btn.setCheckable(True)
        self.show_open_btn.setChecked(True)
        self.show_open_btn.clicked.connect(self._filter_trades)
        header_layout.addWidget(self.show_open_btn)

        self.show_closed_btn = QPushButton("Closed")
        self.show_closed_btn.setCheckable(True)
        self.show_closed_btn.setChecked(True)
        self.show_closed_btn.clicked.connect(self._filter_trades)
        header_layout.addWidget(self.show_closed_btn)

        header_layout.addStretch()

        # Close selected button
        self.close_btn = QPushButton("Close Selected Trade")
        self.close_btn.clicked.connect(self._close_selected_trade)
        header_layout.addWidget(self.close_btn)

        # Delete selected button
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setStyleSheet(f"background-color: {COLORS['chart_red']};")
        self.delete_btn.clicked.connect(self._delete_selected_trade)
        header_layout.addWidget(self.delete_btn)

        layout.addLayout(header_layout)

        # Trades table
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(12)
        self.trades_table.setHorizontalHeaderLabels([
            'ID', 'Symbol', 'Type', 'Entry Time', 'Entry Price',
            'Current Price', 'Amount', 'Leverage', 'P&L ($)', 'P&L (%)',
            'SL/TP', 'Status'
        ])

        self.trades_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_primary']};
                gridline-color: {COLORS['border']};
            }}
            QTableWidget::item {{
                padding: 5px;
            }}
            QHeaderView::section {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_primary']};
                padding: 5px;
                border: 1px solid {COLORS['border']};
            }}
        """)

        header = self.trades_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        for i in range(4, 12):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)

        self.trades_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.trades_table)

        return widget

    def _create_summary_bar(self, parent_layout):
        """Create the summary statistics bar."""
        summary_frame = QFrame()
        summary_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_medium']};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        summary_layout = QHBoxLayout(summary_frame)

        # Total P&L
        summary_layout.addWidget(QLabel("Total P&L:"))
        self.total_pnl_label = QLabel("$0.00")
        self.total_pnl_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        summary_layout.addWidget(self.total_pnl_label)

        summary_layout.addWidget(QLabel("|"))

        # Open trades
        summary_layout.addWidget(QLabel("Open Trades:"))
        self.open_trades_label = QLabel("0")
        self.open_trades_label.setStyleSheet("font-weight: bold;")
        summary_layout.addWidget(self.open_trades_label)

        summary_layout.addWidget(QLabel("|"))

        # Win rate
        summary_layout.addWidget(QLabel("Win Rate:"))
        self.win_rate_label = QLabel("--")
        self.win_rate_label.setStyleSheet("font-weight: bold;")
        summary_layout.addWidget(self.win_rate_label)

        summary_layout.addWidget(QLabel("|"))

        # Total trades
        summary_layout.addWidget(QLabel("Total Trades:"))
        self.total_trades_label = QLabel("0")
        summary_layout.addWidget(self.total_trades_label)

        summary_layout.addStretch()

        # View logs button
        logs_btn = QPushButton("View Logs")
        logs_btn.clicked.connect(self._open_logs_folder)
        summary_layout.addWidget(logs_btn)

        # Export button
        export_btn = QPushButton("Export to Excel")
        export_btn.clicked.connect(self.export_to_excel)
        summary_layout.addWidget(export_btn)

        parent_layout.addWidget(summary_frame)

    def _setup_price_timer(self):
        """Setup timer for price updates."""
        self.price_timer = QTimer()
        self.price_timer.timeout.connect(self._update_prices)
        self.price_timer.start(5000)  # Update every 5 seconds

    def _on_symbol_changed(self, symbol: str):
        """Handle symbol selection change."""
        if symbol:
            self._fetch_current_price(symbol)
            self._update_sl_tp_defaults()

    def _fetch_current_price(self, symbol: str):
        """Fetch current price for symbol."""
        try:
            from src.api.ccxt_client import get_ccxt_client
            client = get_ccxt_client('binance')
            ccxt_symbol = symbol.replace('USDT', '/USDT')
            ticker = client.exchange.fetch_ticker(ccxt_symbol)
            price = ticker['last']
            self.current_prices[symbol] = price
            self.current_price_label.setText(f"${price:,.4f}")
            self._update_sl_tp_defaults()
            self._update_est_info()
        except Exception as e:
            self.current_price_label.setText("Error")
            print(f"Error fetching price for {symbol}: {e}")

    def _update_sl_tp_defaults(self):
        """Update SL/TP spinbox defaults based on current price."""
        symbol = self.symbol_combo.currentText()
        if symbol in self.current_prices:
            price = self.current_prices[symbol]
            # Set reasonable defaults
            self.sl_spin.setValue(price * 0.95)  # 5% below
            self.tp_spin.setValue(price * 1.10)  # 10% above

    def _set_quick_sl(self, pct: int):
        """Set stop loss at percentage below current price."""
        symbol = self.symbol_combo.currentText()
        if symbol in self.current_prices:
            price = self.current_prices[symbol]
            position = self.position_combo.currentText()
            if position == 'LONG':
                sl_price = price * (1 - pct / 100)
            else:
                sl_price = price * (1 + pct / 100)
            self.sl_spin.setValue(sl_price)
            self.sl_enabled.setChecked(True)

    def _set_quick_tp(self, pct: int):
        """Set take profit at percentage above current price."""
        symbol = self.symbol_combo.currentText()
        if symbol in self.current_prices:
            price = self.current_prices[symbol]
            position = self.position_combo.currentText()
            if position == 'LONG':
                tp_price = price * (1 + pct / 100)
            else:
                tp_price = price * (1 - pct / 100)
            self.tp_spin.setValue(tp_price)
            self.tp_enabled.setChecked(True)

    def _update_position_style(self, position: str):
        """Update position combo style based on selection."""
        if position == 'LONG':
            self.position_combo.setStyleSheet(f"""
                QComboBox {{
                    font-weight: bold;
                    color: {COLORS['chart_green']};
                }}
            """)
        else:
            self.position_combo.setStyleSheet(f"""
                QComboBox {{
                    font-weight: bold;
                    color: {COLORS['chart_red']};
                }}
            """)

    def _update_est_info(self):
        """Update estimated trade information."""
        symbol = self.symbol_combo.currentText()
        if symbol not in self.current_prices:
            return

        price = self.current_prices[symbol]
        amount = self.amount_spin.value()
        leverage = self.leverage_spin.value()
        fee = self.fee_spin.value()

        position_size = amount * leverage
        fee_cost = amount * (fee / 100) * 2  # Entry + exit

        info = f"""
        <b>Position Size:</b> ${position_size:,.2f}<br>
        <b>Entry Fee:</b> ${amount * (fee/100):,.2f}<br>
        <b>Total Fees (round-trip):</b> ${fee_cost:,.2f}<br>
        <b>Liquidation Price:</b> ~${price * (1 - 1/leverage) if leverage > 1 else 0:,.2f}
        """
        self.est_info_label.setText(info)

    def _book_trade(self):
        """Book a new trade."""
        symbol = self.symbol_combo.currentText()
        if not symbol:
            QMessageBox.warning(self, "Error", "Please select a symbol")
            return

        if symbol not in self.current_prices:
            self._fetch_current_price(symbol)
            if symbol not in self.current_prices:
                QMessageBox.warning(self, "Error", "Could not fetch current price")
                return

        self._trade_counter += 1
        trade_id = f"T{self._trade_counter:04d}"

        trade = Trade(
            id=trade_id,
            symbol=symbol,
            position_type=self.position_combo.currentText(),
            entry_time=datetime.now(),
            entry_price=self.current_prices[symbol],
            amount=self.amount_spin.value(),
            leverage=self.leverage_spin.value(),
            trading_fee=self.fee_spin.value(),
            stop_loss=self.sl_spin.value() if self.sl_enabled.isChecked() else None,
            take_profit=self.tp_spin.value() if self.tp_enabled.isChecked() else None,
            signals=self.signals_input.text().strip(),
            strategy=self.strategy_combo.currentText(),
            notes=self.notes_input.text().strip(),
        )

        self.trades.append(trade)
        self._save_trades()
        self._refresh_table()
        self._update_summary()

        # Log the trade
        self._trade_logger.log_trade_opened(trade)

        # Clear input fields after booking
        self.signals_input.clear()
        self.notes_input.clear()
        self.strategy_combo.setCurrentIndex(0)

        # Show confirmation
        msg = f"Trade {trade_id} booked!\n\n"
        msg += f"Symbol: {symbol}\n"
        msg += f"Position: {trade.position_type}\n"
        msg += f"Entry: ${trade.entry_price:,.4f}\n"
        msg += f"Amount: ${trade.amount:,.2f}\n"
        msg += f"Leverage: {trade.leverage}x"
        if trade.signals:
            msg += f"\nSignals: {trade.signals}"

        QMessageBox.information(self, "Trade Booked", msg)

        self.trade_booked.emit(trade.to_dict())

    def _close_selected_trade(self):
        """Close the selected trade."""
        selected = self.trades_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Error", "Please select a trade to close")
            return

        row = selected[0].row()
        trade_id = self.trades_table.item(row, 0).text()

        # Find the trade
        trade = next((t for t in self.trades if t.id == trade_id), None)
        if not trade:
            return

        if trade.status != 'OPEN':
            QMessageBox.warning(self, "Error", "Trade is already closed")
            return

        # Get current price
        if trade.symbol in self.current_prices:
            current_price = self.current_prices[trade.symbol]
        else:
            self._fetch_current_price(trade.symbol)
            current_price = self.current_prices.get(trade.symbol, trade.entry_price)

        # Close the trade
        trade.exit_time = datetime.now()
        trade.exit_price = current_price
        trade.status = 'CLOSED'

        pnl, roi, _ = trade.calculate_pnl(current_price)

        # Log the trade closure
        self._trade_logger.log_trade_closed(trade, "Manual Close")

        self._save_trades()
        self._refresh_table()
        self._update_summary()

        QMessageBox.information(
            self, "Trade Closed",
            f"Trade {trade_id} closed!\n\n"
            f"Exit Price: ${current_price:,.4f}\n"
            f"P&L: ${pnl:,.2f} ({roi:+.2f}%)"
        )

    def _delete_selected_trade(self):
        """Delete the selected trade."""
        selected = self.trades_table.selectedItems()
        if not selected:
            return

        row = selected[0].row()
        trade_id = self.trades_table.item(row, 0).text()

        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete trade {trade_id}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.trades = [t for t in self.trades if t.id != trade_id]
            self._save_trades()
            self._refresh_table()
            self._update_summary()

    def _update_prices(self):
        """Update prices for all open trades."""
        open_symbols = set(t.symbol for t in self.trades if t.status == 'OPEN')

        for symbol in open_symbols:
            try:
                self._fetch_current_price(symbol)
            except:
                pass

        # Check for stop loss / take profit triggers
        for trade in self.trades:
            if trade.status != 'OPEN':
                continue

            current_price = self.current_prices.get(trade.symbol)
            if not current_price:
                continue

            if trade.check_stop_loss(current_price):
                trade.exit_time = datetime.now()
                trade.exit_price = current_price
                trade.status = 'STOPPED'
                # Log SL trigger and trade closure
                self._trade_logger.log_sl_triggered(trade, current_price)
                self._trade_logger.log_trade_closed(trade, "Stop Loss")
                self._show_trade_alert(trade, "Stop Loss Triggered!")

            elif trade.check_take_profit(current_price):
                trade.exit_time = datetime.now()
                trade.exit_price = current_price
                trade.status = 'TARGET'
                # Log TP trigger and trade closure
                self._trade_logger.log_tp_triggered(trade, current_price)
                self._trade_logger.log_trade_closed(trade, "Take Profit")
                self._show_trade_alert(trade, "Take Profit Reached!")

        self._save_trades()
        self._refresh_table()
        self._update_summary()

    def _show_trade_alert(self, trade: Trade, message: str):
        """Show alert for trade events."""
        pnl, roi, _ = trade.calculate_pnl(trade.exit_price)
        QMessageBox.information(
            self, message,
            f"Trade {trade.id} - {trade.symbol}\n\n"
            f"{message}\n"
            f"Exit Price: ${trade.exit_price:,.4f}\n"
            f"P&L: ${pnl:,.2f} ({roi:+.2f}%)"
        )

    def _refresh_all_prices(self):
        """Manually refresh all prices."""
        # Get current symbol
        symbol = self.symbol_combo.currentText()
        if symbol:
            self._fetch_current_price(symbol)

        # Update all open trades
        self._update_prices()

    def _filter_trades(self):
        """Filter trades table based on status."""
        self._refresh_table()

    def _refresh_table(self):
        """Refresh the trades table."""
        show_open = self.show_open_btn.isChecked()
        show_closed = self.show_closed_btn.isChecked()

        # Filter trades
        filtered = []
        for t in self.trades:
            if t.status == 'OPEN' and show_open:
                filtered.append(t)
            elif t.status != 'OPEN' and show_closed:
                filtered.append(t)

        # Sort by entry time (newest first)
        filtered.sort(key=lambda x: x.entry_time, reverse=True)

        self.trades_table.setRowCount(len(filtered))

        for row, trade in enumerate(filtered):
            current_price = self.current_prices.get(trade.symbol, trade.entry_price)
            if trade.status != 'OPEN' and trade.exit_price:
                current_price = trade.exit_price

            pnl, roi, price_change = trade.calculate_pnl(current_price)

            # ID
            self.trades_table.setItem(row, 0, QTableWidgetItem(trade.id))

            # Symbol
            self.trades_table.setItem(row, 1, QTableWidgetItem(trade.symbol))

            # Type
            type_item = QTableWidgetItem(trade.position_type)
            type_item.setForeground(
                QColor(COLORS['chart_green'] if trade.position_type == 'LONG' else COLORS['chart_red'])
            )
            self.trades_table.setItem(row, 2, type_item)

            # Entry time
            self.trades_table.setItem(row, 3, QTableWidgetItem(
                trade.entry_time.strftime('%Y-%m-%d %H:%M')
            ))

            # Entry price
            self.trades_table.setItem(row, 4, QTableWidgetItem(f"${trade.entry_price:,.4f}"))

            # Current price
            price_item = QTableWidgetItem(f"${current_price:,.4f}")
            self.trades_table.setItem(row, 5, price_item)

            # Amount
            self.trades_table.setItem(row, 6, QTableWidgetItem(f"${trade.amount:,.0f}"))

            # Leverage
            self.trades_table.setItem(row, 7, QTableWidgetItem(f"{trade.leverage}x"))

            # P&L ($)
            pnl_item = QTableWidgetItem(f"${pnl:+,.2f}")
            pnl_item.setForeground(
                QColor(COLORS['chart_green'] if pnl >= 0 else COLORS['chart_red'])
            )
            self.trades_table.setItem(row, 8, pnl_item)

            # P&L (%)
            roi_item = QTableWidgetItem(f"{roi:+.2f}%")
            roi_item.setForeground(
                QColor(COLORS['chart_green'] if roi >= 0 else COLORS['chart_red'])
            )
            self.trades_table.setItem(row, 9, roi_item)

            # SL/TP
            sl_tp = ""
            if trade.stop_loss:
                sl_tp += f"SL: ${trade.stop_loss:,.2f}"
            if trade.take_profit:
                if sl_tp:
                    sl_tp += " | "
                sl_tp += f"TP: ${trade.take_profit:,.2f}"
            self.trades_table.setItem(row, 10, QTableWidgetItem(sl_tp if sl_tp else "--"))

            # Status
            status_item = QTableWidgetItem(trade.status)
            status_colors = {
                'OPEN': COLORS['primary'],
                'CLOSED': COLORS['text_secondary'],
                'STOPPED': COLORS['chart_red'],
                'TARGET': COLORS['chart_green'],
            }
            status_item.setForeground(QColor(status_colors.get(trade.status, COLORS['text_primary'])))
            self.trades_table.setItem(row, 11, status_item)

    def _update_summary(self):
        """Update summary statistics."""
        open_trades = [t for t in self.trades if t.status == 'OPEN']
        closed_trades = [t for t in self.trades if t.status != 'OPEN']

        # Calculate total P&L
        total_pnl = 0
        for trade in self.trades:
            if trade.status == 'OPEN':
                current_price = self.current_prices.get(trade.symbol, trade.entry_price)
            else:
                current_price = trade.exit_price or trade.entry_price
            pnl, _, _ = trade.calculate_pnl(current_price)
            total_pnl += pnl

        # Update labels
        pnl_color = COLORS['chart_green'] if total_pnl >= 0 else COLORS['chart_red']
        self.total_pnl_label.setText(f"${total_pnl:+,.2f}")
        self.total_pnl_label.setStyleSheet(f"font-weight: bold; font-size: 14px; color: {pnl_color};")

        self.open_trades_label.setText(str(len(open_trades)))
        self.total_trades_label.setText(str(len(self.trades)))

        # Win rate
        if closed_trades:
            wins = sum(1 for t in closed_trades
                      if t.calculate_pnl(t.exit_price or t.entry_price)[0] > 0)
            win_rate = wins / len(closed_trades) * 100
            self.win_rate_label.setText(f"{win_rate:.1f}%")
            wr_color = COLORS['chart_green'] if win_rate >= 50 else COLORS['chart_red']
            self.win_rate_label.setStyleSheet(f"font-weight: bold; color: {wr_color};")
        else:
            self.win_rate_label.setText("--")

    def _save_trades(self):
        """Save trades to file."""
        save_dir = Path("data/trades")
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / "mock_trades.json"

        data = {
            'trade_counter': self._trade_counter,
            'trades': [t.to_dict() for t in self.trades]
        }

        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_trades(self):
        """Load trades from file."""
        save_path = Path("data/trades/mock_trades.json")

        if save_path.exists():
            try:
                with open(save_path, 'r') as f:
                    data = json.load(f)

                self._trade_counter = data.get('trade_counter', 0)
                self.trades = [Trade.from_dict(t) for t in data.get('trades', [])]

                self._refresh_table()
                self._update_summary()
            except Exception as e:
                print(f"Error loading trades: {e}")

    def _open_logs_folder(self):
        """Open the logs folder in file explorer."""
        import subprocess
        import platform

        logs_dir = Path("data/trades/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)

        try:
            if platform.system() == 'Windows':
                subprocess.run(['explorer', str(logs_dir.absolute())])
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', str(logs_dir.absolute())])
            else:  # Linux
                subprocess.run(['xdg-open', str(logs_dir.absolute())])

            # Show log file info
            log_path = self._trade_logger.get_log_file_path()
            if log_path:
                QMessageBox.information(
                    self, "Trade Logs",
                    f"Logs folder opened.\n\n"
                    f"Current session log:\n{log_path}\n\n"
                    f"Individual trade logs are in:\n{logs_dir / 'individual_trades'}"
                )
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open logs folder: {e}")

    def export_to_excel(self):
        """Export all trades to Excel file."""
        if not self.trades:
            QMessageBox.warning(self, "No Trades", "No trades to export")
            return

        try:
            # Prepare data for export
            export_data = []
            for trade in self.trades:
                if trade.status == 'OPEN':
                    current_price = self.current_prices.get(trade.symbol, trade.entry_price)
                    exit_price = None
                    exit_time = None
                else:
                    current_price = trade.exit_price or trade.entry_price
                    exit_price = trade.exit_price
                    exit_time = trade.exit_time

                pnl, roi, price_change = trade.calculate_pnl(current_price)

                export_data.append({
                    'Trade ID': trade.id,
                    'Symbol': trade.symbol,
                    'Position': trade.position_type,
                    'Entry Time': trade.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'Entry Price': trade.entry_price,
                    'Exit Time': exit_time.strftime('%Y-%m-%d %H:%M:%S') if exit_time else '',
                    'Exit Price': exit_price if exit_price else '',
                    'Current Price': current_price,
                    'Amount ($)': trade.amount,
                    'Leverage': trade.leverage,
                    'Trading Fee (%)': trade.trading_fee,
                    'Stop Loss': trade.stop_loss if trade.stop_loss else '',
                    'Take Profit': trade.take_profit if trade.take_profit else '',
                    'P&L ($)': round(pnl, 2),
                    'P&L (%)': round(roi, 2),
                    'Price Change (%)': round(price_change, 2),
                    'Status': trade.status,
                    'Notes': trade.notes,
                })

            # Create DataFrame and save to Excel
            df = pd.DataFrame(export_data)

            # Create export directory
            export_dir = Path("data/trades/exports")
            export_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            excel_path = export_dir / f"trades_export_{timestamp}.xlsx"

            # Also save a "latest" version for easy reference
            latest_path = export_dir / "trades_latest.xlsx"

            # Save with formatting
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Trades', index=False)

                # Add summary sheet
                summary_data = self._calculate_summary_stats()
                summary_df = pd.DataFrame([summary_data])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Save latest version
            with pd.ExcelWriter(latest_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Trades', index=False)
                summary_df = pd.DataFrame([summary_data])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

            QMessageBox.information(
                self, "Export Complete",
                f"Trades exported to:\n{excel_path}\n\nLatest version also saved to:\n{latest_path}"
            )

            return str(excel_path)

        except ImportError:
            QMessageBox.warning(
                self, "Missing Package",
                "openpyxl is required for Excel export.\nInstall with: pip install openpyxl"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")

    def _calculate_summary_stats(self) -> dict:
        """Calculate summary statistics for export."""
        open_trades = [t for t in self.trades if t.status == 'OPEN']
        closed_trades = [t for t in self.trades if t.status != 'OPEN']

        total_pnl = 0
        total_invested = 0

        for trade in self.trades:
            total_invested += trade.amount
            if trade.status == 'OPEN':
                current_price = self.current_prices.get(trade.symbol, trade.entry_price)
            else:
                current_price = trade.exit_price or trade.entry_price
            pnl, _, _ = trade.calculate_pnl(current_price)
            total_pnl += pnl

        wins = 0
        losses = 0
        for t in closed_trades:
            pnl, _, _ = t.calculate_pnl(t.exit_price or t.entry_price)
            if pnl > 0:
                wins += 1
            else:
                losses += 1

        return {
            'Report Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Total Trades': len(self.trades),
            'Open Trades': len(open_trades),
            'Closed Trades': len(closed_trades),
            'Winning Trades': wins,
            'Losing Trades': losses,
            'Win Rate (%)': round(wins / len(closed_trades) * 100, 2) if closed_trades else 0,
            'Total P&L ($)': round(total_pnl, 2),
            'Total Invested ($)': round(total_invested, 2),
            'Overall ROI (%)': round(total_pnl / total_invested * 100, 2) if total_invested > 0 else 0,
            'Stopped Out': sum(1 for t in self.trades if t.status == 'STOPPED'),
            'Target Hit': sum(1 for t in self.trades if t.status == 'TARGET'),
        }
