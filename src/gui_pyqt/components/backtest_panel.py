"""
Backtest Panel Component - Strategy-based backtesting interface

Provides UI for selecting and backtesting trading strategies with results visualization.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QProgressBar, QGroupBox, QLineEdit, QDoubleSpinBox,
    QTextEdit, QSplitter, QFrame, QGridLayout, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from typing import Dict, Optional, List
from src.gui_pyqt.styles import COLORS, get_chart_colors


class BacktestPanel(QWidget):
    """
    Backtesting Panel with Strategy Selection

    Features:
    - Strategy dropdown selector
    - Symbol selection
    - Configuration parameters
    - Results visualization with equity curve

    Signals:
        backtest_requested(dict): Config dict when backtest requested
    """

    backtest_requested = pyqtSignal(dict)

    # Available strategies
    STRATEGIES = [
        ('rsi', 'RSI Oversold/Overbought'),
        ('macd', 'MACD Crossover'),
        ('bollinger', 'Bollinger Bands'),
        ('ma_crossover', 'MA Crossover'),
        ('stochastic', 'Stochastic'),
        ('ensemble', 'Ensemble (Multi-Strategy)'),
        ('prediction', 'AI Model Prediction'),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_running = False
        self._available_symbols = []
        self._setup_ui()

    def _setup_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header
        header = QLabel("Strategy Backtesting")
        header.setProperty("class", "title")
        layout.addWidget(header)

        # Main splitter - config on left, results on right
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side - Configuration
        config_widget = self._create_config_panel()
        splitter.addWidget(config_widget)

        # Right side - Results
        results_widget = self._create_results_panel()
        splitter.addWidget(results_widget)

        splitter.setSizes([400, 600])
        layout.addWidget(splitter, stretch=1)

    def _create_config_panel(self) -> QWidget:
        """Create configuration panel"""
        widget = QFrame()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 8, 0)

        # Strategy selection
        strategy_group = QGroupBox("Strategy Selection")
        strategy_layout = QVBoxLayout(strategy_group)

        # Strategy dropdown
        strat_row = QHBoxLayout()
        strat_row.addWidget(QLabel("Strategy:"))
        self.strategy_combo = QComboBox()
        for key, label in self.STRATEGIES:
            self.strategy_combo.addItem(label, key)
        self.strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        strat_row.addWidget(self.strategy_combo, stretch=1)
        strategy_layout.addLayout(strat_row)

        # Strategy description
        self.strategy_desc = QLabel()
        self.strategy_desc.setProperty("class", "secondary")
        self.strategy_desc.setWordWrap(True)
        strategy_layout.addWidget(self.strategy_desc)

        layout.addWidget(strategy_group)

        # Symbol selection
        symbol_group = QGroupBox("Data Selection")
        symbol_layout = QVBoxLayout(symbol_group)

        sym_row = QHBoxLayout()
        sym_row.addWidget(QLabel("Symbol:"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.setEditable(False)
        sym_row.addWidget(self.symbol_combo, stretch=1)
        symbol_layout.addLayout(sym_row)

        # Model path (for AI strategy)
        self.model_path_row = QHBoxLayout()
        self.model_path_row.addWidget(QLabel("Model:"))
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Select model file...")
        self.model_path_input.setReadOnly(True)
        self.model_path_row.addWidget(self.model_path_input, stretch=1)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setFixedWidth(60)
        self.model_path_row.addWidget(self.browse_btn)
        symbol_layout.addLayout(self.model_path_row)

        layout.addWidget(symbol_group)

        # Parameters
        params_group = QGroupBox("Parameters")
        params_layout = QGridLayout(params_group)

        # Initial capital
        params_layout.addWidget(QLabel("Initial Capital ($):"), 0, 0)
        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(100, 1000000)
        self.capital_spin.setValue(1000)
        self.capital_spin.setPrefix("$")
        params_layout.addWidget(self.capital_spin, 0, 1)

        # Trade fee
        params_layout.addWidget(QLabel("Trade Fee (%):"), 1, 0)
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setRange(0, 5)
        self.fee_spin.setValue(0.1)
        self.fee_spin.setSuffix("%")
        self.fee_spin.setDecimals(2)
        params_layout.addWidget(self.fee_spin, 1, 1)

        # Confidence threshold
        params_layout.addWidget(QLabel("Min Confidence:"), 2, 0)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0, 1)
        self.confidence_spin.setValue(0.5)
        self.confidence_spin.setSingleStep(0.1)
        self.confidence_spin.setDecimals(2)
        params_layout.addWidget(self.confidence_spin, 2, 1)

        layout.addWidget(params_group)

        # Show signals on chart option
        self.show_signals_cb = QCheckBox("Show signals on chart")
        self.show_signals_cb.setChecked(True)
        layout.addWidget(self.show_signals_cb)

        # Run button
        self.run_btn = QPushButton("Run Backtest")
        self.run_btn.clicked.connect(self._on_run_click)
        layout.addWidget(self.run_btn)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setProperty("class", "secondary")
        layout.addWidget(self.status_label)

        layout.addStretch()

        # Update strategy description
        self._on_strategy_changed(0)

        return widget

    def _create_results_panel(self) -> QWidget:
        """Create results panel"""
        widget = QFrame()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 0, 0, 0)

        # Metrics grid
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QGridLayout(metrics_group)

        # Create metric labels
        self.metric_labels = {}
        metrics = [
            ('initial', 'Initial Capital'),
            ('final', 'Final Capital'),
            ('return', 'Total Return'),
            ('return_pct', 'Return %'),
            ('trades', 'Total Trades'),
            ('win_rate', 'Win Rate'),
            ('pl_ratio', 'P/L Ratio'),
            ('buy_hold', 'Buy & Hold'),
            ('excess', 'Excess Return'),
            ('dir_acc', 'Direction Acc'),
        ]

        for i, (key, label) in enumerate(metrics):
            row = i // 2
            col = (i % 2) * 2

            lbl = QLabel(f"{label}:")
            lbl.setProperty("class", "secondary")
            metrics_layout.addWidget(lbl, row, col)

            val_lbl = QLabel("-")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.metric_labels[key] = val_lbl
            metrics_layout.addWidget(val_lbl, row, col + 1)

        layout.addWidget(metrics_group)

        # Chart
        colors = get_chart_colors()
        self.figure = Figure(figsize=(8, 6), facecolor=colors['background'])
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)

        return widget

    def _on_strategy_changed(self, index: int):
        """Handle strategy selection change"""
        key = self.strategy_combo.currentData()

        # Update description
        descriptions = {
            'rsi': 'Buy when RSI < 30 (oversold), Sell when RSI > 70 (overbought)',
            'macd': 'Buy when MACD crosses above signal line, Sell on cross below',
            'bollinger': 'Buy at lower Bollinger Band, Sell at upper band',
            'ma_crossover': 'Buy on Golden Cross (SMA10 > SMA20), Sell on Death Cross',
            'stochastic': 'Buy when %K crosses %D in oversold zone, Sell in overbought',
            'ensemble': 'Combines multiple strategies with weighted voting',
            'prediction': 'Uses AI model predictions for trading signals',
        }
        self.strategy_desc.setText(descriptions.get(key, ''))

        # Show/hide model path for prediction strategy
        is_prediction = (key == 'prediction')
        self.model_path_input.setVisible(is_prediction)
        self.browse_btn.setVisible(is_prediction)
        # Hide the label too by iterating layout
        for i in range(self.model_path_row.count()):
            item = self.model_path_row.itemAt(i)
            if item and item.widget():
                item.widget().setVisible(is_prediction)

    def _on_run_click(self):
        """Handle run button click"""
        if self._is_running:
            return

        config = {
            'mode': 'strategy',
            'strategy': self.strategy_combo.currentData(),
            'symbol': self.symbol_combo.currentText(),
            'initial_capital': self.capital_spin.value(),
            'trade_fee': self.fee_spin.value() / 100,
            'confidence_threshold': self.confidence_spin.value(),
            'show_signals': self.show_signals_cb.isChecked(),
        }

        if config['strategy'] == 'prediction':
            config['mode'] = 'model'
            config['model_path'] = self.model_path_input.text()

        self.backtest_requested.emit(config)

    def set_symbols(self, symbols: List[str]):
        """Set available symbols"""
        self._available_symbols = symbols
        self.symbol_combo.clear()
        self.symbol_combo.addItems(symbols)

    def set_running(self, is_running: bool):
        """Set running state"""
        self._is_running = is_running
        self.run_btn.setEnabled(not is_running)
        self.strategy_combo.setEnabled(not is_running)
        self.symbol_combo.setEnabled(not is_running)
        self.progress_bar.setVisible(is_running)

        if is_running:
            self.run_btn.setText("Running...")
        else:
            self.run_btn.setText("Run Backtest")

    def update_progress(self, current: int, total: int, message: str):
        """Update progress display"""
        if total > 0:
            self.progress_bar.setValue(int((current / total) * 100))
        self.status_label.setText(message)

    def display_results(self, results):
        """
        Display backtest results.

        Args:
            results: BacktestResults object
        """
        # Update metrics
        self.metric_labels['initial'].setText(f"${results.initial_capital:,.2f}")
        self.metric_labels['final'].setText(f"${results.final_capital:,.2f}")

        ret_color = COLORS['chart_green'] if results.total_return >= 0 else COLORS['chart_red']
        self.metric_labels['return'].setText(
            f"<span style='color:{ret_color}'>${results.total_return:+,.2f}</span>"
        )
        self.metric_labels['return_pct'].setText(
            f"<span style='color:{ret_color}'>{results.total_return_pct:+.2f}%</span>"
        )

        self.metric_labels['trades'].setText(str(results.num_trades))
        self.metric_labels['win_rate'].setText(f"{results.win_rate:.1f}%")
        self.metric_labels['pl_ratio'].setText(f"{results.profit_loss_ratio:.2f}")

        bh_color = COLORS['chart_green'] if results.buy_hold_return >= 0 else COLORS['chart_red']
        self.metric_labels['buy_hold'].setText(
            f"<span style='color:{bh_color}'>{results.buy_hold_return_pct:+.2f}%</span>"
        )

        ex_color = COLORS['chart_green'] if results.excess_return >= 0 else COLORS['chart_red']
        self.metric_labels['excess'].setText(
            f"<span style='color:{ex_color}'>${results.excess_return:+,.2f}</span>"
        )

        self.metric_labels['dir_acc'].setText(f"{results.direction_accuracy:.1f}%")

        # Draw equity curve
        self._draw_equity_curve(results)

        self.status_label.setText("Backtest complete")

    def _draw_equity_curve(self, results):
        """Draw equity curve chart"""
        self.figure.clear()
        colors = get_chart_colors()

        ax = self.figure.add_subplot(111)
        ax.set_facecolor(colors['background'])
        ax.tick_params(colors=colors['text'])

        # Plot equity curve
        equity = results.equity_curve
        ax.plot(equity.index, equity.values, color=colors['primary'], linewidth=2, label='Strategy')

        # Plot initial capital line
        ax.axhline(results.initial_capital, color=colors['text'], linestyle='--',
                   alpha=0.5, label='Initial Capital')

        ax.set_xlabel('Date', color=colors['text'])
        ax.set_ylabel('Portfolio Value ($)', color=colors['text'])
        ax.set_title('Equity Curve', color=colors['text'], fontsize=12)
        ax.legend(facecolor=colors['background'], edgecolor=colors['grid'],
                 labelcolor=colors['text'])
        ax.grid(True, alpha=0.3, color=colors['grid'])

        self.figure.tight_layout()
        self.canvas.draw()

    def clear_results(self):
        """Clear results display"""
        for label in self.metric_labels.values():
            label.setText("-")
        self.figure.clear()
        self.canvas.draw()
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")
