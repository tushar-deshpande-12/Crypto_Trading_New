"""
Backtest Panel Component - Strategy-based backtesting interface

Provides UI for selecting and backtesting trading strategies with results visualization.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QProgressBar, QGroupBox, QLineEdit, QDoubleSpinBox,
    QTextEdit, QSplitter, QFrame, QGridLayout, QCheckBox, QFileDialog,
    QSpinBox
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
        # Basic strategies
        ('rsi', 'RSI Oversold/Overbought'),
        ('macd', 'MACD Crossover'),
        ('bollinger', 'Bollinger Bands'),
        ('ma_crossover', 'MA Crossover'),
        ('stochastic', 'Stochastic'),
        # Combo strategies (famous combinations)
        ('rsi_macd', 'RSI + MACD Combo'),
        ('bb_rsi', 'Bollinger + RSI Combo'),
        ('macd_ma', 'MACD + MA Combo'),
        ('stoch_rsi', 'Stochastic + RSI Combo'),
        ('triple_ema', 'Triple EMA Crossover'),
        ('adx_macd', 'ADX + MACD Trend'),
        # Multi-strategy
        ('ensemble', 'Ensemble (Multi-Strategy)'),
        # AI-based
        ('prediction', 'AI Model Prediction (LSTM)'),
        ('walk_forward', 'Walk-Forward AI Backtest'),
        # Prophet-based
        ('prophet', 'Prophet Forecast Backtest'),
        ('prophet_walk_forward', 'Walk-Forward Prophet Backtest'),
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
        self.browse_btn.clicked.connect(self._browse_model)
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

        # Walk-Forward Parameters (hidden by default)
        self.wf_params_group = QGroupBox("Walk-Forward Parameters")
        wf_layout = QGridLayout(self.wf_params_group)

        # Number of folds
        wf_layout.addWidget(QLabel("Number of Folds:"), 0, 0)
        self.wf_folds_spin = QSpinBox()
        self.wf_folds_spin.setRange(3, 20)
        self.wf_folds_spin.setValue(5)
        self.wf_folds_spin.setToolTip("Number of train/test splits for walk-forward analysis")
        wf_layout.addWidget(self.wf_folds_spin, 0, 1)

        # Train ratio
        wf_layout.addWidget(QLabel("Train Ratio:"), 1, 0)
        self.wf_train_ratio_spin = QDoubleSpinBox()
        self.wf_train_ratio_spin.setRange(0.5, 0.9)
        self.wf_train_ratio_spin.setValue(0.7)
        self.wf_train_ratio_spin.setSingleStep(0.05)
        self.wf_train_ratio_spin.setDecimals(2)
        self.wf_train_ratio_spin.setToolTip("Fraction of each fold used for training")
        wf_layout.addWidget(self.wf_train_ratio_spin, 1, 1)

        # Gap periods
        wf_layout.addWidget(QLabel("Gap Periods:"), 2, 0)
        self.wf_gap_spin = QSpinBox()
        self.wf_gap_spin.setRange(0, 48)
        self.wf_gap_spin.setValue(4)
        self.wf_gap_spin.setToolTip("Hours gap between train and test to prevent lookahead")
        wf_layout.addWidget(self.wf_gap_spin, 2, 1)

        # Retrain on each fold
        self.wf_retrain_cb = QCheckBox("Retrain on each fold")
        self.wf_retrain_cb.setChecked(True)
        self.wf_retrain_cb.setToolTip("Retrain model on each fold's training data")
        wf_layout.addWidget(self.wf_retrain_cb, 3, 0, 1, 2)

        self.wf_params_group.setVisible(False)
        layout.addWidget(self.wf_params_group)

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
            # Basic strategies
            'rsi': 'Buy when RSI < 30 (oversold), Sell when RSI > 70 (overbought)',
            'macd': 'Buy when MACD crosses above signal line, Sell on cross below',
            'bollinger': 'Buy at lower Bollinger Band, Sell at upper band',
            'ma_crossover': 'Buy on Golden Cross (SMA10 > SMA20), Sell on Death Cross',
            'stochastic': 'Buy when %K crosses %D in oversold zone, Sell in overbought',
            # Combo strategies
            'rsi_macd': 'RSI oversold/overbought + MACD crossover confirmation',
            'bb_rsi': 'Bollinger Band position + RSI momentum for mean reversion',
            'macd_ma': 'MACD crossover + MA trend direction confirmation',
            'stoch_rsi': 'Double momentum filter using Stochastic and RSI',
            'triple_ema': 'Classic trend following with 3 EMA alignment (5/13/50)',
            'adx_macd': 'Only trade MACD signals when ADX shows strong trend (>25)',
            # Multi-strategy
            'ensemble': 'Combines multiple strategies with weighted voting',
            # AI-based
            'prediction': 'Uses LSTM model predictions for trading signals',
            'walk_forward': 'Rolling train/test splits with LSTM model retraining for realistic performance',
            # Prophet-based
            'prophet': 'Uses Prophet time series forecasts for trading signals (2.5% min profit)',
            'prophet_walk_forward': 'Walk-forward Prophet forecasting with periodic retraining',
        }
        self.strategy_desc.setText(descriptions.get(key, ''))

        # Show/hide model path for LSTM prediction or walk_forward strategy
        is_lstm_strategy = (key in ('prediction', 'walk_forward'))
        self.model_path_input.setVisible(is_lstm_strategy)
        self.browse_btn.setVisible(is_lstm_strategy)
        # Hide the label too by iterating layout
        for i in range(self.model_path_row.count()):
            item = self.model_path_row.itemAt(i)
            if item and item.widget():
                item.widget().setVisible(is_lstm_strategy)

        # Show/hide walk-forward parameters
        is_walk_forward = (key in ('walk_forward', 'prophet_walk_forward'))
        self.wf_params_group.setVisible(is_walk_forward)

    def _browse_model(self):
        """Browse for AI model file"""
        from pathlib import Path

        # Start in models/checkpoints if it exists
        start_dir = "models/checkpoints"
        if not Path(start_dir).exists():
            start_dir = "."

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select AI Model",
            start_dir,
            "Model Files (*.pt *.pth *.ckpt);;All Files (*.*)"
        )

        if file_path:
            self.model_path_input.setText(file_path)

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
        elif config['strategy'] == 'walk_forward':
            config['mode'] = 'walk_forward'
            config['model_path'] = self.model_path_input.text()
            config['n_folds'] = self.wf_folds_spin.value()
            config['train_ratio'] = self.wf_train_ratio_spin.value()
            config['gap_periods'] = self.wf_gap_spin.value()
            config['retrain'] = self.wf_retrain_cb.isChecked()

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

    def display_walk_forward_results(self, results):
        """
        Display walk-forward backtest results.

        Args:
            results: WalkForwardResults object with aggregated metrics
        """
        # Update metrics with walk-forward specific values
        self.metric_labels['initial'].setText(f"${results.initial_capital:,.2f}")
        self.metric_labels['final'].setText(f"${results.final_capital:,.2f}")

        # Total return
        total_return = results.final_capital - results.initial_capital
        total_return_pct = (total_return / results.initial_capital) * 100
        ret_color = COLORS['chart_green'] if total_return >= 0 else COLORS['chart_red']
        self.metric_labels['return'].setText(
            f"<span style='color:{ret_color}'>${total_return:+,.2f}</span>"
        )
        self.metric_labels['return_pct'].setText(
            f"<span style='color:{ret_color}'>{total_return_pct:+.2f}%</span>"
        )

        # Walk-forward specific metrics
        self.metric_labels['trades'].setText(f"{results.n_folds} folds")

        # Win rate as avg accuracy
        acc_pct = results.avg_accuracy * 100
        self.metric_labels['win_rate'].setText(f"{acc_pct:.1f}% acc")

        # P/L ratio as avg Sharpe
        sharpe_color = COLORS['chart_green'] if results.avg_sharpe >= 0 else COLORS['chart_red']
        self.metric_labels['pl_ratio'].setText(
            f"<span style='color:{sharpe_color}'>{results.avg_sharpe:.2f} Sharpe</span>"
        )

        # Buy & Hold as avg IC
        ic_color = COLORS['chart_green'] if results.avg_ic >= 0 else COLORS['chart_red']
        self.metric_labels['buy_hold'].setText(
            f"<span style='color:{ic_color}'>{results.avg_ic:.4f} IC</span>"
        )

        # Excess return as stability
        self.metric_labels['excess'].setText(results.stability)

        # Direction accuracy
        self.metric_labels['dir_acc'].setText(f"{results.avg_accuracy*100:.1f}%")

        # Draw walk-forward equity curve
        self._draw_walk_forward_chart(results)

        self.status_label.setText(f"Walk-forward backtest complete ({results.n_folds} folds)")

    def _draw_walk_forward_chart(self, results):
        """Draw walk-forward results chart with fold performance"""
        self.figure.clear()
        colors = get_chart_colors()

        # Create 2 subplots: equity curve and per-fold metrics
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)

        for ax in [ax1, ax2]:
            ax.set_facecolor(colors['background'])
            ax.tick_params(colors=colors['text'])

        # Top: Cumulative equity curve
        if hasattr(results, 'equity_curve') and results.equity_curve is not None:
            equity = results.equity_curve
            ax1.plot(equity.index, equity.values, color=colors['primary'], linewidth=2)
            ax1.axhline(results.initial_capital, color=colors['text'], linestyle='--', alpha=0.5)
        else:
            # Fallback: show fold returns as cumulative
            fold_returns = [f.get('total_return_pct', 0) for f in results.fold_results]
            cumulative = [100]
            for ret in fold_returns:
                cumulative.append(cumulative[-1] * (1 + ret/100))
            ax1.plot(range(len(cumulative)), cumulative, color=colors['primary'],
                    linewidth=2, marker='o')
            ax1.axhline(100, color=colors['text'], linestyle='--', alpha=0.5)

        ax1.set_title('Walk-Forward Equity', color=colors['text'], fontsize=10)
        ax1.set_ylabel('Portfolio Value', color=colors['text'])
        ax1.grid(True, alpha=0.3, color=colors['grid'])

        # Bottom: Per-fold Sharpe ratios
        fold_sharpes = [f.get('sharpe', 0) for f in results.fold_results]
        fold_ics = [f.get('ic', 0) for f in results.fold_results]
        x = range(1, len(fold_sharpes) + 1)

        bar_width = 0.35
        bars1 = ax2.bar([i - bar_width/2 for i in x], fold_sharpes, bar_width,
                       label='Sharpe', color=colors['primary'], alpha=0.8)
        bars2 = ax2.bar([i + bar_width/2 for i in x], [ic * 10 for ic in fold_ics], bar_width,
                       label='IC (x10)', color=colors.get('secondary', '#ff9800'), alpha=0.8)

        ax2.axhline(0, color=colors['text'], linestyle='-', alpha=0.3)
        ax2.set_title('Per-Fold Performance', color=colors['text'], fontsize=10)
        ax2.set_xlabel('Fold', color=colors['text'])
        ax2.set_ylabel('Sharpe / IC*10', color=colors['text'])
        ax2.set_xticks(list(x))
        ax2.legend(facecolor=colors['background'], edgecolor=colors['grid'],
                  labelcolor=colors['text'], fontsize=8)
        ax2.grid(True, alpha=0.3, color=colors['grid'])

        self.figure.tight_layout()
        self.canvas.draw()
