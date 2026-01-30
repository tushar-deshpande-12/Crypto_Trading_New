"""
Directional AI Backtest Panel - Walk-forward backtesting for RankNet models

Provides a dedicated UI for backtesting Directional AI (RankNet) models with:
- Walk-forward cross-validation
- ICE metrics display (Information Coefficient, Sharpe, Accuracy)
- Per-fold performance visualization
- Equity curve tracking
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QProgressBar, QGroupBox, QDoubleSpinBox,
    QFrame, QGridLayout, QCheckBox, QFileDialog, QSpinBox,
    QSplitter, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from typing import Dict, List
from pathlib import Path

from src.gui_pyqt.styles import COLORS, get_chart_colors


class DirectionalBacktestPanel(QWidget):
    """
    Directional AI Backtest Panel for walk-forward RankNet backtesting.

    Features:
    - Symbol and model selection
    - Walk-forward parameters configuration
    - ICE metrics display (IC, Sharpe, Accuracy)
    - Per-fold performance charts
    - Equity curve visualization

    Signals:
        backtest_requested(dict): Config dict when backtest requested
    """

    backtest_requested = pyqtSignal(dict)

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
        header = QLabel("Directional AI Backtesting")
        header.setProperty("class", "title")
        layout.addWidget(header)

        subtitle = QLabel("Walk-forward backtesting for RankNet direction prediction models")
        subtitle.setProperty("class", "secondary")
        layout.addWidget(subtitle)

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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        widget = QFrame()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 8, 0)

        # Data Selection
        data_group = QGroupBox("Data Selection")
        data_layout = QVBoxLayout(data_group)

        # Symbol selection
        sym_row = QHBoxLayout()
        sym_row.addWidget(QLabel("Symbol:"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.setEditable(True)
        try:
            from src.gui_pyqt.utils.symbols import get_all_usdt_symbols
            self.symbol_combo.addItems(get_all_usdt_symbols())
        except Exception:
            from src.gui_pyqt.utils.symbols import get_default_symbols
            self.symbol_combo.addItems(get_default_symbols())
        sym_row.addWidget(self.symbol_combo, stretch=1)
        data_layout.addLayout(sym_row)

        # Model selection (optional - for pre-trained models)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Pre-trained Model:"))
        self.model_path_label = QLabel("(Optional - will train fresh if not selected)")
        self.model_path_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        model_row.addWidget(self.model_path_label, stretch=1)
        data_layout.addLayout(model_row)

        model_btn_row = QHBoxLayout()
        self.browse_btn = QPushButton("Browse Model")
        self.browse_btn.clicked.connect(self._browse_model)
        model_btn_row.addWidget(self.browse_btn)
        self.clear_model_btn = QPushButton("Clear")
        self.clear_model_btn.clicked.connect(self._clear_model)
        model_btn_row.addWidget(self.clear_model_btn)
        model_btn_row.addStretch()
        data_layout.addLayout(model_btn_row)

        self.model_path = ""

        layout.addWidget(data_group)

        # Walk-Forward Parameters
        wf_group = QGroupBox("Walk-Forward Parameters")
        wf_layout = QGridLayout(wf_group)

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
        self.wf_gap_spin.setToolTip("Hours gap between train and test to prevent lookahead bias")
        wf_layout.addWidget(self.wf_gap_spin, 2, 1)

        # Retrain on each fold
        self.wf_retrain_cb = QCheckBox("Retrain on each fold")
        self.wf_retrain_cb.setChecked(True)
        self.wf_retrain_cb.setToolTip("Retrain model on each fold's training data")
        wf_layout.addWidget(self.wf_retrain_cb, 3, 0, 1, 2)

        layout.addWidget(wf_group)

        # Model Parameters
        model_group = QGroupBox("Model Parameters")
        model_layout = QGridLayout(model_group)

        # Hidden size
        model_layout.addWidget(QLabel("Hidden Size:"), 0, 0)
        self.hidden_size_spin = QSpinBox()
        self.hidden_size_spin.setRange(32, 512)
        self.hidden_size_spin.setValue(128)
        model_layout.addWidget(self.hidden_size_spin, 0, 1)

        # Num blocks
        model_layout.addWidget(QLabel("Num Blocks:"), 1, 0)
        self.num_blocks_spin = QSpinBox()
        self.num_blocks_spin.setRange(1, 8)
        self.num_blocks_spin.setValue(3)
        model_layout.addWidget(self.num_blocks_spin, 1, 1)

        # Dropout
        model_layout.addWidget(QLabel("Dropout:"), 2, 0)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.5)
        self.dropout_spin.setValue(0.1)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setDecimals(2)
        model_layout.addWidget(self.dropout_spin, 2, 1)

        # Max epochs
        model_layout.addWidget(QLabel("Max Epochs:"), 3, 0)
        self.max_epochs_spin = QSpinBox()
        self.max_epochs_spin.setRange(5, 100)
        self.max_epochs_spin.setValue(20)
        model_layout.addWidget(self.max_epochs_spin, 3, 1)

        # Learning rate
        model_layout.addWidget(QLabel("Learning Rate:"), 4, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.01)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setDecimals(4)
        model_layout.addWidget(self.lr_spin, 4, 1)

        layout.addWidget(model_group)

        # Capital Parameters
        capital_group = QGroupBox("Capital Parameters")
        capital_layout = QGridLayout(capital_group)

        # Initial capital
        capital_layout.addWidget(QLabel("Initial Capital ($):"), 0, 0)
        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(100, 1000000)
        self.capital_spin.setValue(1000)
        self.capital_spin.setPrefix("$")
        capital_layout.addWidget(self.capital_spin, 0, 1)

        # Trade fee
        capital_layout.addWidget(QLabel("Trade Fee (%):"), 1, 0)
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setRange(0, 5)
        self.fee_spin.setValue(0.1)
        self.fee_spin.setSuffix("%")
        self.fee_spin.setDecimals(2)
        capital_layout.addWidget(self.fee_spin, 1, 1)

        layout.addWidget(capital_group)

        # Run button
        self.run_btn = QPushButton("Run Walk-Forward Backtest")
        self.run_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                font-weight: bold;
                padding: 14px 28px;
                border-radius: 6px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_secondary']};
            }}
        """)
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

        scroll.setWidget(widget)
        return scroll

    def _create_results_panel(self) -> QWidget:
        """Create results panel"""
        widget = QFrame()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 0, 0, 0)

        # ICE Metrics Box (Prominent display)
        ice_group = QGroupBox("ICE Metrics (Information Coefficient & Performance)")
        ice_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        ice_layout = QHBoxLayout(ice_group)

        # IC Display
        ic_frame = QFrame()
        ic_frame.setStyleSheet(f"background-color: {COLORS['bg_medium']}; border-radius: 6px; padding: 8px;")
        ic_inner = QVBoxLayout(ic_frame)
        ic_inner.setContentsMargins(12, 8, 12, 8)

        ic_title = QLabel("Information Coefficient")
        ic_title.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        ic_inner.addWidget(ic_title)

        self.ic_display = QLabel("--")
        self.ic_display.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {COLORS['accent']};")
        self.ic_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ic_inner.addWidget(self.ic_display)

        self.ic_rating_display = QLabel("Run backtest to see IC")
        self.ic_rating_display.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 10px;")
        self.ic_rating_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ic_inner.addWidget(self.ic_rating_display)

        ice_layout.addWidget(ic_frame)

        # Sharpe Display
        sharpe_frame = QFrame()
        sharpe_frame.setStyleSheet(f"background-color: {COLORS['bg_medium']}; border-radius: 6px; padding: 8px;")
        sharpe_inner = QVBoxLayout(sharpe_frame)
        sharpe_inner.setContentsMargins(12, 8, 12, 8)

        sharpe_title = QLabel("Sharpe Ratio")
        sharpe_title.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        sharpe_inner.addWidget(sharpe_title)

        self.sharpe_display = QLabel("--")
        self.sharpe_display.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {COLORS['accent']};")
        self.sharpe_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sharpe_inner.addWidget(self.sharpe_display)

        self.sharpe_rating_display = QLabel("Risk-adjusted return")
        self.sharpe_rating_display.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 10px;")
        self.sharpe_rating_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sharpe_inner.addWidget(self.sharpe_rating_display)

        ice_layout.addWidget(sharpe_frame)

        # Direction Accuracy Display
        acc_frame = QFrame()
        acc_frame.setStyleSheet(f"background-color: {COLORS['bg_medium']}; border-radius: 6px; padding: 8px;")
        acc_inner = QVBoxLayout(acc_frame)
        acc_inner.setContentsMargins(12, 8, 12, 8)

        acc_title = QLabel("Direction Accuracy")
        acc_title.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        acc_inner.addWidget(acc_title)

        self.acc_display = QLabel("--")
        self.acc_display.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {COLORS['accent']};")
        self.acc_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        acc_inner.addWidget(self.acc_display)

        self.acc_rating_display = QLabel("Predictive accuracy")
        self.acc_rating_display.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 10px;")
        self.acc_rating_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        acc_inner.addWidget(self.acc_rating_display)

        ice_layout.addWidget(acc_frame)

        # Stability Display
        stab_frame = QFrame()
        stab_frame.setStyleSheet(f"background-color: {COLORS['bg_medium']}; border-radius: 6px; padding: 8px;")
        stab_inner = QVBoxLayout(stab_frame)
        stab_inner.setContentsMargins(12, 8, 12, 8)

        stab_title = QLabel("Stability")
        stab_title.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        stab_inner.addWidget(stab_title)

        self.stab_display = QLabel("--")
        self.stab_display.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {COLORS['accent']};")
        self.stab_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stab_inner.addWidget(self.stab_display)

        self.stab_rating_display = QLabel("Consistency across folds")
        self.stab_rating_display.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 10px;")
        self.stab_rating_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stab_inner.addWidget(self.stab_rating_display)

        ice_layout.addWidget(stab_frame)

        layout.addWidget(ice_group)

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
            ('n_folds', 'Folds Tested'),
            ('avg_edge', 'Avg Edge (%)'),
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

        # Charts
        colors = get_chart_colors()
        self.figure = Figure(figsize=(8, 8), facecolor=colors['background'])
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)

        return widget

    def _browse_model(self):
        """Browse for AI model file"""
        start_dir = "models/checkpoints"
        if not Path(start_dir).exists():
            start_dir = "."

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select RankNet Model",
            start_dir,
            "Model Files (*.pt *.pth);;All Files (*.*)"
        )

        if file_path:
            self.model_path = file_path
            self.model_path_label.setText(Path(file_path).name)
            self.model_path_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 11px; font-weight: bold;")

    def _clear_model(self):
        """Clear selected model"""
        self.model_path = ""
        self.model_path_label.setText("(Optional - will train fresh if not selected)")
        self.model_path_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")

    def _on_run_click(self):
        """Handle run button click"""
        if self._is_running:
            return

        config = {
            'mode': 'walk_forward',
            'strategy': 'walk_forward',
            'symbol': self.symbol_combo.currentText(),
            'model_path': self.model_path,
            'initial_capital': self.capital_spin.value(),
            'trade_fee': self.fee_spin.value() / 100,
            'n_folds': self.wf_folds_spin.value(),
            'train_ratio': self.wf_train_ratio_spin.value(),
            'gap_periods': self.wf_gap_spin.value(),
            'retrain': self.wf_retrain_cb.isChecked(),
            'hidden_size': self.hidden_size_spin.value(),
            'num_blocks': self.num_blocks_spin.value(),
            'dropout': self.dropout_spin.value(),
            'max_epochs': self.max_epochs_spin.value(),
            'learning_rate': self.lr_spin.value(),
        }

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
        self.symbol_combo.setEnabled(not is_running)
        self.progress_bar.setVisible(is_running)

        if is_running:
            self.run_btn.setText("Running...")
        else:
            self.run_btn.setText("Run Walk-Forward Backtest")

    def update_progress(self, current: int, total: int, message: str):
        """Update progress display"""
        if total > 0:
            self.progress_bar.setValue(int((current / total) * 100))
        self.status_label.setText(message)

    def display_results(self, results):
        """Display walk-forward backtest results"""
        # Update ICE metrics display
        self._update_ice_display(results)

        # Update metrics
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

        self.metric_labels['n_folds'].setText(f"{results.n_folds}")

        # Average edge
        avg_edge = (results.avg_accuracy - 0.5) * 100
        edge_color = COLORS['chart_green'] if avg_edge > 0 else COLORS['chart_red']
        self.metric_labels['avg_edge'].setText(
            f"<span style='color:{edge_color}'>{avg_edge:+.1f}%</span>"
        )

        # Draw charts
        self._draw_charts(results)

        self.status_label.setText(f"Walk-forward backtest complete ({results.n_folds} folds)")

    def _update_ice_display(self, results):
        """Update ICE display for walk-forward results"""
        # Information Coefficient
        ic = getattr(results, 'avg_ic', 0)
        self.ic_display.setText(f"{ic:.4f}")

        if ic > 0.10:
            ic_color = COLORS['chart_green']
            ic_rating = "EXCELLENT"
        elif ic > 0.05:
            ic_color = "#81c784"
            ic_rating = "GOOD"
        elif ic > 0.02:
            ic_color = COLORS['accent']
            ic_rating = "MODERATE"
        elif ic > 0:
            ic_color = COLORS['text_secondary']
            ic_rating = "WEAK"
        else:
            ic_color = COLORS['chart_red']
            ic_rating = "NONE"

        self.ic_display.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {ic_color};")
        self.ic_rating_display.setText(f"{ic_rating} (avg {results.n_folds} folds)")
        self.ic_rating_display.setStyleSheet(f"color: {ic_color}; font-size: 10px; font-weight: bold;")

        # Sharpe Ratio
        sharpe = getattr(results, 'avg_sharpe', 0)
        self.sharpe_display.setText(f"{sharpe:.2f}")

        if sharpe > 2:
            sharpe_color = COLORS['chart_green']
            sharpe_rating = "EXCELLENT"
        elif sharpe > 1:
            sharpe_color = "#81c784"
            sharpe_rating = "GOOD"
        elif sharpe > 0.5:
            sharpe_color = COLORS['accent']
            sharpe_rating = "ACCEPTABLE"
        elif sharpe > 0:
            sharpe_color = COLORS['text_secondary']
            sharpe_rating = "MARGINAL"
        else:
            sharpe_color = COLORS['chart_red']
            sharpe_rating = "POOR"

        self.sharpe_display.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {sharpe_color};")
        self.sharpe_rating_display.setText(sharpe_rating)
        self.sharpe_rating_display.setStyleSheet(f"color: {sharpe_color}; font-size: 10px; font-weight: bold;")

        # Direction Accuracy
        accuracy = getattr(results, 'avg_accuracy', 0.5) * 100
        self.acc_display.setText(f"{accuracy:.1f}%")

        if accuracy > 55:
            acc_color = COLORS['chart_green']
            acc_rating = "STRONG EDGE"
        elif accuracy > 52:
            acc_color = "#81c784"
            acc_rating = "GOOD EDGE"
        elif accuracy > 50:
            acc_color = COLORS['accent']
            acc_rating = "SLIGHT EDGE"
        else:
            acc_color = COLORS['chart_red']
            acc_rating = "NO EDGE"

        self.acc_display.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {acc_color};")
        self.acc_rating_display.setText(acc_rating)
        self.acc_rating_display.setStyleSheet(f"color: {acc_color}; font-size: 10px; font-weight: bold;")

        # Stability
        stability = getattr(results, 'stability', 'Unknown')
        self.stab_display.setText(stability)

        if stability == "Stable":
            stab_color = COLORS['chart_green']
        elif stability == "Moderate":
            stab_color = COLORS['accent']
        else:
            stab_color = COLORS['chart_red']

        self.stab_display.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {stab_color};")

    def _draw_charts(self, results):
        """Draw walk-forward results charts"""
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

        ax1.set_title('Walk-Forward Equity Curve', color=colors['text'], fontsize=11)
        ax1.set_ylabel('Portfolio Value ($)', color=colors['text'])
        ax1.grid(True, alpha=0.3, color=colors['grid'])

        # Bottom: Per-fold metrics
        fold_sharpes = [f.get('sharpe', 0) for f in results.fold_results]
        fold_accuracies = [(f.get('accuracy', 0.5) - 0.5) * 100 for f in results.fold_results]  # Edge %
        x = range(1, len(fold_sharpes) + 1)

        bar_width = 0.35
        bars1 = ax2.bar([i - bar_width/2 for i in x], fold_sharpes, bar_width,
                       label='Sharpe Ratio', color=colors['primary'], alpha=0.8)
        bars2 = ax2.bar([i + bar_width/2 for i in x], fold_accuracies, bar_width,
                       label='Edge (%)', color=colors.get('secondary', '#ff9800'), alpha=0.8)

        ax2.axhline(0, color=colors['text'], linestyle='-', alpha=0.3)
        ax2.set_title('Per-Fold Performance', color=colors['text'], fontsize=11)
        ax2.set_xlabel('Fold', color=colors['text'])
        ax2.set_ylabel('Sharpe / Edge (%)', color=colors['text'])
        ax2.set_xticks(list(x))
        ax2.legend(facecolor=colors['background'], edgecolor=colors['grid'],
                  labelcolor=colors['text'], fontsize=8)
        ax2.grid(True, alpha=0.3, color=colors['grid'])

        self.figure.tight_layout()
        self.canvas.draw()

    def clear_results(self):
        """Clear results display"""
        for label in self.metric_labels.values():
            label.setText("-")

        # Clear ICE display
        self.ic_display.setText("--")
        self.ic_display.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {COLORS['accent']};")
        self.ic_rating_display.setText("Run backtest to see IC")
        self.ic_rating_display.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 10px;")

        self.sharpe_display.setText("--")
        self.sharpe_display.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {COLORS['accent']};")
        self.sharpe_rating_display.setText("Risk-adjusted return")
        self.sharpe_rating_display.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 10px;")

        self.acc_display.setText("--")
        self.acc_display.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {COLORS['accent']};")
        self.acc_rating_display.setText("Predictive accuracy")
        self.acc_rating_display.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 10px;")

        self.stab_display.setText("--")
        self.stab_display.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {COLORS['accent']};")

        self.figure.clear()
        self.canvas.draw()
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")
