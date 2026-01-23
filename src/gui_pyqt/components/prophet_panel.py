# prophet_panel.py - Prophet Forecasting Tab
"""
Prophet-based Time Series Forecasting Panel

Features:
- Price forecasting (not just direction)
- Multiple forecast horizons (1h, 4h, 24h, 7d)
- Confidence intervals
- Trend and seasonality decomposition
- Visual forecast charts
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QProgressBar, QTextEdit, QFrame, QSplitter,
    QCheckBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont, QColor

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Matplotlib for charts
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ProphetWorker(QThread):
    """Background worker for Prophet model training and forecasting."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, df, config):
        super().__init__()
        self.df = df
        self.config = config

    def run(self):
        try:
            from src.ml.models.prophet_model import CryptoProphetModel

            self.progress.emit(10, "Initializing Prophet model...")

            model = CryptoProphetModel(
                changepoint_prior_scale=self.config.get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=self.config.get('seasonality_prior_scale', 10.0),
                seasonality_mode=self.config.get('seasonality_mode', 'multiplicative'),
                include_volume_regressor=self.config.get('include_volume', True),
                include_volatility_regressor=self.config.get('include_volatility', True),
                verbose=False
            )

            self.progress.emit(30, "Fitting model on historical data...")
            model.fit(self.df)

            self.progress.emit(60, "Generating forecasts...")
            forecast_periods = self.config.get('forecast_periods', 168)  # 7 days default
            forecast = model.predict(periods=forecast_periods, include_history=True)

            self.progress.emit(80, "Computing forecast summary...")
            summary = model.get_forecast_summary(periods=forecast_periods)

            self.progress.emit(90, "Getting components...")
            components = model.get_components()

            self.progress.emit(100, "Complete!")

            result = {
                'forecast': forecast,
                'summary': summary,
                'components': components,
                'model': model,
            }
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


class ProphetPanel(QWidget):
    """Prophet Forecasting Panel"""

    # Signals
    forecast_requested = pyqtSignal(str, dict)  # symbol, config
    backtest_requested = pyqtSignal(str, dict)  # symbol, config

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.current_forecast = None
        self.current_model = None
        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Title
        title = QLabel("Prophet Time Series Forecasting")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        main_layout.addWidget(title)

        # Description
        desc = QLabel("Forecast future prices with confidence intervals using Facebook Prophet")
        desc.setStyleSheet("color: gray;")
        main_layout.addWidget(desc)

        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        # Left panel - Configuration
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Data Selection
        data_group = QGroupBox("Data Selection")
        data_layout = QGridLayout(data_group)

        data_layout.addWidget(QLabel("Symbol:"), 0, 0)
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT'])
        data_layout.addWidget(self.symbol_combo, 0, 1)

        data_layout.addWidget(QLabel("Dataset:"), 1, 0)
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItem("Auto-detect latest")
        data_layout.addWidget(self.dataset_combo, 1, 1)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_datasets)
        data_layout.addWidget(self.refresh_btn, 1, 2)

        left_layout.addWidget(data_group)

        # Model Configuration
        config_group = QGroupBox("Model Configuration")
        config_layout = QGridLayout(config_group)

        # Forecast horizon
        config_layout.addWidget(QLabel("Forecast Horizon:"), 0, 0)
        self.horizon_combo = QComboBox()
        self.horizon_combo.addItems(['24 hours', '48 hours', '7 days', '14 days', '30 days'])
        self.horizon_combo.setCurrentIndex(2)  # Default 7 days
        config_layout.addWidget(self.horizon_combo, 0, 1)

        # Changepoint prior scale
        config_layout.addWidget(QLabel("Trend Flexibility:"), 1, 0)
        self.changepoint_spin = QDoubleSpinBox()
        self.changepoint_spin.setRange(0.001, 0.5)
        self.changepoint_spin.setValue(0.05)
        self.changepoint_spin.setSingleStep(0.01)
        self.changepoint_spin.setToolTip("Higher = more flexible trend (0.001-0.5)")
        config_layout.addWidget(self.changepoint_spin, 1, 1)

        # Seasonality prior scale
        config_layout.addWidget(QLabel("Seasonality Strength:"), 2, 0)
        self.seasonality_spin = QDoubleSpinBox()
        self.seasonality_spin.setRange(0.1, 50.0)
        self.seasonality_spin.setValue(10.0)
        self.seasonality_spin.setSingleStep(1.0)
        self.seasonality_spin.setToolTip("Higher = stronger seasonality effects")
        config_layout.addWidget(self.seasonality_spin, 2, 1)

        # Seasonality mode
        config_layout.addWidget(QLabel("Seasonality Mode:"), 3, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['multiplicative', 'additive'])
        config_layout.addWidget(self.mode_combo, 3, 1)

        # Regressors
        self.volume_cb = QCheckBox("Include Volume")
        self.volume_cb.setChecked(True)
        config_layout.addWidget(self.volume_cb, 4, 0)

        self.volatility_cb = QCheckBox("Include Volatility")
        self.volatility_cb.setChecked(True)
        config_layout.addWidget(self.volatility_cb, 4, 1)

        left_layout.addWidget(config_group)

        # Action buttons
        action_layout = QHBoxLayout()
        self.train_btn = QPushButton("Generate Forecast")
        self.train_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.train_btn.clicked.connect(self._on_forecast_click)
        action_layout.addWidget(self.train_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_click)
        action_layout.addWidget(self.stop_btn)

        left_layout.addLayout(action_layout)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray;")
        left_layout.addWidget(self.status_label)

        # Forecast Summary
        summary_group = QGroupBox("Forecast Summary")
        summary_layout = QGridLayout(summary_group)

        self.current_price_label = QLabel("Current: --")
        self.current_price_label.setFont(QFont("Arial", 11))
        summary_layout.addWidget(self.current_price_label, 0, 0, 1, 2)

        # 1h forecast
        summary_layout.addWidget(QLabel("1h Forecast:"), 1, 0)
        self.forecast_1h_label = QLabel("--")
        self.forecast_1h_label.setFont(QFont("Arial", 10, QFont.Bold))
        summary_layout.addWidget(self.forecast_1h_label, 1, 1)

        # 4h forecast
        summary_layout.addWidget(QLabel("4h Forecast:"), 2, 0)
        self.forecast_4h_label = QLabel("--")
        self.forecast_4h_label.setFont(QFont("Arial", 10, QFont.Bold))
        summary_layout.addWidget(self.forecast_4h_label, 2, 1)

        # 24h forecast
        summary_layout.addWidget(QLabel("24h Forecast:"), 3, 0)
        self.forecast_24h_label = QLabel("--")
        self.forecast_24h_label.setFont(QFont("Arial", 10, QFont.Bold))
        summary_layout.addWidget(self.forecast_24h_label, 3, 1)

        # Direction
        summary_layout.addWidget(QLabel("Direction:"), 4, 0)
        self.direction_label = QLabel("--")
        self.direction_label.setFont(QFont("Arial", 12, QFont.Bold))
        summary_layout.addWidget(self.direction_label, 4, 1)

        # Confidence range
        summary_layout.addWidget(QLabel("Confidence Range:"), 5, 0)
        self.confidence_label = QLabel("--")
        summary_layout.addWidget(self.confidence_label, 5, 1)

        left_layout.addWidget(summary_group)
        left_layout.addStretch()

        splitter.addWidget(left_widget)

        # Right panel - Charts
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Chart tabs
        self.chart_tabs = QTabWidget()

        # Forecast chart
        self.forecast_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        self.forecast_ax = self.forecast_canvas.figure.add_subplot(111)
        self.chart_tabs.addTab(self.forecast_canvas, "Forecast")

        # Components chart
        self.components_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        self.chart_tabs.addTab(self.components_canvas, "Components")

        right_layout.addWidget(self.chart_tabs)

        splitter.addWidget(right_widget)
        splitter.setSizes([350, 650])

        # Initial dataset refresh
        QTimer.singleShot(100, self._refresh_datasets)

    def _refresh_datasets(self):
        """Refresh available datasets."""
        self.dataset_combo.clear()
        self.dataset_combo.addItem("Auto-detect latest")

        symbol = self.symbol_combo.currentText()
        dataset_dir = Path(f"dataset/{symbol}")

        if dataset_dir.exists():
            datasets = sorted(dataset_dir.glob("*/data.csv"), reverse=True)
            for ds in datasets[:10]:  # Show last 10
                name = ds.parent.name
                self.dataset_combo.addItem(name, str(ds))

    def _get_config(self) -> dict:
        """Get current configuration."""
        horizon_map = {
            '24 hours': 24,
            '48 hours': 48,
            '7 days': 168,
            '14 days': 336,
            '30 days': 720,
        }

        return {
            'forecast_periods': horizon_map.get(self.horizon_combo.currentText(), 168),
            'changepoint_prior_scale': self.changepoint_spin.value(),
            'seasonality_prior_scale': self.seasonality_spin.value(),
            'seasonality_mode': self.mode_combo.currentText(),
            'include_volume': self.volume_cb.isChecked(),
            'include_volatility': self.volatility_cb.isChecked(),
        }

    def _on_forecast_click(self):
        """Handle forecast button click."""
        symbol = self.symbol_combo.currentText()

        # Find dataset
        if self.dataset_combo.currentIndex() == 0:
            # Auto-detect
            dataset_dir = Path(f"dataset/{symbol}")
            if not dataset_dir.exists():
                QMessageBox.warning(self, "Error", f"No dataset found for {symbol}")
                return
            datasets = sorted(dataset_dir.glob("*/data.csv"), reverse=True)
            if not datasets:
                QMessageBox.warning(self, "Error", f"No data.csv found for {symbol}")
                return
            data_path = datasets[0]
        else:
            data_path = Path(self.dataset_combo.currentData())

        # Load data
        try:
            df = pd.read_csv(data_path)
            if len(df) < 100:
                QMessageBox.warning(self, "Error", "Insufficient data (need at least 100 rows)")
                return
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load data: {e}")
            return

        # Start worker
        config = self._get_config()
        self.worker = ProphetWorker(df, config)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_forecast_complete)
        self.worker.error.connect(self._on_error)

        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker.start()

    def _on_stop_click(self):
        """Stop the worker."""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()

        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Stopped")

    def _on_progress(self, value: int, message: str):
        """Update progress."""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def _on_forecast_complete(self, result: dict):
        """Handle forecast completion."""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        self.current_forecast = result['forecast']
        self.current_model = result['model']
        summary = result['summary']

        # Update summary labels
        self.current_price_label.setText(f"Current: ${summary['current_price']:,.2f}")

        # 1h forecast
        change_1h = summary['change_1h_pct']
        color_1h = "#4CAF50" if change_1h > 0 else "#F44336" if change_1h < 0 else "#888"
        self.forecast_1h_label.setText(f"${summary['forecast_1h']:,.2f} ({change_1h:+.2f}%)")
        self.forecast_1h_label.setStyleSheet(f"color: {color_1h};")

        # 4h forecast
        change_4h = summary['change_4h_pct']
        color_4h = "#4CAF50" if change_4h > 0 else "#F44336" if change_4h < 0 else "#888"
        self.forecast_4h_label.setText(f"${summary['forecast_4h']:,.2f} ({change_4h:+.2f}%)")
        self.forecast_4h_label.setStyleSheet(f"color: {color_4h};")

        # 24h forecast
        change_24h = summary['change_24h_pct']
        color_24h = "#4CAF50" if change_24h > 0 else "#F44336" if change_24h < 0 else "#888"
        self.forecast_24h_label.setText(f"${summary['forecast_24h']:,.2f} ({change_24h:+.2f}%)")
        self.forecast_24h_label.setStyleSheet(f"color: {color_24h};")

        # Direction
        direction = summary['direction_24h']
        dir_color = "#4CAF50" if direction == "UP" else "#F44336" if direction == "DOWN" else "#888"
        self.direction_label.setText(direction)
        self.direction_label.setStyleSheet(f"color: {dir_color}; font-size: 14px;")

        # Confidence
        self.confidence_label.setText(f"${summary['confidence_lower_1h']:,.2f} - ${summary['confidence_upper_1h']:,.2f}")

        self.status_label.setText(f"Forecast generated at {datetime.now().strftime('%H:%M:%S')}")

        # Update charts
        self._update_forecast_chart(result['forecast'], summary)
        self._update_components_chart(result['components'])

    def _on_error(self, error_msg: str):
        """Handle error."""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {error_msg}")
        QMessageBox.critical(self, "Error", f"Forecast failed:\n{error_msg}")

    def _update_forecast_chart(self, forecast: pd.DataFrame, summary: dict):
        """Update forecast chart."""
        self.forecast_ax.clear()

        # Plot historical + forecast
        self.forecast_ax.plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecast', linewidth=1.5)

        # Confidence intervals
        self.forecast_ax.fill_between(
            forecast['ds'],
            forecast['yhat_lower'],
            forecast['yhat_upper'],
            alpha=0.2, color='blue', label='95% Confidence'
        )

        # Current price line
        current_price = summary['current_price']
        self.forecast_ax.axhline(y=current_price, color='gray', linestyle='--', alpha=0.5, label='Current Price')

        self.forecast_ax.set_xlabel('Date')
        self.forecast_ax.set_ylabel('Price ($)')
        self.forecast_ax.set_title('Prophet Price Forecast with Confidence Intervals')
        self.forecast_ax.legend(loc='upper left')
        self.forecast_ax.grid(True, alpha=0.3)

        # Rotate x labels
        self.forecast_canvas.figure.autofmt_xdate()
        self.forecast_canvas.draw()

    def _update_components_chart(self, components: pd.DataFrame):
        """Update components chart."""
        fig = self.components_canvas.figure
        fig.clear()

        # Create subplots for trend and seasonalities
        axes = fig.subplots(3, 1, sharex=False)

        # Trend
        axes[0].plot(components['ds'], components['trend'], 'b-')
        axes[0].set_ylabel('Trend')
        axes[0].set_title('Trend Component')
        axes[0].grid(True, alpha=0.3)

        # Weekly seasonality
        if 'weekly' in components.columns:
            axes[1].plot(components['ds'], components['weekly'], 'g-')
            axes[1].set_ylabel('Weekly')
            axes[1].set_title('Weekly Seasonality')
            axes[1].grid(True, alpha=0.3)

        # Daily seasonality
        if 'daily' in components.columns:
            axes[2].plot(components['ds'], components['daily'], 'r-')
            axes[2].set_ylabel('Daily')
            axes[2].set_title('Daily Seasonality')
            axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        self.components_canvas.draw()
