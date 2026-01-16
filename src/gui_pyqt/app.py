"""
Crypto AI Predictor - Main PyQt6 Application

Unified GUI combining symbol list, charts, data fetching, AI training, and backtesting.
"""

import logging
import warnings
from typing import Optional, List, Dict
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTabWidget, QSplitter, QStatusBar, QMessageBox, QFrame, QApplication
)
from PyQt6.QtCore import Qt, QTimer

from src.core.config import AppConfig
from src.gui_pyqt.styles import apply_stylesheet, COLORS
from src.gui_pyqt.components import SymbolTable, ChartPanel, DataPanel, MLPanel, BacktestPanel
from src.gui_pyqt.workers import (
    MarketDataWorker, ChartDataWorker, DataFetchWorker,
    TrainingWorker, PredictionWorker, BacktestWorker
)
from src.api.binance_client import BinanceAPIClient
from src.data.manager import DataManager

# Suppress common PyTorch warnings
warnings.filterwarnings('ignore', message='.*triton not found.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

# ML imports (with fallback)
try:
    from src.ml.preprocessing.preprocessor import CryptoPreprocessor
    from src.ml.training.dataset import create_dataloaders
    from src.ml.models.model_config import TFTConfig
    from src.ml.training.trainer import TFTTrainer
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    ML_IMPORT_ERROR = str(e)

logger = logging.getLogger(__name__)


class CryptoAIPredictorApp(QMainWindow):
    """
    Main application window integrating all functionality.

    Tabs:
    1. Market Overview - Symbol table and price chart
    2. Data Pipeline - Historical data downloading
    3. AI Training - Model configuration and training
    4. Backtesting - Strategy backtesting
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle(f"{AppConfig.APP_NAME} v{AppConfig.VERSION}")
        self.setGeometry(100, 100, AppConfig.WINDOW_WIDTH, AppConfig.WINDOW_HEIGHT)

        # Apply dark theme
        apply_stylesheet(self)

        # API and Data clients
        self.api_client = BinanceAPIClient()
        self.data_manager = DataManager(
            storage_dir=AppConfig.DATASET_DIR,
            interval=AppConfig.DEFAULT_INTERVAL
        )

        # State
        self.market_data = []
        self._active_workers = []

        # Setup UI
        self._create_ui()
        self._connect_signals()

        # Load initial data
        QTimer.singleShot(100, self._load_market_data)

    def _create_ui(self):
        """Create the unified UI layout"""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header bar
        header = self._create_header()
        main_layout.addWidget(header)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        main_layout.addWidget(self.tabs, stretch=1)

        # Create tabs
        self._create_market_tab()
        self._create_data_tab()
        self._create_ml_tab()
        self._create_backtest_tab()

        # Status bar
        self._create_status_bar()

    def _create_header(self) -> QWidget:
        """Create application header"""
        header = QFrame()
        header.setStyleSheet(f"background-color: {COLORS['primary']};")
        header.setFixedHeight(60)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(25, 0, 25, 0)

        # App title
        title = QLabel(f"{AppConfig.APP_NAME}")
        title.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        # Version
        version = QLabel(f"v{AppConfig.VERSION}")
        version.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(version)

        layout.addStretch()

        return header

    def _create_market_tab(self):
        """Create Market Overview tab"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        # Splitter for table and chart
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Symbol table
        self.symbol_table = SymbolTable()
        self.symbol_table.setMaximumWidth(600)
        splitter.addWidget(self.symbol_table)

        # Right: Chart panel
        self.chart_panel = ChartPanel()
        splitter.addWidget(self.chart_panel)

        splitter.setSizes([500, 900])
        layout.addWidget(splitter)

        self.tabs.addTab(tab, "Market Overview")

    def _create_data_tab(self):
        """Create Data Pipeline tab"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Symbol table (compact)
        self.data_symbol_table = SymbolTable()
        self.data_symbol_table.setMaximumWidth(400)
        splitter.addWidget(self.data_symbol_table)

        # Right: Data panel
        self.data_panel = DataPanel()
        splitter.addWidget(self.data_panel)

        splitter.setSizes([350, 850])
        layout.addWidget(splitter)

        self.tabs.addTab(tab, "Data Pipeline")

    def _create_ml_tab(self):
        """Create AI Training tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        self.ml_panel = MLPanel()
        layout.addWidget(self.ml_panel)

        self.tabs.addTab(tab, "AI Training")

    def _create_backtest_tab(self):
        """Create Backtesting tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        self.backtest_panel = BacktestPanel()
        layout.addWidget(self.backtest_panel)

        self.tabs.addTab(tab, "Backtesting")

    def _create_status_bar(self):
        """Create status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label, stretch=1)

        self.symbol_count_label = QLabel("Symbols: 0")
        self.status_bar.addPermanentWidget(self.symbol_count_label)

    def _connect_signals(self):
        """Connect component signals"""
        # Symbol table signals
        self.symbol_table.chart_clicked.connect(self._on_chart_click)
        self.symbol_table.download_clicked.connect(self._on_download_click)
        self.symbol_table.refresh_btn.clicked.connect(self._load_market_data)

        # Data tab symbol table
        self.data_symbol_table.chart_clicked.connect(lambda d: self.data_panel.set_symbol(d))
        self.data_symbol_table.download_clicked.connect(lambda d: self.data_panel.set_symbol(d))

        # Data panel signals
        self.data_panel.fetch_requested.connect(self._on_fetch_data)

        # ML panel signals
        self.ml_panel.train_requested.connect(self._on_train_start)
        self.ml_panel.predict_requested.connect(self._on_predict_click)
        self.ml_panel.model_test_requested.connect(self._on_model_test_click)
        self.ml_panel.stop_training_requested.connect(self._on_stop_training)

        # Backtest panel signals
        self.backtest_panel.backtest_requested.connect(self._on_run_backtest)

    def _update_status(self, message: str):
        """Update status bar message"""
        self.status_label.setText(message)

    def _load_market_data(self):
        """Load market data from Binance"""
        self._update_status("Loading market data...")

        worker = MarketDataWorker(self.api_client)
        worker.result.connect(self._on_market_data_loaded)
        worker.error.connect(self._on_market_data_error)
        worker.finished.connect(lambda: self._cleanup_worker(worker))

        self._active_workers.append(worker)
        worker.start()

    def _on_market_data_loaded(self, data: List[Dict]):
        """Handle loaded market data"""
        self.market_data = data
        self.symbol_table.load_data(data)
        self.data_symbol_table.load_data(data)
        self.symbol_count_label.setText(f"Symbols: {len(data)}")
        self._update_status(f"Loaded {len(data)} symbols")

        # Update backtest panel with available symbols
        symbols = [d.get('symbol', '') for d in data]
        self.backtest_panel.set_symbols(symbols[:100])  # Limit to top 100

    def _on_market_data_error(self, error: str):
        """Handle market data loading error"""
        self._update_status(f"Error: {error}")
        logger.error(f"Market data error: {error}")

    def _on_chart_click(self, symbol_data: Dict):
        """Handle chart click"""
        symbol = symbol_data.get('symbol', 'Unknown')
        self._update_status(f"Loading chart for {symbol}...")

        worker = ChartDataWorker(self.api_client, symbol)
        worker.result.connect(lambda data: self._on_chart_data_loaded(symbol_data, data))
        worker.error.connect(self._on_chart_data_error)
        worker.finished.connect(lambda: self._cleanup_worker(worker))

        self._active_workers.append(worker)
        worker.start()

    def _on_chart_data_loaded(self, symbol_data: Dict, chart_data: List[Dict]):
        """Handle loaded chart data"""
        self.tabs.setCurrentIndex(0)  # Switch to Market tab
        self.chart_panel.show_chart(symbol_data, chart_data)
        self._update_status(f"Displaying chart for {symbol_data.get('symbol')}")

    def _on_chart_data_error(self, error: str):
        """Handle chart data loading error"""
        self._update_status(f"Chart error: {error}")
        QMessageBox.warning(self, "Chart Error", f"Failed to load chart: {error}")

    def _on_download_click(self, symbol_data: Dict):
        """Handle download click"""
        self.tabs.setCurrentIndex(1)  # Switch to Data Pipeline tab
        self.data_panel.set_symbol(symbol_data)
        self._update_status(f"Ready to fetch data for {symbol_data.get('symbol')}")

    def _on_fetch_data(self, symbol_data: Dict, max_candles: int):
        """Handle data fetch request"""
        symbol = symbol_data.get('symbol', '')
        self._update_status(f"Fetching {max_candles} candles for {symbol}...")
        self.data_panel.set_fetching(True)
        self.data_panel.log(f"Starting fetch for {symbol} ({max_candles} candles)")

        worker = DataFetchWorker(self.api_client, self.data_manager, symbol, max_candles)
        worker.progress.connect(self._on_fetch_progress)
        worker.result.connect(self._on_fetch_complete)
        worker.error.connect(self._on_fetch_error)
        worker.finished.connect(lambda: self._cleanup_worker(worker))

        self._active_workers.append(worker)
        worker.start()

    def _on_fetch_progress(self, current: int, total: int, message: str):
        """Handle fetch progress update"""
        self.data_panel.update_progress(current, total, message)
        self.data_panel.log(message)

    def _on_fetch_complete(self, result: Dict):
        """Handle fetch completion"""
        success = result.get('success', False)
        message = result.get('message', 'Unknown result')
        self.data_panel.complete_fetch(success, message)
        self._update_status(message)

        # Refresh ML panel datasets
        self.ml_panel.refresh_datasets()

    def _on_fetch_error(self, error: str):
        """Handle fetch error"""
        self.data_panel.complete_fetch(False, error)
        self._update_status(f"Fetch error: {error}")

    def _on_train_start(self, config: Dict, symbols: List[str]):
        """Handle training start request"""
        if not ML_AVAILABLE:
            QMessageBox.warning(self, "ML Not Available",
                              f"Machine learning modules are not available: {ML_IMPORT_ERROR}")
            return

        self._update_status(f"Starting training with {len(symbols)} symbols...")
        self.ml_panel.log_message(f"[TRAINING] Initializing with config: {config}")

        worker = TrainingWorker(config, symbols)
        worker.progress.connect(self._on_training_progress)
        worker.epoch_complete.connect(self._on_training_epoch)
        worker.result.connect(self._on_training_complete)
        worker.error.connect(self._on_training_error)
        worker.finished.connect(lambda: self._cleanup_worker(worker))

        self._training_worker = worker
        self._active_workers.append(worker)
        worker.start()

    def _on_training_progress(self, current: int, total: int, message: str):
        """Handle training progress"""
        self.ml_panel.update_progress(current, total, message)
        self.ml_panel.log_message(message)

    def _on_training_epoch(self, epoch: int, train_loss: float, val_loss: float, metrics: Dict):
        """Handle training epoch completion"""
        self.ml_panel.update_loss_curve(epoch, train_loss, val_loss)
        self.ml_panel.update_metrics(metrics)
        self.ml_panel.log_message(f"[EPOCH {epoch}] Train: {train_loss:.4f}, Val: {val_loss:.4f}")

    def _on_training_complete(self, result: Dict):
        """Handle training completion"""
        success = result.get('success', False)
        message = result.get('message', 'Training complete')
        model_path = result.get('model_path', '')

        self.ml_panel.set_training_complete(success, message)
        self._update_status(message)

        if success and model_path:
            self.ml_panel.log_message(f"[TRAINING] Model saved to: {model_path}")

    def _on_training_error(self, error: str):
        """Handle training error"""
        self.ml_panel.set_training_complete(False, error)
        self._update_status(f"Training error: {error}")

    def _on_stop_training(self):
        """Handle stop training request"""
        if hasattr(self, '_training_worker') and self._training_worker:
            self._training_worker.stop()
            self._update_status("Training stop requested...")

    def _on_predict_click(self, model_path: str, symbol: str):
        """Handle prediction request"""
        if not ML_AVAILABLE:
            QMessageBox.warning(self, "ML Not Available",
                              "Machine learning modules are not available")
            return

        self._update_status(f"Generating prediction for {symbol}...")

        worker = PredictionWorker(model_path, symbol, self.api_client)
        worker.result.connect(self._on_prediction_complete)
        worker.error.connect(self._on_prediction_error)
        worker.finished.connect(lambda: self._cleanup_worker(worker))

        self._active_workers.append(worker)
        worker.start()

    def _on_prediction_complete(self, result: Dict):
        """Handle prediction completion"""
        self.ml_panel.display_prediction(result)
        self._update_status("Prediction complete")

    def _on_prediction_error(self, error: str):
        """Handle prediction error"""
        self._update_status(f"Prediction error: {error}")
        QMessageBox.warning(self, "Prediction Error", error)

    def _on_model_test_click(self, model_path: str, symbol: str):
        """Handle model test request"""
        if not ML_AVAILABLE:
            QMessageBox.warning(self, "ML Not Available",
                              "Machine learning modules are not available")
            return

        self._update_status(f"Testing model on {symbol}...")
        self.ml_panel.log_message(f"[TEST] Starting model test on {symbol}")

        worker = PredictionWorker(model_path, symbol, self.api_client, test_mode=True)
        worker.result.connect(self._on_model_test_complete)
        worker.error.connect(self._on_model_test_error)
        worker.finished.connect(lambda: self._cleanup_worker(worker))

        self._active_workers.append(worker)
        worker.start()

    def _on_model_test_complete(self, result: Dict):
        """Handle model test completion"""
        self.ml_panel.display_test_results(result)
        self._update_status("Model test complete")

    def _on_model_test_error(self, error: str):
        """Handle model test error"""
        self._update_status(f"Test error: {error}")

    def _on_run_backtest(self, config: Dict):
        """Handle backtest request"""
        symbol = config.get('symbol', '')
        strategy = config.get('strategy', 'rsi')

        self._update_status(f"Running {strategy} backtest on {symbol}...")
        self.backtest_panel.set_running(True)
        self.backtest_panel.clear_results()

        worker = BacktestWorker(config, self.api_client, self.data_manager)
        worker.progress.connect(self._on_backtest_progress)
        worker.result.connect(self._on_backtest_complete)
        worker.error.connect(self._on_backtest_error)
        worker.finished.connect(lambda: self._cleanup_worker(worker))

        self._active_workers.append(worker)
        worker.start()

    def _on_backtest_progress(self, current: int, total: int, message: str):
        """Handle backtest progress"""
        self.backtest_panel.update_progress(current, total, message)

    def _on_backtest_complete(self, results):
        """Handle backtest completion"""
        self.backtest_panel.set_running(False)
        self.backtest_panel.display_results(results)
        self._update_status("Backtest complete")

    def _on_backtest_error(self, error: str):
        """Handle backtest error"""
        self.backtest_panel.set_running(False)
        self._update_status(f"Backtest error: {error}")
        QMessageBox.warning(self, "Backtest Error", error)

    def _cleanup_worker(self, worker):
        """Clean up finished worker"""
        if worker in self._active_workers:
            self._active_workers.remove(worker)

    def closeEvent(self, event):
        """Handle window close"""
        # Stop any active workers
        for worker in self._active_workers:
            if hasattr(worker, 'stop'):
                worker.stop()
            worker.quit()
            worker.wait(1000)

        event.accept()


def main():
    """Main entry point"""
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    app = QApplication(sys.argv)
    app.setApplicationName(AppConfig.APP_NAME)
    app.setApplicationVersion(AppConfig.VERSION)

    window = CryptoAIPredictorApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
