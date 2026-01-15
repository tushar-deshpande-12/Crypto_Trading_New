"""
Unified Crypto AI Predictor Application
Integrated GUI combining symbol list, charts, and data fetching
"""

import tkinter as tk
from tkinter import messagebox
import threading
import logging
import warnings
from typing import Optional
from pathlib import Path

from src.core.config import AppConfig
from src.gui.styles import setup_styles, get_color
from src.gui.components import SymbolTable, ChartPanel, DataPanel, MLPanel
from src.api.binance_client import BinanceAPIClient
from src.data.manager import DataManager

# Suppress common PyTorch warnings
warnings.filterwarnings('ignore', message='.*triton not found.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

# ML imports (with fallback if not installed)
try:
    from src.ml.preprocessing.preprocessor import CryptoPreprocessor
    from src.ml.training.dataset import create_dataloaders
    from src.ml.models.model_config import TFTConfig
    from src.ml.training.trainer import TFTTrainer
    from src.ml.training.model_evaluator import safe_direction_accuracy
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    ML_IMPORT_ERROR = str(e)

logger = logging.getLogger(__name__)


class CryptoAIPredictorApp:
    """
    Main application window integrating all functionality
    """

    def __init__(self, root: tk.Tk):
        """
        Initialize the application

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title(f"{AppConfig.APP_NAME} v{AppConfig.VERSION}")
        self.root.geometry(f"{AppConfig.WINDOW_WIDTH}x{AppConfig.WINDOW_HEIGHT}")
        self.root.configure(bg=get_color('bg_dark'))

        # API and Data clients
        self.api_client = BinanceAPIClient()
        self.data_manager = DataManager(
            storage_dir=AppConfig.DATASET_DIR,
            interval=AppConfig.DEFAULT_INTERVAL
        )

        # State
        self.market_data = []

        # Setup
        setup_styles()
        self._create_ui()
        self._load_market_data()

    def _create_ui(self):
        """Create the unified UI layout"""

        # Top menu bar
        self._create_menu_bar()

        # Main content area with notebook (tabs)
        content_frame = tk.Frame(self.root, bg=get_color('bg_dark'))
        content_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 0))

        # Create notebook for tabs
        from tkinter import ttk
        style = ttk.Style()
        style.theme_use('clam')

        # Configure notebook style
        style.configure('Custom.TNotebook',
                       background=get_color('bg_dark'),
                       borderwidth=0,
                       tabmargins=[0, 0, 0, 0])

        # Configure tab style
        style.configure('Custom.TNotebook.Tab',
                       background=get_color('bg_medium'),
                       foreground=get_color('text_primary'),
                       padding=[20, 10],
                       borderwidth=0,
                       font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold'))

        # Configure selected tab
        style.map('Custom.TNotebook.Tab',
                 background=[('selected', get_color('bg_light'))],
                 foreground=[('selected', 'white')],
                 expand=[('selected', [1, 1, 1, 0])])

        self.notebook = ttk.Notebook(content_frame, style='Custom.TNotebook')
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Market Overview (Symbol Table + Chart)
        market_tab = tk.Frame(self.notebook, bg=get_color('bg_dark'))
        self.notebook.add(market_tab, text='  📊 Market Overview  ')

        # Split market tab into left (table) and right (chart)
        left_panel = tk.Frame(market_tab, bg=get_color('bg_dark'), width=600)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 4), pady=0)
        left_panel.pack_propagate(False)

        self.symbol_table = SymbolTable(
            left_panel,
            on_download_click=self._on_download_click,
            on_chart_click=self._on_chart_click
        )
        self.symbol_table.pack(fill=tk.BOTH, expand=True)

        right_panel = tk.Frame(market_tab, bg=get_color('bg_dark'))
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0), pady=0)

        self.chart_panel = ChartPanel(right_panel)
        self.chart_panel.pack(fill=tk.BOTH, expand=True)

        # Tab 2: Data Pipeline
        data_tab = tk.Frame(self.notebook, bg=get_color('bg_dark'))
        self.notebook.add(data_tab, text='  📥 Data Pipeline  ')

        # Data panel with symbol selector on the side
        data_content = tk.Frame(data_tab, bg=get_color('bg_dark'))
        data_content.pack(fill=tk.BOTH, expand=True)

        # Left: compact symbol selector
        data_left = tk.Frame(data_content, bg=get_color('bg_dark'), width=400)
        data_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 4), pady=0)
        data_left.pack_propagate(False)

        self.data_symbol_table = SymbolTable(
            data_left,
            on_download_click=self._on_download_click,
            on_chart_click=None  # No chart in data tab
        )
        self.data_symbol_table.pack(fill=tk.BOTH, expand=True)

        # Right: data panel
        data_right = tk.Frame(data_content, bg=get_color('bg_dark'))
        data_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0), pady=0)

        self.data_panel = DataPanel(
            data_right,
            on_fetch_click=self._on_fetch_data
        )
        self.data_panel.pack(fill=tk.BOTH, expand=True)

        # Tab 3: AI Training
        ml_tab = tk.Frame(self.notebook, bg=get_color('bg_dark'))
        self.notebook.add(ml_tab, text='  🤖 AI Training  ')

        self.ml_panel = MLPanel(
            ml_tab,
            on_train_start=self._on_train_start,
            on_predict_click=self._on_predict_click,
            on_model_test_click=self._on_model_test_click
        )

        # Tab 4: Backtesting
        backtest_tab = tk.Frame(self.notebook, bg=get_color('bg_dark'))
        self.notebook.add(backtest_tab, text='  📊 Backtesting  ')

        from src.gui.components.backtest_panel import BacktestPanel
        self.backtest_panel = BacktestPanel(
            backtest_tab,
            on_run_backtest=self._on_run_backtest
        )

        # Status bar
        self._create_status_bar()

    def _create_menu_bar(self):
        """Create top menu bar"""
        menu_bar = tk.Frame(self.root, bg=get_color('primary'))
        menu_bar.pack(fill=tk.X)

        # App title
        tk.Label(
            menu_bar,
            text=f"📈 {AppConfig.APP_NAME}",
            bg=get_color('primary'),
            fg="white",
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_HEADER, 'bold')
        ).pack(side=tk.LEFT, padx=25, pady=18)

        # Version info
        tk.Label(
            menu_bar,
            text=f"v{AppConfig.VERSION}",
            bg=get_color('primary'),
            fg=get_color('text_secondary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_SMALL)
        ).pack(side=tk.LEFT, padx=(0, 25), pady=18)

    def _create_status_bar(self):
        """Create bottom status bar"""
        status_frame = tk.Frame(self.root, bg=get_color('bg_medium'))
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = tk.Label(
            status_frame,
            text="Ready",
            bg=get_color('bg_medium'),
            fg=get_color('text_secondary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_SMALL),
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, padx=20, pady=10)

        self.symbol_count_label = tk.Label(
            status_frame,
            text="Symbols: 0",
            bg=get_color('bg_medium'),
            fg=get_color('text_secondary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_SMALL)
        )
        self.symbol_count_label.pack(side=tk.RIGHT, padx=20, pady=10)


    def _load_market_data(self):
        """Load market data from Binance"""
        def load_thread():
            try:
                self._update_status("Loading market data...")
                logger.info("Fetching market data from Binance")

                market_data = self.api_client.get_usdt_pairs_detailed()

                if market_data:
                    self.market_data = market_data
                    # Load data into both symbol tables (Market tab and Data tab)
                    self.root.after(0, lambda: self.symbol_table.load_data(market_data))
                    self.root.after(0, lambda: self.data_symbol_table.load_data(market_data))
                    self.root.after(0, lambda: self.symbol_count_label.config(
                        text=f"Symbols: {len(market_data)}"
                    ))
                    self.root.after(0, lambda: self._update_status(
                        f"Loaded {len(market_data)} symbols"
                    ))
                    logger.info(f"Loaded {len(market_data)} symbols")
                else:
                    self.root.after(0, lambda: self._update_status("Failed to load data"))
                    logger.error("Failed to fetch market data")

            except Exception as e:
                logger.error(f"Error loading market data: {e}", exc_info=True)
                self.root.after(0, lambda: self._update_status(f"Error: {str(e)}"))

        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()

    def _on_download_click(self, symbol_data: dict):
        """Handle download button click from symbol table"""
        # Switch to data tab and set symbol
        self.notebook.select(1)  # Switch to Data Pipeline tab (index 1)
        self.data_panel.set_symbol(symbol_data)

        self._update_status(f"Ready to fetch data for {symbol_data.get('symbol')}")

    def _on_chart_click(self, symbol_data: dict):
        """Handle chart button click from symbol table"""
        symbol = symbol_data.get('symbol', 'Unknown')

        self._update_status(f"Loading chart for {symbol}...")

        def load_chart():
            try:
                # Fetch candlestick data
                chart_data = self.api_client.get_klines_formatted(
                    symbol=symbol,
                    interval="1h",
                    limit=500
                )

                if chart_data:
                    # Make sure we're on Market Overview tab and display chart
                    self.root.after(0, lambda: self.notebook.select(0))  # Market Overview tab
                    self.root.after(0, lambda: self.chart_panel.show_chart(
                        symbol_data,
                        chart_data
                    ))
                    self.root.after(0, lambda: self._update_status(
                        f"Displaying chart for {symbol}"
                    ))
                else:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Chart Error",
                        f"Failed to load chart data for {symbol}"
                    ))
                    self.root.after(0, lambda: self._update_status("Chart load failed"))

            except Exception as e:
                logger.error(f"Error loading chart: {e}", exc_info=True)
                self.root.after(0, lambda: messagebox.showerror(
                    "Chart Error",
                    f"Error: {str(e)}"
                ))

        thread = threading.Thread(target=load_chart, daemon=True)
        thread.start()

    def _on_fetch_data(self, symbol_data: dict, max_candles: int):
        """Handle data fetch request"""
        symbol = symbol_data.get('symbol', 'Unknown')

        self.data_panel.start_fetch()
        self._update_status(f"Fetching {max_candles} candles for {symbol}...")

        def fetch_thread():
            try:
                def progress_callback(current, total, message):
                    self.root.after(0, lambda: self.data_panel.update_progress(
                        current, total, message
                    ))

                dataset_path = self.data_manager.fetch_and_save(
                    symbol=symbol,
                    max_candles=max_candles,
                    progress_callback=progress_callback
                )

                if dataset_path:
                    self.root.after(0, lambda: self.data_panel.complete_fetch(
                        True,
                        f"Saved to {dataset_path.name}"
                    ))
                    self.root.after(0, lambda: self._update_status(
                        f"Successfully fetched {symbol} data"
                    ))
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Success",
                        f"Data saved successfully!\n\n{dataset_path}"
                    ))
                else:
                    self.root.after(0, lambda: self.data_panel.complete_fetch(
                        False,
                        "Fetch failed"
                    ))
                    self.root.after(0, lambda: self._update_status("Fetch failed"))

            except Exception as e:
                logger.error(f"Fetch error: {e}", exc_info=True)
                self.root.after(0, lambda: self.data_panel.complete_fetch(
                    False,
                    str(e)
                ))
                self.root.after(0, lambda: messagebox.showerror(
                    "Fetch Error",
                    f"Failed to fetch data:\n{str(e)}"
                ))

        thread = threading.Thread(target=fetch_thread, daemon=True)
        thread.start()

    def _update_status(self, message: str):
        """Update status bar message"""
        self.status_label.config(text=message)

    def _on_train_start(self, config: dict, symbols: list):
        """Handle training start"""
        print(f"\n[APP] Training started")
        print(f"[APP]   - Config: {config}")
        print(f"[APP]   - Symbols: {symbols}")

        # Check if ML modules are available
        if not ML_AVAILABLE:
            self.ml_panel.log_message(f"[APP] [X] PyTorch not installed!")
            self.ml_panel.log_message(f"[APP] Error: {ML_IMPORT_ERROR}")
            self.ml_panel.log_message(f"[APP] Install with: pip install torch pytorch-lightning==1.9.5 pytorch-forecasting")
            messagebox.showerror(
                "PyTorch Not Installed",
                f"PyTorch and dependencies are required for training.\n\n"
                f"Install with:\n"
                f"pip install torch pytorch-lightning==1.9.5 pytorch-forecasting\n\n"
                f"Error: {ML_IMPORT_ERROR}"
            )
            self.ml_panel.training_complete(False, "PyTorch not installed")
            return

        # Check PyTorch Lightning version compatibility
        try:
            import importlib.metadata
            pl_version = importlib.metadata.version('pytorch-lightning')
            pl_major = int(pl_version.split('.')[0])

            if pl_major >= 2:
                self.ml_panel.log_message(f"[APP] [!] PyTorch Lightning version incompatibility detected!")
                self.ml_panel.log_message(f"[APP] Current version: {pl_version}")
                self.ml_panel.log_message(f"[APP] Required: < 2.0.0")
                self.ml_panel.log_message(f"[APP] ")
                self.ml_panel.log_message(f"[APP] To fix, run:")
                self.ml_panel.log_message(f"[APP]   pip install pytorch-lightning==1.9.5")
                self.ml_panel.log_message(f"[APP] ")

                result = messagebox.askyesno(
                    "Version Incompatibility Warning",
                    f"PyTorch Lightning {pl_version} is incompatible with pytorch-forecasting.\n\n"
                    f"Required version: < 2.0.0\n"
                    f"Recommended: 1.9.5\n\n"
                    f"Training will likely fail. Continue anyway?",
                    icon='warning'
                )

                if not result:
                    self.ml_panel.log_message(f"[APP] Training cancelled by user")
                    self.ml_panel.training_complete(False, "Version incompatibility - please downgrade pytorch-lightning")
                    return
                else:
                    self.ml_panel.log_message(f"[APP] [!] Proceeding despite version warning...")
        except:
            pass  # Version check failed, proceed anyway

        # Extract configuration
        use_gpu = config.get('use_gpu', False)
        gpu_count = config.get('gpu_count', 0)
        model_arch = config.get('model_architecture', 'lstm')

        self.ml_panel.log_message(f"[TRAINING] Starting training pipeline...")
        self.ml_panel.log_message(f"[TRAINING] Model: {model_arch.upper()}")
        self.ml_panel.log_message(f"[TRAINING] Selected symbols: {', '.join(symbols)}")
        self.ml_panel.log_message(f"[TRAINING] Configuration: Batch={config.get('batch_size')}, Epochs={config.get('max_epochs')}")
        gpu_status = 'Enabled' if use_gpu else 'Disabled'
        self.ml_panel.log_message(f"[TRAINING] GPU: {gpu_status} ({gpu_count} GPU(s))")

        device_name = 'GPU' if use_gpu else 'CPU'
        self._update_status(f"Training model with {len(symbols)} symbols on {device_name}...")

        # Run training in background thread
        def training_thread():
            try:
                # Log start
                self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Initializing preprocessor..."))

                # Initialize preprocessor
                preprocessor = CryptoPreprocessor(dataset_dir=AppConfig.DATASET_DIR)

                # Use process_all - it does EVERYTHING (load, features, split, normalize, save scaler)
                self.root.after(0, lambda: self.ml_panel.log_message(f"[TRAINING] Running complete preprocessing pipeline..."))
                self.root.after(0, lambda: self.ml_panel.log_message(f"[TRAINING] Processing {len(symbols)} symbols..."))

                train_df, val_df, test_df = preprocessor.process_all(
                    symbols=symbols,
                    save_scaler_path="models/scalers.pkl"  # Automatically saves scaler
                )

                self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] OK Preprocessing complete (features, normalization, scaler saved)"))

                # Branch based on model architecture
                if model_arch == 'lstm':
                    # ========== LSTM TRAINING ==========
                    self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Using Simple LSTM model"))

                    from src.ml.models.lstm_model import SimpleLSTMTrainer

                    # Create LSTM trainer
                    # CLASSIFICATION MODEL with strong regularization:
                    # - hidden_size=32: Very simple to prevent overfitting
                    # - dropout=0.4: Strong regularization
                    # - num_layers=1: Single layer to reduce capacity
                    self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Creating LSTM CLASSIFIER..."))
                    self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Mode: Classification (UP/NEUTRAL/DOWN)"))
                    lstm_trainer = SimpleLSTMTrainer(
                        sequence_length=168,  # 1 week of hourly data
                        prediction_horizon=config.get('prediction_horizon', 1),
                        hidden_size=config.get('hidden_size', 32),  # Very small
                        num_layers=config.get('lstm_layers', 1),  # Single layer
                        dropout=config.get('dropout', 0.4),  # High dropout
                        learning_rate=config.get('learning_rate', 0.0005),
                        batch_size=config.get('batch_size', 128),  # Larger batches
                        max_epochs=config.get('max_epochs', 50),
                        early_stopping_patience=15,
                        checkpoint_dir="models/checkpoints/lstm",
                        verbose=True,
                        progress_callback=self.ml_panel.update_progress,
                        root=self.root
                    )

                    # Create dataloaders
                    self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Creating LSTM dataloaders..."))
                    train_loader, val_loader, test_loader = lstm_trainer.create_dataloaders(
                        train_df, val_df, test_df
                    )

                    # Train
                    self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Starting LSTM training..."))
                    self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] LR will auto-reduce after 3 stagnant epochs"))
                    self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Displaying: MAE, RMSE, R2, Loss on train & val"))

                    lstm_trainer.train(train_loader, val_loader, gpus=gpu_count)

                    # Save model
                    self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Saving LSTM model..."))
                    lstm_trainer.save_model("models/checkpoints/lstm/best_model.pt")

                    best_model_path = "models/checkpoints/lstm/best_model.pt"
                    self.root.after(0, lambda: self.ml_panel.log_message(f"[TRAINING] Model saved to {best_model_path}"))

                else:
                    # ========== TFT TRAINING ==========
                    self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Using TFT (Transformer) model"))

                    # Add time index for TFT
                    self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Adding time index..."))
                    train_df = preprocessor.add_time_index(train_df)
                    val_df = preprocessor.add_time_index(val_df)
                    test_df = preprocessor.add_time_index(test_df)

                    # Create TFT config
                    self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Creating model configuration..."))
                    tft_config = TFTConfig(
                    hidden_size=config.get('hidden_size', 160),
                    lstm_layers=config.get('lstm_layers', 2),
                    attention_head_size=config.get('attention_head_size', 4),
                    dropout=config.get('dropout', 0.15),
                    learning_rate=config.get('learning_rate', 0.0005),
                    batch_size=config.get('batch_size', 64),
                    max_epochs=config.get('max_epochs', 50),
                )

                    # Create dataloaders
                    self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Creating datasets and dataloaders..."))
                    train_loader, val_loader, test_loader = create_dataloaders(
                        train_df, val_df, test_df,
                        context_length=tft_config.max_encoder_length,
                        prediction_length=tft_config.max_prediction_length,
                        batch_size=tft_config.batch_size,
                        verbose=True
                    )

                    # Create trainer with GUI progress callback
                    self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Initializing trainer..."))
                    trainer = TFTTrainer(
                        config=tft_config,
                        checkpoint_dir="models/checkpoints",
                        log_dir="logs/training",
                        verbose=True,
                        progress_callback=self.ml_panel.update_progress,
                        root=self.root  # For thread-safe GUI updates
                    )

                    # Setup model
                    training_mode = config.get('training_mode', 'scratch')
                    pretrained_path = config.get('pretrained_path', None)

                    if training_mode == 'finetune' and pretrained_path:
                        self.root.after(0, lambda: self.ml_panel.log_message(f"[TRAINING] Loading pretrained model for fine-tuning: {Path(pretrained_path).name}"))
                        trainer.setup_model(train_loader.dataset, pretrained_path=pretrained_path)
                    else:
                        self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Setting up model from scratch..."))
                        trainer.setup_model(train_loader.dataset)

                    # Setup PyTorch Lightning trainer
                    self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Setting up PyTorch Lightning trainer..."))
                    self.root.after(0, lambda: self.ml_panel.log_message(f"[TRAINING] Device: {'GPU' if use_gpu else 'CPU'} ({gpu_count} GPU(s))"))
                    trainer.setup_trainer(gpus=gpu_count, enable_progress_bar=True, verbose_callbacks=True)

                    # Train
                    self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Starting training loop..."))
                    self.root.after(0, lambda: self.ml_panel.log_message(f"[TRAINING] This may take a while. Please be patient..."))

                    trainer.train(train_loader, val_loader)

                    # Save best model
                    self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Saving best model..."))
                    trainer.save_best_model("models/checkpoints/best_model.ckpt")

                    # Export metrics
                    self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Exporting metrics..."))
                    trainer.export_metrics("models/checkpoints")

                    # COMPREHENSIVE MODEL EVALUATION
                    self.root.after(0, lambda: self.ml_panel.log_message("[EVALUATION] Running comprehensive model evaluation..."))
                    try:
                        from src.ml.training.model_evaluator import evaluate_trained_model

                        # Evaluate on all splits
                        eval_metrics = evaluate_trained_model(
                            model=trainer.model_wrapper.model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            test_loader=test_loader,
                            save_dir="models/checkpoints/evaluation",
                            device='cuda' if use_gpu and gpu_count > 0 else 'cpu'
                        )

                        self.root.after(0, lambda: self.ml_panel.log_message("[EVALUATION] OK Evaluation complete!"))
                        self.root.after(0, lambda: self.ml_panel.log_message(f"[EVALUATION]   - Train MAE: {eval_metrics['train']['mae']:.4f}"))
                        self.root.after(0, lambda: self.ml_panel.log_message(f"[EVALUATION]   - Val MAE: {eval_metrics['val']['mae']:.4f}"))
                        self.root.after(0, lambda: self.ml_panel.log_message(f"[EVALUATION]   - Test MAE: {eval_metrics['test']['mae']:.4f}"))
                        self.root.after(0, lambda: self.ml_panel.log_message(f"[EVALUATION]   - Plots saved to: models/checkpoints/evaluation/"))

                    except Exception as e:
                        self.root.after(0, lambda: self.ml_panel.log_message(f"[EVALUATION] Warning: {e}"))
                        print(f"[APP] Evaluation error: {e}")

                    best_model_path = "models/checkpoints/best_model.ckpt"

                # Complete (common for both LSTM and TFT)
                self.root.after(0, lambda: self.ml_panel.training_complete(
                    True,
                    f"Training completed successfully! Model saved to {best_model_path}"
                ))
                self.root.after(0, lambda: self.ml_panel.set_model_path(best_model_path))
                self.root.after(0, lambda: self._update_status("Training completed successfully"))

            except Exception as e:
                logger.error(f"Training error: {e}", exc_info=True)
                error_msg = f"Training failed: {str(e)}"
                self.root.after(0, lambda: self.ml_panel.log_message(f"[TRAINING] [X] {error_msg}"))
                self.root.after(0, lambda: self.ml_panel.training_complete(False, error_msg))
                self.root.after(0, lambda: self._update_status("Training failed"))

        # Start training thread
        thread = threading.Thread(target=training_thread, daemon=True)
        thread.start()

    def _on_predict_click(self, model_path: str, symbol: str):
        """Handle prediction request"""
        print(f"\n[APP] Prediction requested")
        print(f"[APP]   - Model: {model_path}")
        print(f"[APP]   - Symbol: {symbol}")

        self.ml_panel.log_message(f"[PREDICTION] Generating 10-hour prediction for {symbol}...")
        self.ml_panel.log_message(f"[PREDICTION] Using model: {model_path}")

        if not ML_AVAILABLE:
            messagebox.showerror("Error", "ML libraries not available. Please install PyTorch and dependencies.")
            return

        # Run prediction in background thread
        def prediction_thread():
            try:
                from src.api.binance_client import BinanceAPIClient
                from src.ml.preprocessing.preprocessor import CryptoPreprocessor
                from src.ml.training.dataset import create_dataloaders
                from src.ml.models.tft_model import CryptoTFT
                from src.ml.models.model_config import TFTConfig
                import torch
                import pandas as pd
                import numpy as np

                # STEP 1: Fetch LIVE data from Binance
                self.root.after(0, lambda: self.ml_panel.log_message(f"[PREDICTION] Fetching live market data from Binance..."))

                binance_client = BinanceAPIClient()
                # Fetch enough data for context + lags
                # Model needs: max_encoder_length (168h) + max_prediction_length (10h) + max_lag (168h) = 346h
                # Fetch 500 to be safe
                live_klines = binance_client.get_klines_formatted(symbol=symbol, interval="1h", limit=500)

                if not live_klines:
                    raise ValueError(f"Failed to fetch live data for {symbol}")

                # Convert to DataFrame and match CSV column format EXACTLY
                live_df = pd.DataFrame(live_klines)

                # CRITICAL: Ensure datetime column is proper datetime type
                if 'timestamp' in live_df.columns:
                    # Binance API returns 'timestamp' as datetime object
                    live_df['datetime'] = pd.to_datetime(live_df['timestamp'])
                    # Keep numeric timestamp for features (milliseconds since epoch)
                    live_df['timestamp'] = (live_df['datetime'].astype('int64') // 10**6)  # Convert nanoseconds to milliseconds

                # Add symbol column
                live_df['symbol'] = symbol

                # Ensure all required columns exist (some might be renamed by Binance API)
                # The Binance API already provides all these columns correctly

                # Verify we have all essential OHLCV columns
                required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'symbol']
                missing = [col for col in required_cols if col not in live_df.columns]
                if missing:
                    raise ValueError(f"Missing required columns in live data: {missing}")

                # Capture current price BEFORE preprocessing (actual market price)
                current_price = float(live_df['close'].iloc[-1])
                current_timestamp = live_df['datetime'].iloc[-1] if 'datetime' in live_df.columns else 'N/A'

                # Extract 1 week (168 hours) of historical prices for visualization
                historical_prices = None
                try:
                    if len(live_df) >= 168:
                        historical_prices = live_df['close'].iloc[-168:].values.tolist()
                    else:
                        historical_prices = live_df['close'].values.tolist()
                except Exception as e:
                    print(f"[APP] Could not extract historical data: {e}")

                self.root.after(0, lambda p=current_price: self.ml_panel.log_message(
                    f"[PREDICTION] OK Fetched {len(live_df)} candles. Current price: ${p:.2f}"
                ))
                self.root.after(0, lambda t=current_timestamp: self.ml_panel.log_message(
                    f"[PREDICTION] Latest data timestamp: {t}"
                ))
                self.root.after(0, lambda: self.ml_panel.log_message(
                    f"[PREDICTION] OK Columns matched to training format ({len(live_df.columns)} columns)"
                ))

                # STEP 2: Load preprocessor with EXISTING scaler (from training)
                self.root.after(0, lambda: self.ml_panel.log_message(f"[PREDICTION] Loading scaler from training..."))

                preprocessor = CryptoPreprocessor(dataset_dir="dataset")

                # Try to load scaler from file
                scaler_path = "models/scalers.pkl"
                try:
                    preprocessor.load_scaler(scaler_path)
                    self.root.after(0, lambda: self.ml_panel.log_message(f"[PREDICTION] OK Loaded scaler from {scaler_path}"))
                except:
                    # Fallback: process training data to get scaler (only if scaler file doesn't exist)
                    self.root.after(0, lambda: self.ml_panel.log_message(f"[PREDICTION] Scaler file not found, generating from training data..."))
                    train_df, _, _ = preprocessor.process_all(symbols=[symbol])

                # STEP 3: Preprocess live data using TRAINING scaler (no refit!)
                self.root.after(0, lambda: self.ml_panel.log_message(f"[PREDICTION] Preprocessing live data..."))

                # Generate features (must match training pipeline!)
                live_df = preprocessor.generate_temporal_features(live_df)
                live_df = preprocessor.generate_technical_indicators(live_df)
                live_df = preprocessor.generate_lagged_features(live_df)
                live_df = preprocessor.generate_rolling_features(live_df)
                live_df = preprocessor.generate_volatility_features(live_df)

                # Remove NaN rows (from lag, rolling, and volatility features)
                live_df = live_df.dropna().reset_index(drop=True)

                # Normalize using EXISTING scaler (fit=False is critical!)
                live_df = preprocessor.normalize(live_df, fit=False)

                # Add time index for TFT
                live_df = preprocessor.add_time_index(live_df)

                self.root.after(0, lambda: self.ml_panel.log_message(f"[PREDICTION] Creating prediction dataset..."))

                # STEP 4: Load model first to get training dataset parameters
                self.root.after(0, lambda: self.ml_panel.log_message(f"[PREDICTION] Loading trained model..."))
                loaded_model = CryptoTFT.load_model(model_path, verbose=False)

                # Check if this is an LSTM model or TFT model
                from src.ml.models.lstm_model import SimpleLSTMTrainer
                is_lstm_model = isinstance(loaded_model, SimpleLSTMTrainer)

                if is_lstm_model:
                    # LSTM prediction path
                    self.root.after(0, lambda: self.ml_panel.log_message(f"[PREDICTION] Detected LSTM model"))
                    lstm_trainer = loaded_model

                    # Get the last sequence_length rows for prediction
                    sequence_length = lstm_trainer.sequence_length
                    if len(live_df) < sequence_length:
                        raise ValueError(f"Not enough data for prediction. Need {sequence_length} rows, have {len(live_df)}")

                    # Prepare input sequence - use centralized feature selection
                    from src.ml.models.lstm_model import get_lstm_feature_columns

                    model_feature_cols = getattr(lstm_trainer, 'feature_columns', None)
                    feature_columns = get_lstm_feature_columns(
                        live_df,
                        saved_feature_cols=model_feature_cols,
                        expected_input_size=lstm_trainer.input_size
                    )

                    self.root.after(0, lambda n=len(feature_columns), exp=lstm_trainer.input_size:
                        self.ml_panel.log_message(f"[PREDICTION] Using {n} features (model expects {exp})"))

                    # Get the last sequence_length rows as input
                    input_data = live_df.tail(sequence_length)[feature_columns].values.astype(np.float32)
                    input_tensor = torch.FloatTensor(input_data).unsqueeze(0)  # (1, seq_len, features)

                    self.root.after(0, lambda sl=sequence_length, nf=len(feature_columns):
                        self.ml_panel.log_message(f"[PREDICTION] Input: {sl} timesteps, {nf} features"))

                    # Run prediction
                    model = lstm_trainer.model
                    model.eval()
                    device = next(model.parameters()).device
                    input_tensor = input_tensor.to(device)

                    with torch.no_grad():
                        output = model(input_tensor)
                        # Handle dual output (regression, direction) from enhanced model
                        if isinstance(output, tuple):
                            reg_preds, dir_logits = output
                            pred_values = reg_preds.squeeze(0).cpu().numpy()
                            dir_probs = torch.sigmoid(dir_logits).squeeze(0).cpu().numpy()
                        else:
                            pred_values = output.squeeze(0).cpu().numpy()
                            dir_probs = (pred_values > 0).astype(float)

                    # Format predictions like TFT output
                    predictions = {
                        'prediction': pred_values,
                        'median': pred_values,
                        'direction_prob': dir_probs,  # Probability of positive return
                    }

                    self.root.after(0, lambda d=dir_probs[0] if len(dir_probs) > 0 else 0.5:
                        self.ml_panel.log_message(f"[PREDICTION] OK LSTM: direction prob={d:.2%}"))

                else:
                    # TFT prediction path
                    crypto_tft = loaded_model

                    # Get training dataset from model (TFT models store this internally)
                    if not hasattr(crypto_tft.model, 'dataset_parameters'):
                        raise ValueError("Model doesn't have training dataset parameters. Please retrain the model.")

                    # Create prediction dataset using from_parameters to match training config
                    from pytorch_forecasting import TimeSeriesDataSet

                    self.root.after(0, lambda: self.ml_panel.log_message(f"[PREDICTION] Creating dataset from model training parameters..."))

                    # Modify parameters to allow unknown categories
                    params = crypto_tft.model.dataset_parameters.copy()
                    if 'categorical_encoders' in params:
                        for key, encoder in params['categorical_encoders'].items():
                            if hasattr(encoder, 'add_nan'):
                                encoder.add_nan = True

                    prediction_dataset = TimeSeriesDataSet.from_parameters(
                        params,
                        live_df,
                        predict=True,  # Inference mode
                        stop_randomization=True  # No augmentation
                    )

                    # Create DataLoader
                    live_loader = prediction_dataset.to_dataloader(
                        train=False,  # Inference mode: no shuffling
                        batch_size=1,
                        num_workers=0
                    )

                    self.root.after(0, lambda: self.ml_panel.log_message(
                        f"[PREDICTION] OK Dataset created with {len(prediction_dataset)} samples"
                    ))

                    # STEP 5: Generate predictions (model already loaded)
                    self.root.after(0, lambda: self.ml_panel.log_message(f"[PREDICTION] Generating predictions from current market data..."))
                    predictions = crypto_tft.predict_next_n_hours(live_loader, n_hours=10)

                    self.root.after(0, lambda: self.ml_panel.log_message(f"[PREDICTION] OK Predictions generated!"))

                # STEP 6: Denormalize and display
                scaler = preprocessor.scalers.get(symbol)
                feature_columns = preprocessor.feature_columns

                # Display results in GUI (thread-safe) - use lambda defaults to capture values
                self.root.after(0, lambda p=predictions, s=symbol, sc=scaler, fc=feature_columns, cp=current_price, hp=historical_prices:
                               self.ml_panel.display_prediction(p, s, sc, fc, cp, hp))
                self.root.after(0, lambda sym=symbol: self.ml_panel.log_message(f"[PREDICTION] OK Prediction complete for {sym}"))

            except FileNotFoundError:
                error_msg = f"Model not found: {model_path}"
                self.root.after(0, lambda: self.ml_panel.log_message(f"[PREDICTION] X {error_msg}"))
                self.root.after(0, lambda: messagebox.showerror("Model Not Found", error_msg))
            except Exception as e:
                error_msg = f"Prediction failed: {str(e)}"
                self.root.after(0, lambda: self.ml_panel.log_message(f"[PREDICTION] X {error_msg}"))
                self.root.after(0, lambda: messagebox.showerror("Prediction Failed", error_msg))
                print(f"[APP] Prediction error: {e}")
                import traceback
                traceback.print_exc()

        # Start prediction thread
        thread = threading.Thread(target=prediction_thread, daemon=True)
        thread.start()

    def _on_model_test_click(self, model_path: str, symbol: str):
        """Handle model test request"""
        print(f"\n[APP] Model test requested")
        print(f"[APP]   - Model: {model_path}")
        print(f"[APP]   - Symbol: {symbol}")

        self.ml_panel.log_message(f"[TEST] Running model test on {symbol}...")
        self.ml_panel.log_message(f"[TEST] Using model: {model_path}")

        if not ML_AVAILABLE:
            messagebox.showerror("Error", "ML libraries not available. Please install PyTorch and dependencies.")
            return

        # Run test in background thread
        def test_thread():
            try:
                from src.ml.preprocessing.preprocessor import CryptoPreprocessor
                from src.ml.training.dataset import create_dataloaders
                from src.ml.models.tft_model import CryptoTFT
                from src.ml.models.model_config import TFTConfig
                import torch
                import numpy as np

                self.root.after(0, lambda: self.ml_panel.log_message(f"[TEST] Loading preprocessor and test data..."))

                # Load preprocessed data (from training)
                preprocessor = CryptoPreprocessor(dataset_dir="dataset")

                # Load scaler
                scaler_path = "models/scalers.pkl"
                try:
                    preprocessor.load_scaler(scaler_path)
                    self.root.after(0, lambda: self.ml_panel.log_message(f"[TEST] OK Loaded scaler"))
                except:
                    raise FileNotFoundError(f"Scaler not found. Please train a model first.")

                # Process data to get test set
                train_df, val_df, test_df = preprocessor.process_all(
                    symbols=[symbol],
                    save_scaler_path=scaler_path
                )

                self.root.after(0, lambda: self.ml_panel.log_message(f"[TEST] OK Test set has {len(test_df)} samples"))

                # Load model first to get training dataset parameters
                self.root.after(0, lambda: self.ml_panel.log_message(f"[TEST] Loading model..."))
                loaded_model = CryptoTFT.load_model(model_path, verbose=False)

                # Check if this is an LSTM model (SimpleLSTMTrainer) or TFT model (CryptoTFT)
                from src.ml.models.lstm_model import SimpleLSTMTrainer
                is_lstm_model = isinstance(loaded_model, SimpleLSTMTrainer)

                if is_lstm_model:
                    # LSTM needs more data - combine all splits for evaluation
                    lstm_trainer = loaded_model
                    min_required = lstm_trainer.sequence_length + lstm_trainer.prediction_horizon + 1

                    if len(test_df) < min_required:
                        # Combine val and test for more data
                        import pandas as pd
                        test_df = pd.concat([val_df, test_df], ignore_index=True)
                        test_df = test_df.sort_values('datetime').reset_index(drop=True)
                        self.root.after(0, lambda c=len(test_df):
                            self.ml_panel.log_message(f"[TEST] Combined val + test = {c} samples for LSTM"))

                    if len(test_df) < min_required:
                        # Still not enough - use all data
                        test_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
                        test_df = test_df.drop_duplicates(subset=['datetime', 'symbol']).sort_values('datetime').reset_index(drop=True)
                        self.root.after(0, lambda c=len(test_df):
                            self.ml_panel.log_message(f"[TEST] Using all {c} samples for LSTM evaluation"))

                if is_lstm_model:
                    # LSTM model testing path
                    self.root.after(0, lambda: self.ml_panel.log_message(f"[TEST] Detected LSTM model"))
                    lstm_trainer = loaded_model

                    # Create LSTM test dataset with centralized feature selection
                    from src.ml.models.lstm_model import SimpleLSTMDataset, get_lstm_feature_columns
                    from torch.utils.data import DataLoader

                    # Get feature columns using centralized function
                    model_feature_cols = getattr(lstm_trainer, 'feature_columns', None)
                    feature_columns = get_lstm_feature_columns(
                        test_df,
                        saved_feature_cols=model_feature_cols,
                        expected_input_size=lstm_trainer.input_size
                    )

                    test_dataset = SimpleLSTMDataset(
                        test_df,
                        sequence_length=lstm_trainer.sequence_length,
                        prediction_horizon=lstm_trainer.prediction_horizon,
                        feature_columns=feature_columns
                    )

                    self.root.after(0, lambda n=len(test_dataset), f=len(feature_columns), exp=lstm_trainer.input_size:
                        self.ml_panel.log_message(f"[TEST] LSTM dataset: {n} samples, {f} features (model expects {exp})"))

                    # Check if we have enough data
                    if len(test_dataset) == 0:
                        min_required = lstm_trainer.sequence_length + lstm_trainer.prediction_horizon + 1
                        raise ValueError(
                            f"Not enough test data for LSTM. Have {len(test_df)} rows, "
                            f"need at least {min_required} (sequence_length={lstm_trainer.sequence_length} + "
                            f"prediction_horizon={lstm_trainer.prediction_horizon} + 1)"
                        )

                    test_loader = DataLoader(
                        test_dataset,
                        batch_size=min(32, len(test_dataset)),
                        shuffle=False,
                        num_workers=0
                    )

                    # Run LSTM predictions
                    model = lstm_trainer.model
                    model.eval()
                    device = next(model.parameters()).device

                    all_predictions = []
                    all_actuals = []

                    self.root.after(0, lambda: self.ml_panel.log_message(f"[TEST] Processing {len(test_loader)} LSTM batches..."))

                    with torch.no_grad():
                        for batch_idx, (x, y) in enumerate(test_loader):
                            x = x.to(device)
                            y = y.to(device)

                            output = model(x)
                            # Handle dual output (regression, direction) from enhanced model
                            if isinstance(output, tuple):
                                preds, _ = output  # Use regression output
                            else:
                                preds = output
                            all_predictions.extend(preds[:, 0].cpu().numpy())  # First prediction
                            all_actuals.extend(y[:, 0].cpu().numpy())  # First target

                    predictions_arr = np.array(all_predictions)
                    actuals_arr = np.array(all_actuals)

                else:
                    # TFT model testing path
                    crypto_tft = loaded_model

                    # Check if model has dataset parameters
                    if not hasattr(crypto_tft.model, 'dataset_parameters'):
                        raise ValueError("Model doesn't have training dataset parameters. Please retrain the model.")

                    # Add time index for TFT
                    test_df = preprocessor.add_time_index(test_df)

                    # Create test dataset with SMALLER context to get more samples
                    from pytorch_forecasting import TimeSeriesDataSet

                    self.root.after(0, lambda: self.ml_panel.log_message(f"[TEST] Creating test dataset from model training parameters..."))

                    # Modify parameters to allow unknown categories and use smaller context
                    params = crypto_tft.model.dataset_parameters.copy()

                    # CRITICAL: Reduce encoder length for testing to get more samples
                    # Use a small encoder length to maximize number of test samples
                    original_encoder_length = params.get('max_encoder_length', 2000)
                    prediction_length = params.get('max_prediction_length', 10)

                    # Calculate how many samples we can get with different encoder lengths
                    # Each sample needs (encoder_length + prediction_length) rows
                    # Target at least 50 test samples, ideally 200+
                    available_rows = len(test_df)
                    target_samples = 100

                    # Calculate ideal encoder length to get target_samples
                    # available_rows = encoder_length + prediction_length + (target_samples - 1)
                    # encoder_length = available_rows - prediction_length - target_samples + 1
                    ideal_encoder_length = available_rows - prediction_length - target_samples + 1
                    test_encoder_length = max(24, min(168, ideal_encoder_length))  # Between 24 hours and 1 week

                    params['max_encoder_length'] = test_encoder_length

                    estimated_samples = max(1, available_rows - test_encoder_length - prediction_length + 1)
                    self.root.after(0, lambda orig=original_encoder_length, new=test_encoder_length, est=estimated_samples:
                        self.ml_panel.log_message(f"[TEST] Using encoder_length={new} (training used {orig}), estimated ~{est} samples"))

                    if 'categorical_encoders' in params:
                        for key, encoder in params['categorical_encoders'].items():
                            if hasattr(encoder, 'add_nan'):
                                encoder.add_nan = True

                    test_dataset = TimeSeriesDataSet.from_parameters(
                        params,
                        test_df,
                        predict=False,  # Use training-style mode to get more samples
                        stop_randomization=True  # No augmentation for consistent evaluation
                    )

                    self.root.after(0, lambda n=len(test_dataset):
                        self.ml_panel.log_message(f"[TEST] Dataset created with {n} valid sequences"))

                    # Create DataLoader with smaller batch size for more batches
                    config = TFTConfig()
                    test_loader = test_dataset.to_dataloader(
                        train=False,  # Inference mode: no shuffling
                        batch_size=min(32, config.batch_size),  # Smaller batches = more iterations
                        num_workers=0
                    )

                    self.root.after(0, lambda: self.ml_panel.log_message(f"[TEST] OK Test dataset created with {len(test_dataset)} samples"))

                    self.root.after(0, lambda: self.ml_panel.log_message(f"[TEST] Running predictions on test set..."))

                    # Generate predictions on test set
                    model = crypto_tft.model
                    model.eval()

                    # Detect model device
                    device = next(model.parameters()).device
                    self.root.after(0, lambda d=str(device): self.ml_panel.log_message(f"[TEST] Model device: {d}"))

                    all_predictions = []
                    all_actuals = []

                    self.root.after(0, lambda: self.ml_panel.log_message(f"[TEST] Processing {len(test_loader)} batches..."))

                    # Process ALL batches to get comprehensive test results
                    with torch.no_grad():
                        for batch_idx, batch in enumerate(test_loader):
                            # Batch is a tuple: (x, y) where x is dict of inputs, y is target
                            if isinstance(batch, (tuple, list)):
                                x, y = batch
                            else:
                                x = batch
                                y = None

                            # Move batch to model device
                            if isinstance(x, dict):
                                x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
                            elif isinstance(x, torch.Tensor):
                                x = x.to(device)

                            if y is not None:
                                if isinstance(y, torch.Tensor):
                                    y = y.to(device)
                                elif isinstance(y, (tuple, list)):
                                    y = tuple(yi.to(device) if isinstance(yi, torch.Tensor) else yi for yi in y)

                            # Get predictions - pass the input dict
                            outputs = model(x)

                            # Extract predictions
                            if hasattr(outputs, 'prediction'):
                                preds = outputs.prediction.cpu().numpy()
                            else:
                                preds = outputs[0].cpu().numpy() if isinstance(outputs, tuple) else outputs.cpu().numpy()

                            # Get actual values from target
                            if y is not None:
                                actuals = y[0].cpu().numpy() if isinstance(y, tuple) else y.cpu().numpy()
                            else:
                                # Fall back to getting from x dict if y not available
                                actuals = x['encoder_target'].cpu().numpy()

                            # Handle different output shapes
                            if len(preds.shape) == 3:
                                all_predictions.extend(preds[:, 0, 0])  # (batch, time, features)
                            elif len(preds.shape) == 2:
                                all_predictions.extend(preds[:, 0])  # (batch, time)
                            else:
                                all_predictions.extend(preds)

                            # Handle actual values
                            if len(actuals.shape) == 2:
                                all_actuals.extend(actuals[:, -1])  # Last encoder value
                            else:
                                all_actuals.extend(actuals)

                    # Convert to numpy arrays for TFT
                    predictions_arr = np.array(all_predictions)
                    actuals_arr = np.array(all_actuals)

                # Common metrics calculation for both LSTM and TFT
                self.root.after(0, lambda: self.ml_panel.log_message(f"[TEST] Calculating metrics..."))

                # Log shapes for debugging
                print(f"[TEST] Predictions shape: {predictions_arr.shape}, Actuals shape: {actuals_arr.shape}")
                self.root.after(0, lambda p=len(predictions_arr), a=len(actuals_arr):
                    self.ml_panel.log_message(f"[TEST] Collected {p} predictions and {a} actuals"))

                # Check if we have enough data
                if len(predictions_arr) < 10:
                    self.root.after(0, lambda: self.ml_panel.log_message(
                        f"[TEST] ⚠️ Warning: Only {len(predictions_arr)} samples collected. Results may not be representative."
                    ))
                    self.root.after(0, lambda: self.ml_panel.log_message(
                        "[TEST] Tip: Model needs sufficient context (encoder_length). Test set may be too small."
                    ))

                # Validate data
                if len(predictions_arr) == 0 or len(actuals_arr) == 0:
                    raise ValueError(f"No predictions generated. Predictions: {len(predictions_arr)}, Actuals: {len(actuals_arr)}")

                if len(predictions_arr) != len(actuals_arr):
                    # Truncate to minimum length
                    min_len = min(len(predictions_arr), len(actuals_arr))
                    predictions_arr = predictions_arr[:min_len]
                    actuals_arr = actuals_arr[:min_len]
                    self.root.after(0, lambda: self.ml_panel.log_message(f"[TEST] [!] Truncated to {min_len} samples"))

                # Denormalize predictions and actuals
                scaler = preprocessor.scalers.get(symbol)
                if scaler:
                    # Get the index of 'close' in feature columns
                    feature_columns = preprocessor.feature_columns
                    close_idx = feature_columns.index('close')

                    # Create dummy arrays with all features
                    dummy_pred = np.zeros((len(predictions_arr), len(feature_columns)))
                    dummy_actual = np.zeros((len(actuals_arr), len(feature_columns)))

                    # Set close values
                    dummy_pred[:, close_idx] = predictions_arr
                    dummy_actual[:, close_idx] = actuals_arr

                    # Inverse transform
                    pred_denorm = scaler.inverse_transform(dummy_pred)[:, close_idx]
                    actual_denorm = scaler.inverse_transform(dummy_actual)[:, close_idx]
                else:
                    pred_denorm = predictions_arr
                    actual_denorm = actuals_arr

                # Calculate metrics
                mae = np.mean(np.abs(pred_denorm - actual_denorm))
                rmse = np.sqrt(np.mean((pred_denorm - actual_denorm) ** 2))

                # Directional accuracy (using safe calculation to prevent NaN)
                directional_accuracy = safe_direction_accuracy(pred_denorm, actual_denorm)

                # Test loss (normalized)
                test_loss = np.mean((predictions_arr - actuals_arr) ** 2)

                metrics = {
                    'mae': mae,
                    'rmse': rmse,
                    'directional_accuracy': directional_accuracy,
                    'test_loss': test_loss
                }

                self.root.after(0, lambda: self.ml_panel.log_message(f"[TEST] ===== TESTING COMPLETE ====="))
                self.root.after(0, lambda n=len(predictions_arr): self.ml_panel.log_message(f"[TEST] Tested on {n} prediction points"))
                self.root.after(0, lambda: self.ml_panel.log_message(f"[TEST] MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, Dir Acc: {directional_accuracy:.1f}%"))

                # Calculate additional useful metrics
                avg_error_pct = (mae / np.mean(actual_denorm)) * 100
                self.root.after(0, lambda pct=avg_error_pct: self.ml_panel.log_message(f"[TEST] Average Error: {pct:.2f}% of price"))

                # Display results
                self.root.after(0, lambda m=metrics, a=actual_denorm, p=pred_denorm:
                               self.ml_panel.display_test_results(m, a.tolist(), p.tolist()))

            except FileNotFoundError as e:
                error_msg = str(e)
                self.root.after(0, lambda: self.ml_panel.log_message(f"[TEST] X {error_msg}"))
                self.root.after(0, lambda: messagebox.showerror("File Not Found", error_msg))
            except Exception as e:
                error_msg = f"Model test failed: {str(e)}"
                self.root.after(0, lambda: self.ml_panel.log_message(f"[TEST] X {error_msg}"))
                self.root.after(0, lambda: messagebox.showerror("Test Failed", error_msg))
                print(f"[APP] Test error: {e}")
                import traceback
                traceback.print_exc()

        # Start test thread
        thread = threading.Thread(target=test_thread, daemon=True)
        thread.start()

    def _on_run_backtest(self, config: dict):
        """Handle backtest request"""
        print(f"\n[APP] Backtest requested")
        print(f"[APP]   - Config: {config}")

        if not ML_AVAILABLE:
            messagebox.showerror("Error", "ML libraries not available. Please install PyTorch and dependencies.")
            return

        # Run backtest in background thread
        def backtest_thread():
            try:
                from src.ml.backtest import CryptoBacktester, BacktestConfig
                from src.ml.preprocessing.preprocessor import CryptoPreprocessor
                import torch
                import numpy as np
                import pandas as pd

                # Load model
                model_path = config['model_path']
                initial_capital = config['initial_capital']
                trade_fee = config['trade_fee']
                confidence_threshold = config['confidence_threshold']

                # Use unified model loading (auto-detects LSTM vs TFT)
                from src.ml.models.tft_model import CryptoTFT
                from src.ml.models.lstm_model import SimpleLSTMTrainer

                loaded_model = CryptoTFT.load_model(model_path, verbose=False)

                # Determine model type from loaded object
                if isinstance(loaded_model, SimpleLSTMTrainer):
                    model = loaded_model.model
                    model_type = 'lstm'
                    lstm_trainer = loaded_model
                else:
                    model = loaded_model.model
                    model_type = 'tft'
                    crypto_tft = loaded_model

                # Load preprocessed test data
                preprocessor = CryptoPreprocessor(dataset_dir="dataset")

                # Load scaler
                try:
                    preprocessor.load_scaler("models/scalers.pkl")
                except:
                    raise FileNotFoundError("Scaler not found. Please train a model first.")

                # Get test data (using first available symbol)
                symbols = ['ETHUSDT']  # TODO: Make this configurable
                train_df, val_df, test_df = preprocessor.process_all(
                    symbols=symbols,
                    save_scaler_path="models/scalers.pkl"
                )

                # For LSTM, combine data if test set is too small
                if model_type == 'lstm':
                    min_required = lstm_trainer.sequence_length + lstm_trainer.prediction_horizon + 1
                    print(f"[APP] LSTM needs {min_required} rows, test has {len(test_df)}")

                    if len(test_df) < min_required:
                        # Combine val and test
                        import pandas as pd
                        test_df = pd.concat([val_df, test_df], ignore_index=True)
                        test_df = test_df.sort_values('datetime').reset_index(drop=True)
                        print(f"[APP] Combined val + test = {len(test_df)} samples")

                    if len(test_df) < min_required:
                        # Still not enough - use all data
                        test_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
                        test_df = test_df.drop_duplicates(subset=['datetime', 'symbol']).sort_values('datetime').reset_index(drop=True)
                        print(f"[APP] Using all {len(test_df)} samples for LSTM backtest")

                # Generate predictions on test set
                model.eval()
                predictions_list = []
                actuals_list = []
                timestamps_list = []

                with torch.no_grad():
                    if model_type == 'lstm':
                        # LSTM predictions with centralized feature selection
                        from src.ml.models.lstm_model import SimpleLSTMDataset, get_lstm_feature_columns
                        import torch

                        # Get feature columns using centralized function
                        model_feature_cols = getattr(lstm_trainer, 'feature_columns', None)
                        feature_columns = get_lstm_feature_columns(
                            test_df,
                            saved_feature_cols=model_feature_cols,
                            expected_input_size=lstm_trainer.input_size
                        )

                        test_dataset = SimpleLSTMDataset(
                            test_df,
                            sequence_length=lstm_trainer.sequence_length,
                            prediction_horizon=lstm_trainer.prediction_horizon,
                            feature_columns=feature_columns
                        )

                        print(f"[APP] LSTM backtest: {len(test_dataset)} samples, {len(feature_columns)} features (model expects {lstm_trainer.input_size})")

                        # Check if we have enough data
                        if len(test_dataset) == 0:
                            min_required = lstm_trainer.sequence_length + lstm_trainer.prediction_horizon + 1
                            raise ValueError(
                                f"Not enough test data for LSTM backtest. Have {len(test_df)} rows, "
                                f"need at least {min_required} (sequence_length={lstm_trainer.sequence_length} + "
                                f"prediction_horizon={lstm_trainer.prediction_horizon} + 1)"
                            )

                        device = next(model.parameters()).device

                        for i in range(min(len(test_dataset), 500)):  # Limit to 500 samples
                            x, y = test_dataset[i]
                            x_tensor = x.unsqueeze(0).to(device)  # Add batch dimension
                            output = model(x_tensor)  # Forward pass

                            # Handle dual output (regression, direction) from enhanced model
                            if isinstance(output, tuple):
                                pred, _ = output  # Use regression output
                            else:
                                pred = output
                            pred_value = pred.squeeze(0).cpu().numpy()

                            predictions_list.append(pred_value[0])  # First hour prediction
                            actuals_list.append(y[0].numpy())  # First hour actual

                            # Get timestamp
                            idx = test_dataset.valid_indices[i]
                            if 'datetime' in test_df.columns:
                                timestamps_list.append(test_df.iloc[idx]['datetime'])
                    else:
                        # TFT predictions
                        from pytorch_forecasting import TimeSeriesDataSet

                        # Check if model has dataset parameters
                        if not hasattr(model, 'dataset_parameters'):
                            raise ValueError("Model doesn't have training dataset parameters. Please retrain the model.")

                        # Add time_idx for TFT
                        test_df = preprocessor.add_time_index(test_df)

                        # Create test dataset using model's training parameters with smaller context
                        # Modify parameters to allow unknown categories and use smaller context
                        params = model.dataset_parameters.copy()

                        # CRITICAL: Reduce encoder length for backtesting to get MORE samples
                        original_encoder_length = params.get('max_encoder_length', 2000)
                        test_encoder_length = min(168, len(test_df) // 10)  # Use 1 week or 10% of data
                        params['max_encoder_length'] = test_encoder_length

                        print(f"[APP] Backtest using encoder_length={test_encoder_length} (training used {original_encoder_length})")

                        # Allow unknown categories for static categoricals (e.g., symbol)
                        if 'categorical_encoders' in params:
                            for key, encoder in params['categorical_encoders'].items():
                                if hasattr(encoder, 'add_nan'):
                                    encoder.add_nan = True

                        test_dataset = TimeSeriesDataSet.from_parameters(
                            params,
                            test_df,
                            predict=True,  # Inference mode
                            stop_randomization=True  # No augmentation
                        )

                        print(f"[APP] Backtest dataset created with {len(test_dataset)} samples")

                        # Create DataLoader
                        test_loader = test_dataset.to_dataloader(
                            train=False,  # Inference mode: no shuffling
                            batch_size=32,
                            num_workers=0
                        )

                        for batch_idx, batch in enumerate(test_loader):
                            if batch_idx >= 200:  # Limit to 200 batches (6400 samples max)
                                break

                            # Extract batch data
                            if isinstance(batch, (tuple, list)):
                                x, y = batch
                            else:
                                x = batch
                                y = None

                            # Move to device if needed
                            device = next(model.parameters()).device
                            if isinstance(x, dict):
                                x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}

                            # Get predictions
                            outputs = model(x)

                            # Extract predictions
                            if hasattr(outputs, 'prediction'):
                                preds = outputs.prediction.cpu().numpy()
                            else:
                                preds = outputs[0].cpu().numpy() if isinstance(outputs, tuple) else outputs.cpu().numpy()

                            # Get actual values
                            if y is not None:
                                actuals = y[0].cpu().numpy() if isinstance(y, tuple) else y.cpu().numpy()
                            else:
                                actuals = x['encoder_target'].cpu().numpy()

                            # Handle different output shapes - take first hour prediction
                            if len(preds.shape) == 3:
                                batch_preds = preds[:, 0, 0]  # (batch, time, features)
                            elif len(preds.shape) == 2:
                                batch_preds = preds[:, 0]  # (batch, time)
                            else:
                                batch_preds = preds

                            # Handle actual values
                            if len(actuals.shape) == 3:
                                batch_actuals = actuals[:, 0, 0]
                            elif len(actuals.shape) == 2:
                                batch_actuals = actuals[:, 0]
                            else:
                                batch_actuals = actuals

                            predictions_list.extend(batch_preds)
                            actuals_list.extend(batch_actuals)

                            # Get timestamps (approximate from test_df)
                            for i in range(len(batch_preds)):
                                idx = batch_idx * 32 + i
                                if idx < len(test_df) and 'datetime' in test_df.columns:
                                    timestamps_list.append(test_df.iloc[idx]['datetime'])

                predictions = np.array(predictions_list)
                actuals = np.array(actuals_list)

                # Validate we have predictions
                if len(predictions) == 0:
                    raise ValueError("No predictions generated. Check if test data has enough samples.")

                # Handle timestamps
                if timestamps_list:
                    timestamps = pd.DatetimeIndex(timestamps_list)
                    print(f"[APP] Backtest collected {len(predictions)} prediction points")
                    print(f"[APP] Date range: {timestamps[0]} to {timestamps[-1]}")
                else:
                    # Create synthetic timestamps if not available
                    timestamps = pd.date_range(start='2024-01-01', periods=len(predictions), freq='h')
                    print(f"[APP] Backtest collected {len(predictions)} prediction points (synthetic timestamps)")

                # Denormalize predictions and actuals for backtesting
                # (Backtester needs real prices, not normalized values)
                scaler = preprocessor.scalers.get(symbols[0])
                if scaler:
                    feature_columns = preprocessor.feature_columns
                    close_idx = feature_columns.index('close')

                    # Create dummy arrays with all features
                    dummy_pred = np.zeros((len(predictions), len(feature_columns)))
                    dummy_actual = np.zeros((len(actuals), len(feature_columns)))

                    # Set close values
                    dummy_pred[:, close_idx] = predictions
                    dummy_actual[:, close_idx] = actuals

                    # Inverse transform to get real prices
                    predictions_denorm = scaler.inverse_transform(dummy_pred)[:, close_idx]
                    actuals_denorm = scaler.inverse_transform(dummy_actual)[:, close_idx]

                    print(f"[APP] Denormalized predictions: {predictions_denorm[:5]}")
                    print(f"[APP] Denormalized actuals: {actuals_denorm[:5]}")
                else:
                    predictions_denorm = predictions
                    actuals_denorm = actuals
                    print(f"[APP] Warning: No scaler found, using normalized values")

                # Run backtest
                backtest_config = BacktestConfig(
                    initial_capital=initial_capital,
                    trade_fee=trade_fee,
                    confidence_threshold=confidence_threshold
                )

                backtester = CryptoBacktester(backtest_config, verbose=True)
                results = backtester.run_backtest(predictions_denorm, actuals_denorm, timestamps)

                # Display results in GUI
                results_dict = {
                    'initial_capital': results.initial_capital,
                    'final_capital': results.final_capital,
                    'total_return': results.total_return,
                    'total_return_pct': results.total_return_pct,
                    'num_trades': results.num_trades,
                    'win_rate': results.win_rate,
                    'profit_loss_ratio': results.profit_loss_ratio,
                    'prediction_mae': results.prediction_mae,
                    'prediction_rmse': results.prediction_rmse,
                    'prediction_r2': results.prediction_r2,
                    'direction_accuracy': results.direction_accuracy
                }

                self.root.after(0, lambda: self.backtest_panel.display_results(results_dict))

                # Create and save charts
                # Note: plot_results saves to file and returns closed figure
                # We'll load the image file to display in GUI instead
                plot_path = "backtest_results.png"
                backtester.plot_results(results, save_path=plot_path)

                # Display chart by loading the saved image
                self.root.after(0, lambda: self.backtest_panel.update_chart_from_file(plot_path))

            except Exception as e:
                error_msg = f"Backtest failed: {str(e)}"
                print(f"[APP] {error_msg}")
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: messagebox.showerror("Backtest Error", error_msg))

        # Start backtest thread
        thread = threading.Thread(target=backtest_thread, daemon=True)
        thread.start()

    def run(self):
        """Start the application"""
        self.root.mainloop()

    def cleanup(self):
        """Cleanup resources"""
        if self.api_client:
            self.api_client.close()
        if self.data_manager:
            self.data_manager.close()
