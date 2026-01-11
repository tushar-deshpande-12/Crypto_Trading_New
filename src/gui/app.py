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
            on_predict_click=self._on_predict_click
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
            self.ml_panel.log_message(f"[APP] ✗ PyTorch not installed!")
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
                self.ml_panel.log_message(f"[APP] ⚠ PyTorch Lightning version incompatibility detected!")
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
                    self.ml_panel.log_message(f"[APP] ⚠ Proceeding despite version warning...")
        except:
            pass  # Version check failed, proceed anyway

        # Extract GPU configuration
        use_gpu = config.get('use_gpu', False)
        gpu_count = config.get('gpu_count', 0)

        self.ml_panel.log_message(f"[TRAINING] Starting training pipeline...")
        self.ml_panel.log_message(f"[TRAINING] Selected symbols: {', '.join(symbols)}")
        self.ml_panel.log_message(f"[TRAINING] Configuration: Batch={config.get('batch_size')}, Epochs={config.get('max_epochs')}")
        self.ml_panel.log_message(f"[TRAINING] GPU: {'Enabled' if use_gpu else 'Disabled'} ({gpu_count} GPU(s))")

        self._update_status(f"Training model with {len(symbols)} symbols on {'GPU' if use_gpu else 'CPU'}...")

        # Run training in background thread
        def training_thread():
            try:
                # Log start
                self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Initializing preprocessor..."))

                # Initialize preprocessor
                preprocessor = CryptoPreprocessor(dataset_dir=AppConfig.DATASET_DIR)

                # Load data
                self.root.after(0, lambda: self.ml_panel.log_message(f"[TRAINING] Loading data for {len(symbols)} symbols..."))
                combined_df = preprocessor.load_multi_symbol_data(symbols)

                # Generate features
                self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Generating features..."))
                combined_df = preprocessor.generate_temporal_features(combined_df)
                combined_df = preprocessor.generate_technical_indicators(combined_df)
                combined_df = preprocessor.generate_lagged_features(combined_df)

                # Handle missing values
                self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Handling missing values..."))
                combined_df = preprocessor.handle_missing_values(combined_df)

                # Add time index (required by pytorch-forecasting)
                self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Adding time index..."))
                combined_df = preprocessor.add_time_index(combined_df)

                # Split data
                self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Splitting data..."))
                train_df, val_df, test_df = preprocessor.split_data(
                    combined_df,
                    train_ratio=config.get('train_split', 0.7),
                    val_ratio=config.get('val_split', 0.15)
                )

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
                self.root.after(0, lambda: self.ml_panel.log_message("[TRAINING] Setting up model from dataset..."))
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

                # Complete
                self.root.after(0, lambda: self.ml_panel.training_complete(
                    True,
                    f"Training completed successfully! Model saved to models/checkpoints/best_model.ckpt"
                ))
                self.root.after(0, lambda: self._update_status("Training completed successfully"))

            except Exception as e:
                logger.error(f"Training error: {e}", exc_info=True)
                error_msg = f"Training failed: {str(e)}"
                self.root.after(0, lambda: self.ml_panel.log_message(f"[TRAINING] ✗ {error_msg}"))
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

        self.ml_panel.log_message(f"[APP] Prediction requested for {symbol}")
        self.ml_panel.log_message(f"[APP] Using model: {model_path}")

        # Note: Actual prediction would be implemented here
        # For now, this is a placeholder that demonstrates the callback
        messagebox.showinfo(
            "Prediction Requested",
            f"Generating 10-hour prediction for {symbol}\n\n"
            f"Model: {model_path}\n\n"
            f"Note: Full prediction implementation requires trained model and PyTorch"
        )

        self.ml_panel.log_message(f"[APP] ⚠ Prediction implementation requires trained model and PyTorch")

    def run(self):
        """Start the application"""
        self.root.mainloop()

    def cleanup(self):
        """Cleanup resources"""
        if self.api_client:
            self.api_client.close()
        if self.data_manager:
            self.data_manager.close()
