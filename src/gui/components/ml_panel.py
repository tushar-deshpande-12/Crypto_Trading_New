"""
ML Panel Component for AI Model Training and Prediction
Provides GUI for dataset selection, model configuration, training, and prediction
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
from pathlib import Path
from typing import Optional, Callable, Dict, List, Any
import json

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from src.core.config import AppConfig
from src.gui.styles import get_color


class MLPanel:
    """
    Machine Learning Panel for model training and prediction

    4 Sections:
    1. Dataset Configuration - Select datasets and train/val/test split
    2. Model Configuration - Hyperparameters and settings
    3. Training Progress - Live metrics, loss curves, logs
    4. Prediction & Inference - Generate and visualize predictions
    """

    def __init__(
        self,
        parent: tk.Frame,
        on_train_start: Optional[Callable] = None,
        on_predict_click: Optional[Callable] = None
    ):
        """
        Initialize ML Panel

        Args:
            parent: Parent frame
            on_train_start: Callback when training starts(config, symbols)
            on_predict_click: Callback when prediction requested(model_path, symbol)
        """
        print(f"\n[ML_PANEL] Initializing ML Panel...")

        self.parent = parent
        self.on_train_start = on_train_start
        self.on_predict_click = on_predict_click

        # State
        self.selected_symbols = []
        self.available_datasets = {}
        self.training_active = False
        self.progress_queue = queue.Queue()

        # Training metrics history
        self.train_loss_history = []
        self.val_loss_history = []
        self.epochs = []

        # Create main frame
        self.frame = tk.Frame(parent, bg=get_color('bg_dark'))
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Create UI
        self._create_ui()

        # Scan for available datasets
        self._scan_datasets()

        print(f"[ML_PANEL] ✓ ML Panel initialized")

    def _create_ui(self):
        """Create all UI sections"""
        print(f"[ML_PANEL] Creating UI sections...")

        # Create scrollable canvas
        canvas = tk.Canvas(self.frame, bg=get_color('bg_dark'), highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=get_color('bg_dark'))

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Header
        self._create_header(scrollable_frame)

        # Section 1: Dataset Configuration
        self._create_dataset_section(scrollable_frame)

        # Section 2: Model Configuration
        self._create_model_config_section(scrollable_frame)

        # Section 3: Training Progress
        self._create_training_progress_section(scrollable_frame)

        # Section 4: Prediction
        self._create_prediction_section(scrollable_frame)

        print(f"[ML_PANEL] ✓ UI sections created")

    def _create_header(self, parent):
        """Create panel header"""
        header_frame = tk.Frame(parent, bg=get_color('bg_medium'))
        header_frame.pack(fill=tk.X, padx=0, pady=0)

        title = tk.Label(
            header_frame,
            text="🤖 AI Model Training & Prediction",
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_HEADER, 'bold'),
            bg=get_color('bg_medium'),
            fg=get_color('text_primary')
        )
        title.pack(pady=(20, 5), padx=20)

        subtitle = tk.Label(
            header_frame,
            text="Temporal Fusion Transformer for multi-asset cryptocurrency forecasting",
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL),
            bg=get_color('bg_medium'),
            fg=get_color('text_secondary')
        )
        subtitle.pack(pady=(0, 20), padx=20)

    def _create_dataset_section(self, parent):
        """Create dataset configuration section"""
        print(f"[ML_PANEL]   - Creating dataset section...")

        section = tk.LabelFrame(
            parent,
            text=" 📊 Dataset Configuration ",
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_LARGE, 'bold'),
            bg=get_color('bg_medium'),
            fg=get_color('text_primary'),
            padx=20,
            pady=15,
            borderwidth=0,
            relief='flat'
        )
        section.pack(fill=tk.X, padx=15, pady=(0, 12))

        # Dataset selection
        tk.Label(
            section,
            text="Select Datasets for Training:",
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold'),
            bg=get_color('bg_medium'),
            fg=get_color('text_primary')
        ).pack(anchor=tk.W, pady=(0, 5))

        # Checkboxes frame
        self.dataset_checkboxes_frame = tk.Frame(section, bg=get_color('bg_medium'))
        self.dataset_checkboxes_frame.pack(fill=tk.X, pady=5)

        # Train/Val/Test split
        split_frame = tk.Frame(section, bg=get_color('bg_medium'))
        split_frame.pack(fill=tk.X, pady=(10, 0))

        tk.Label(
            split_frame,
            text="Data Split Ratios:",
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold'),
            bg=get_color('bg_medium'),
            fg=get_color('text_primary')
        ).pack(anchor=tk.W)

        # Split sliders
        split_controls = tk.Frame(split_frame, bg=get_color('bg_medium'))
        split_controls.pack(fill=tk.X, pady=5)

        # Train split
        train_frame = tk.Frame(split_controls, bg=get_color('bg_medium'))
        train_frame.pack(fill=tk.X, pady=2)

        tk.Label(train_frame, text="Train:", width=8, anchor=tk.W,
                bg=get_color('bg_medium'), fg=get_color('text_primary')).pack(side=tk.LEFT)

        self.train_split_var = tk.DoubleVar(value=0.70)
        self.train_split_label = tk.Label(train_frame, text="70%", width=6,
                                          bg=get_color('bg_medium'), fg=get_color('accent'))
        self.train_split_label.pack(side=tk.RIGHT)

        tk.Scale(train_frame, from_=0.5, to=0.8, resolution=0.05, orient=tk.HORIZONTAL,
                variable=self.train_split_var, showvalue=0, bg=get_color('bg_medium'),
                fg=get_color('text_primary'), highlightthickness=0,
                command=self._update_split_labels).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Val split
        val_frame = tk.Frame(split_controls, bg=get_color('bg_medium'))
        val_frame.pack(fill=tk.X, pady=2)

        tk.Label(val_frame, text="Val:", width=8, anchor=tk.W,
                bg=get_color('bg_medium'), fg=get_color('text_primary')).pack(side=tk.LEFT)

        self.val_split_var = tk.DoubleVar(value=0.15)
        self.val_split_label = tk.Label(val_frame, text="15%", width=6,
                                        bg=get_color('bg_medium'), fg=get_color('accent'))
        self.val_split_label.pack(side=tk.RIGHT)

        tk.Scale(val_frame, from_=0.1, to=0.3, resolution=0.05, orient=tk.HORIZONTAL,
                variable=self.val_split_var, showvalue=0, bg=get_color('bg_medium'),
                fg=get_color('text_primary'), highlightthickness=0,
                command=self._update_split_labels).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Test split (auto-calculated)
        test_frame = tk.Frame(split_controls, bg=get_color('bg_medium'))
        test_frame.pack(fill=tk.X, pady=2)

        tk.Label(test_frame, text="Test:", width=8, anchor=tk.W,
                bg=get_color('bg_medium'), fg=get_color('text_primary')).pack(side=tk.LEFT)

        self.test_split_label = tk.Label(test_frame, text="15%", width=6,
                                         bg=get_color('bg_medium'), fg=get_color('accent'))
        self.test_split_label.pack(side=tk.RIGHT)

    def _create_model_config_section(self, parent):
        """Create model configuration section"""
        print(f"[ML_PANEL]   - Creating model config section...")

        section = tk.LabelFrame(
            parent,
            text=" ⚙️ Model Configuration ",
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_LARGE, 'bold'),
            bg=get_color('bg_medium'),
            fg=get_color('text_primary'),
            padx=20,
            pady=15,
            borderwidth=0,
            relief='flat'
        )
        section.pack(fill=tk.X, padx=15, pady=(0, 12))

        # GPU Configuration
        gpu_frame = tk.Frame(section, bg=get_color('bg_medium'))
        gpu_frame.pack(fill=tk.X, pady=(0, 10))

        # Detect GPUs
        self.gpu_available = False
        self.gpu_count = 0
        self.gpu_info = "No GPU detected"

        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_available = True
                self.gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if self.gpu_count > 0 else "Unknown"
                self.gpu_info = f"{self.gpu_count} GPU(s) available: {gpu_name}"
            else:
                self.gpu_info = "CUDA not available - CPU training only"
        except ImportError:
            self.gpu_info = "PyTorch not installed"
        except Exception as e:
            self.gpu_info = f"GPU detection failed: {str(e)}"

        # GPU status header frame
        gpu_status_frame = tk.Frame(gpu_frame, bg=get_color('bg_medium'))
        gpu_status_frame.pack(fill=tk.X, pady=(0, 5))

        # GPU status label
        self.gpu_status_label = tk.Label(
            gpu_status_frame,
            text=f"🎮 GPU Status: {self.gpu_info}",
            bg=get_color('bg_medium'),
            fg=get_color('accent') if self.gpu_available else get_color('text_secondary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_SMALL, 'bold')
        )
        self.gpu_status_label.pack(side=tk.LEFT, anchor=tk.W)

        # GPU memory info (if available)
        if self.gpu_available:
            try:
                import torch
                gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
                gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.gpu_mem_label = tk.Label(
                    gpu_status_frame,
                    text=f"  •  Memory: {gpu_mem_allocated:.2f}GB / {gpu_mem_total:.2f}GB",
                    bg=get_color('bg_medium'),
                    fg=get_color('text_secondary'),
                    font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_SMALL)
                )
                self.gpu_mem_label.pack(side=tk.LEFT, padx=(10, 0))

                # Refresh button
                refresh_gpu_btn = tk.Button(
                    gpu_status_frame,
                    text="↻",
                    command=self._refresh_gpu_status,
                    bg=get_color('bg_light'),
                    fg=get_color('text_primary'),
                    font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_SMALL),
                    padx=5,
                    pady=2,
                    relief=tk.FLAT,
                    borderwidth=0,
                    cursor="hand2"
                )
                refresh_gpu_btn.pack(side=tk.LEFT, padx=(5, 0))
            except:
                self.gpu_mem_label = None
        else:
            self.gpu_mem_label = None

        # GPU enable checkbox
        self.use_gpu_var = tk.BooleanVar(value=self.gpu_available)
        gpu_checkbox = tk.Checkbutton(
            gpu_frame,
            text=f"Use GPU for Training ({self.gpu_count} GPU(s) available)" if self.gpu_available else "Use GPU (Not available)",
            variable=self.use_gpu_var,
            bg=get_color('bg_medium'),
            fg=get_color('text_primary'),
            selectcolor=get_color('bg_light'),
            activebackground=get_color('bg_medium'),
            activeforeground=get_color('text_primary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL),
            state=tk.NORMAL if self.gpu_available else tk.DISABLED
        )
        gpu_checkbox.pack(anchor=tk.W, pady=(0, 2))

        # GPU note
        if self.gpu_available:
            gpu_note = tk.Label(
                gpu_frame,
                text="Note: GPU training disables deterministic mode for compatibility with CUDA operations",
                bg=get_color('bg_medium'),
                fg=get_color('text_secondary'),
                font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_SMALL - 1),
                wraplength=500,
                justify=tk.LEFT
            )
            gpu_note.pack(anchor=tk.W, padx=(20, 0), pady=(0, 5))

        # GPU count selector (if multiple GPUs)
        if self.gpu_count > 1:
            gpu_count_frame = tk.Frame(gpu_frame, bg=get_color('bg_medium'))
            gpu_count_frame.pack(fill=tk.X, pady=(5, 0))

            tk.Label(
                gpu_count_frame,
                text="Number of GPUs to use:",
                bg=get_color('bg_medium'),
                fg=get_color('text_primary'),
                font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL)
            ).pack(side=tk.LEFT, padx=(20, 10))

            self.gpu_count_var = tk.IntVar(value=1)
            gpu_count_spinbox = tk.Spinbox(
                gpu_count_frame,
                from_=1,
                to=self.gpu_count,
                textvariable=self.gpu_count_var,
                width=5,
                bg=get_color('bg_light'),
                fg=get_color('text_primary'),
                font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL)
            )
            gpu_count_spinbox.pack(side=tk.LEFT)
        else:
            self.gpu_count_var = tk.IntVar(value=1)

        # Hyperparameters in grid
        params_frame = tk.Frame(section, bg=get_color('bg_medium'))
        params_frame.pack(fill=tk.X, pady=(10, 5))

        # Create entry fields for hyperparameters
        self.config_vars = {}

        params = [
            ('Hidden Size', 'hidden_size', 160),
            ('LSTM Layers', 'lstm_layers', 2),
            ('Attention Heads', 'attention_head_size', 4),
            ('Dropout', 'dropout', 0.15),
            ('Batch Size', 'batch_size', 64),
            ('Max Epochs', 'max_epochs', 50),
            ('Learning Rate', 'learning_rate', 0.0005),
        ]

        for i, (label, var_name, default) in enumerate(params):
            row = i // 2
            col = (i % 2) * 2

            tk.Label(
                params_frame,
                text=f"{label}:",
                bg=get_color('bg_medium'),
                fg=get_color('text_primary')
            ).grid(row=row, column=col, sticky=tk.W, padx=(0, 5), pady=2)

            var = tk.StringVar(value=str(default))
            self.config_vars[var_name] = var

            entry = tk.Entry(
                params_frame,
                textvariable=var,
                width=15,
                bg=get_color('bg_light'),
                fg=get_color('text_primary'),
                insertbackground=get_color('text_primary')
            )
            entry.grid(row=row, column=col+1, sticky=tk.W, padx=(0, 20), pady=2)

        # Buttons
        btn_frame = tk.Frame(section, bg=get_color('bg_medium'))
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        tk.Button(
            btn_frame,
            text="Use Defaults",
            command=self._load_default_config,
            bg=get_color('bg_light'),
            fg=get_color('text_primary'),
            padx=10,
            pady=5
        ).pack(side=tk.LEFT, padx=(0, 5))

        tk.Button(
            btn_frame,
            text="Load Config",
            command=self._load_config_file,
            bg=get_color('bg_light'),
            fg=get_color('text_primary'),
            padx=10,
            pady=5
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame,
            text="Save Config",
            command=self._save_config_file,
            bg=get_color('bg_light'),
            fg=get_color('text_primary'),
            padx=10,
            pady=5
        ).pack(side=tk.LEFT, padx=5)

        # Action buttons
        action_frame = tk.Frame(section, bg=get_color('bg_medium'))
        action_frame.pack(fill=tk.X, pady=(10, 0))

        self.start_training_btn = tk.Button(
            action_frame,
            text="🚀 Start Training",
            command=self._start_training,
            bg=get_color('success'),
            fg='white',
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold'),
            padx=24,
            pady=12,
            relief=tk.FLAT,
            borderwidth=0,
            cursor="hand2"
        )
        self.start_training_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.stop_training_btn = tk.Button(
            action_frame,
            text="🛑 Stop Training",
            command=self._stop_training,
            bg=get_color('danger'),
            fg='white',
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold'),
            padx=24,
            pady=12,
            relief=tk.FLAT,
            borderwidth=0,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.stop_training_btn.pack(side=tk.LEFT, padx=8)

    def _create_training_progress_section(self, parent):
        """Create training progress section"""
        print(f"[ML_PANEL]   - Creating training progress section...")

        section = tk.LabelFrame(
            parent,
            text=" 📈 Training Progress ",
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_LARGE, 'bold'),
            bg=get_color('bg_medium'),
            fg=get_color('text_primary'),
            padx=20,
            pady=15,
            borderwidth=0,
            relief='flat'
        )
        section.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 12))

        # Progress bar
        progress_frame = tk.Frame(section, bg=get_color('bg_medium'))
        progress_frame.pack(fill=tk.X, pady=(0, 10))

        self.progress_label = tk.Label(
            progress_frame,
            text="Ready to train",
            bg=get_color('bg_medium'),
            fg=get_color('text_primary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL)
        )
        self.progress_label.pack(anchor=tk.W)

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            length=400
        )
        self.progress_bar.pack(fill=tk.X, pady=5)

        # Loss curves
        chart_frame = tk.Frame(section, bg=get_color('bg_medium'))
        chart_frame.pack(fill=tk.X, pady=5)

        # Create matplotlib figure with smaller size
        self.loss_figure = Figure(figsize=(8, 3), facecolor=get_color('bg_medium'))
        self.loss_ax = self.loss_figure.add_subplot(111)
        self.loss_ax.set_facecolor(get_color('bg_light'))
        self.loss_ax.set_xlabel('Epoch', color=get_color('text_primary'), fontsize=9)
        self.loss_ax.set_ylabel('Loss', color=get_color('text_primary'), fontsize=9)
        self.loss_ax.set_title('Training & Validation Loss', color=get_color('text_primary'), fontsize=10)
        self.loss_ax.tick_params(colors=get_color('text_primary'), labelsize=8)
        self.loss_ax.grid(True, alpha=0.3)

        self.loss_canvas = FigureCanvasTkAgg(self.loss_figure, chart_frame)
        self.loss_canvas.draw()
        self.loss_canvas.get_tk_widget().pack(fill=tk.X)

        # Metrics display
        metrics_frame = tk.Frame(section, bg=get_color('bg_medium'))
        metrics_frame.pack(fill=tk.X, pady=5)

        self.metrics_label = tk.Label(
            metrics_frame,
            text="Metrics: Waiting for training...",
            bg=get_color('bg_medium'),
            fg=get_color('text_secondary'),
            font=(AppConfig.FONT_MONO, AppConfig.FONT_SIZE_SMALL)
        )
        self.metrics_label.pack(anchor=tk.W)

        # Log output
        log_frame = tk.Frame(section, bg=get_color('bg_medium'))
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        tk.Label(
            log_frame,
            text="Training Logs:",
            bg=get_color('bg_medium'),
            fg=get_color('text_primary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold')
        ).pack(anchor=tk.W)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=6,
            bg=get_color('bg_dark'),
            fg=get_color('text_primary'),
            font=(AppConfig.FONT_MONO, AppConfig.FONT_SIZE_SMALL),
            wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.X, pady=5)

    def _create_prediction_section(self, parent):
        """Create prediction section"""
        print(f"[ML_PANEL]   - Creating prediction section...")

        section = tk.LabelFrame(
            parent,
            text=" 🔮 Prediction & Inference ",
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_LARGE, 'bold'),
            bg=get_color('bg_medium'),
            fg=get_color('text_primary'),
            padx=20,
            pady=15,
            borderwidth=0,
            relief='flat'
        )
        section.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        # Model selection
        model_frame = tk.Frame(section, bg=get_color('bg_medium'))
        model_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            model_frame,
            text="Load Model:",
            bg=get_color('bg_medium'),
            fg=get_color('text_primary')
        ).pack(side=tk.LEFT, padx=(0, 5))

        self.model_path_var = tk.StringVar(value="No model loaded")
        tk.Entry(
            model_frame,
            textvariable=self.model_path_var,
            state='readonly',
            bg=get_color('bg_light'),
            fg=get_color('text_primary'),
            width=40
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        tk.Button(
            model_frame,
            text="Browse",
            command=self._browse_model,
            bg=get_color('bg_light'),
            fg=get_color('text_primary'),
            padx=10
        ).pack(side=tk.LEFT)

        # Symbol selection
        symbol_frame = tk.Frame(section, bg=get_color('bg_medium'))
        symbol_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            symbol_frame,
            text="Predict For:",
            bg=get_color('bg_medium'),
            fg=get_color('text_primary')
        ).pack(side=tk.LEFT, padx=(0, 5))

        self.predict_symbol_var = tk.StringVar()
        self.predict_symbol_combo = ttk.Combobox(
            symbol_frame,
            textvariable=self.predict_symbol_var,
            state='readonly',
            width=20
        )
        self.predict_symbol_combo.pack(side=tk.LEFT, padx=5)

        tk.Button(
            symbol_frame,
            text="🔮 Generate 10-Hour Prediction",
            command=self._generate_prediction,
            bg=get_color('primary'),
            fg='white',
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold'),
            padx=24,
            pady=10,
            relief=tk.FLAT,
            borderwidth=0,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=15)

        # Prediction results placeholder
        self.prediction_label = tk.Label(
            section,
            text="No predictions yet. Load a model and select a symbol to predict.",
            bg=get_color('bg_medium'),
            fg=get_color('text_secondary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL)
        )
        self.prediction_label.pack(pady=20)

    def _scan_datasets(self):
        """Scan dataset directory for available datasets"""
        print(f"\n[ML_PANEL] Scanning for available datasets...")

        try:
            dataset_dir = Path(AppConfig.DATASET_DIR)

            if not dataset_dir.exists():
                print(f"[ML_PANEL] ⚠ Dataset directory not found: {dataset_dir}")
                return

            # Find symbol directories
            for symbol_dir in dataset_dir.iterdir():
                if symbol_dir.is_dir() and symbol_dir.name not in ['.', '..']:
                    # Find datasets for this symbol
                    datasets = [d for d in symbol_dir.iterdir() if d.is_dir()]

                    if datasets:
                        # Get latest dataset
                        latest = sorted(datasets)[-1]

                        # Read metadata
                        metadata_path = latest / 'metadata.json'
                        if metadata_path.exists():
                            with open(metadata_path) as f:
                                metadata = json.load(f)

                            self.available_datasets[symbol_dir.name] = {
                                'path': latest,
                                'candles': metadata.get('candle_count', 0),
                                'metadata': metadata
                            }

                            print(f"[ML_PANEL]   - Found {symbol_dir.name}: {metadata.get('candle_count', 0)} candles")

            # Update UI
            self._update_dataset_checkboxes()
            self._update_symbol_combo()

            print(f"[ML_PANEL] ✓ Found {len(self.available_datasets)} datasets")

        except Exception as e:
            print(f"[ML_PANEL] ✗ Failed to scan datasets: {e}")

    def _update_dataset_checkboxes(self):
        """Update dataset selection checkboxes"""
        # Clear existing checkboxes
        for widget in self.dataset_checkboxes_frame.winfo_children():
            widget.destroy()

        if not self.available_datasets:
            tk.Label(
                self.dataset_checkboxes_frame,
                text="No datasets found. Use Data Pipeline to download cryptocurrency data.",
                bg=get_color('bg_medium'),
                fg=get_color('text_secondary'),
                font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_SMALL, 'italic')
            ).pack(anchor=tk.W, pady=5)
            return

        # Create checkboxes
        self.dataset_vars = {}

        for symbol, info in self.available_datasets.items():
            var = tk.BooleanVar(value=True)  # Default: selected
            self.dataset_vars[symbol] = var

            cb = tk.Checkbutton(
                self.dataset_checkboxes_frame,
                text=f"{symbol} ({info['candles']:,} candles)",
                variable=var,
                bg=get_color('bg_medium'),
                fg=get_color('text_primary'),
                selectcolor=get_color('bg_light'),
                activebackground=get_color('bg_medium'),
                activeforeground=get_color('text_primary')
            )
            cb.pack(anchor=tk.W, pady=2)

    def _update_symbol_combo(self):
        """Update prediction symbol dropdown"""
        symbols = list(self.available_datasets.keys())
        self.predict_symbol_combo['values'] = symbols

        if symbols:
            self.predict_symbol_combo.current(0)

    def _update_split_labels(self, *args):
        """Update split ratio labels"""
        train = self.train_split_var.get()
        val = self.val_split_var.get()
        test = 1.0 - train - val

        self.train_split_label.config(text=f"{int(train*100)}%")
        self.val_split_label.config(text=f"{int(val*100)}%")
        self.test_split_label.config(text=f"{int(test*100)}%")

    def _load_default_config(self):
        """Load default configuration"""
        print(f"[ML_PANEL] Loading default configuration...")

        defaults = {
            'hidden_size': '160',
            'lstm_layers': '2',
            'attention_head_size': '4',
            'dropout': '0.15',
            'batch_size': '64',
            'max_epochs': '50',
            'learning_rate': '0.0005',
        }

        for key, value in defaults.items():
            if key in self.config_vars:
                self.config_vars[key].set(value)

        self.log_message("[CONFIG] Loaded default configuration")

    def _load_config_file(self):
        """Load configuration from JSON file"""
        filepath = filedialog.askopenfilename(
            title="Load Model Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filepath:
            print(f"[ML_PANEL] Loading config from {filepath}")
            try:
                with open(filepath) as f:
                    config = json.load(f)

                for key, value in config.items():
                    if key in self.config_vars:
                        self.config_vars[key].set(str(value))

                self.log_message(f"[CONFIG] Loaded configuration from {Path(filepath).name}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load config: {e}")

    def _save_config_file(self):
        """Save configuration to JSON file"""
        filepath = filedialog.asksaveasfilename(
            title="Save Model Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filepath:
            print(f"[ML_PANEL] Saving config to {filepath}")
            try:
                config = {key: var.get() for key, var in self.config_vars.items()}

                with open(filepath, 'w') as f:
                    json.dump(config, f, indent=2)

                self.log_message(f"[CONFIG] Saved configuration to {Path(filepath).name}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to save config: {e}")

    def _start_training(self):
        """Start model training"""
        print(f"\n[ML_PANEL] Starting training...")

        # Get selected symbols
        selected = [symbol for symbol, var in self.dataset_vars.items() if var.get()]

        if not selected:
            messagebox.showwarning("No Datasets", "Please select at least one dataset for training.")
            return

        print(f"[ML_PANEL]   - Selected symbols: {selected}")

        # Get config
        config = self._get_config()

        # Update UI
        self.training_active = True
        self.start_training_btn.config(state=tk.DISABLED)
        self.stop_training_btn.config(state=tk.NORMAL)
        self.progress_bar['value'] = 0
        self.progress_label.config(text="Initializing training...")

        # Clear previous data
        self.train_loss_history = []
        self.val_loss_history = []
        self.epochs = []

        self.log_message(f"[TRAINING] Starting training with {len(selected)} symbols")
        self.log_message(f"[TRAINING] Config: {config}")

        # Call callback
        if self.on_train_start:
            self.on_train_start(config, selected)

    def _stop_training(self):
        """Stop training"""
        print(f"[ML_PANEL] Stopping training...")

        self.training_active = False
        self.start_training_btn.config(state=tk.NORMAL)
        self.stop_training_btn.config(state=tk.DISABLED)

        self.log_message("[TRAINING] Training stopped by user")

    def _get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        config = {}

        try:
            config['hidden_size'] = int(self.config_vars['hidden_size'].get())
            config['lstm_layers'] = int(self.config_vars['lstm_layers'].get())
            config['attention_head_size'] = int(self.config_vars['attention_head_size'].get())
            config['dropout'] = float(self.config_vars['dropout'].get())
            config['batch_size'] = int(self.config_vars['batch_size'].get())
            config['max_epochs'] = int(self.config_vars['max_epochs'].get())
            config['learning_rate'] = float(self.config_vars['learning_rate'].get())
            config['train_split'] = self.train_split_var.get()
            config['val_split'] = self.val_split_var.get()

            # GPU configuration
            config['use_gpu'] = self.use_gpu_var.get() if self.gpu_available else False
            config['gpu_count'] = self.gpu_count_var.get() if config['use_gpu'] else 0
        except ValueError as e:
            print(f"[ML_PANEL] ✗ Invalid config value: {e}")
            messagebox.showerror("Invalid Configuration", f"Please check configuration values: {e}")
            return {}

        return config

    def _browse_model(self):
        """Browse for model checkpoint"""
        filepath = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[("Checkpoint files", "*.ckpt"), ("All files", "*.*")]
        )

        if filepath:
            self.model_path_var.set(filepath)
            self.log_message(f"[MODEL] Loaded model: {Path(filepath).name}")

    def _generate_prediction(self):
        """Generate prediction"""
        model_path = self.model_path_var.get()

        if model_path == "No model loaded":
            messagebox.showwarning("No Model", "Please load a trained model first.")
            return

        symbol = self.predict_symbol_var.get()

        if not symbol:
            messagebox.showwarning("No Symbol", "Please select a cryptocurrency symbol.")
            return

        print(f"[ML_PANEL] Generating prediction for {symbol}...")

        self.prediction_label.config(text=f"Generating 10-hour prediction for {symbol}...")

        # Call callback
        if self.on_predict_click:
            self.on_predict_click(model_path, symbol)

    def update_progress(self, epoch: int, max_epochs: int, train_loss: float, val_loss: float):
        """Update training progress"""
        progress = (epoch / max_epochs) * 100
        self.progress_bar['value'] = progress
        self.progress_label.config(text=f"Epoch {epoch}/{max_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Update history
        self.epochs.append(epoch)
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)

        # Update chart
        self._update_loss_chart()

    def _update_loss_chart(self):
        """Update loss curve chart"""
        self.loss_ax.clear()
        self.loss_ax.set_facecolor(get_color('bg_light'))
        self.loss_ax.set_xlabel('Epoch', color=get_color('text_primary'))
        self.loss_ax.set_ylabel('Loss', color=get_color('text_primary'))
        self.loss_ax.set_title('Training & Validation Loss', color=get_color('text_primary'))
        self.loss_ax.tick_params(colors=get_color('text_primary'))
        self.loss_ax.grid(True, alpha=0.3)

        if self.epochs:
            self.loss_ax.plot(self.epochs, self.train_loss_history, label='Train Loss', color='#4fc3f7', linewidth=2)
            self.loss_ax.plot(self.epochs, self.val_loss_history, label='Val Loss', color='#ff9800', linewidth=2)
            self.loss_ax.legend()

        self.loss_canvas.draw()

    def log_message(self, message: str):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)

    def training_complete(self, success: bool, message: str):
        """Called when training completes"""
        print(f"[ML_PANEL] Training complete: {success}")

        self.training_active = False
        self.start_training_btn.config(state=tk.NORMAL)
        self.stop_training_btn.config(state=tk.DISABLED)

        if success:
            self.progress_label.config(text=f"✓ Training Complete: {message}")
            self.log_message(f"[TRAINING] ✓ {message}")
            messagebox.showinfo("Training Complete", message)
        else:
            self.progress_label.config(text=f"✗ Training Failed: {message}")
            self.log_message(f"[TRAINING] ✗ {message}")
            messagebox.showerror("Training Failed", message)

    def _refresh_gpu_status(self):
        """Refresh GPU memory status"""
        if not self.gpu_available or self.gpu_mem_label is None:
            return

        try:
            import torch
            gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

            self.gpu_mem_label.config(
                text=f"  •  Memory: {gpu_mem_allocated:.2f}GB allocated, {gpu_mem_reserved:.2f}GB reserved / {gpu_mem_total:.2f}GB total"
            )

            # Log the refresh
            self.log_message(f"[GPU] Memory: {gpu_mem_allocated:.2f}GB / {gpu_mem_total:.2f}GB")
        except Exception as e:
            print(f"[ML_PANEL] Failed to refresh GPU status: {e}")

    def display_prediction(self, predictions: Dict[str, Any], symbol: str):
        """Display prediction results"""
        print(f"[ML_PANEL] Displaying prediction for {symbol}...")

        # Update prediction label with summary
        median = predictions.get('median', [])

        if len(median) > 0:
            price_change = ((median[-1] - median[0]) / median[0]) * 100
            trend = "↗ Bullish" if price_change > 0.5 else ("↘ Bearish" if price_change < -0.5 else "→ Neutral")

            summary = (
                f"Prediction for {symbol}:\n"
                f"Current → 10h: ${median[0]:.2f} → ${median[-1]:.2f} "
                f"({price_change:+.2f}%) {trend}"
            )

            self.prediction_label.config(text=summary, fg=get_color('accent'))
            self.log_message(f"[PREDICTION] {summary}")
        else:
            self.prediction_label.config(text="Prediction failed - no data returned")


if __name__ == "__main__":
    # Test ML Panel
    print("Testing ML Panel")
    print("=" * 60)

    root = tk.Tk()
    root.title("ML Panel Test")
    root.geometry("1200x900")

    def on_train(config, symbols):
        print(f"Training started with config: {config}, symbols: {symbols}")

    def on_predict(model_path, symbol):
        print(f"Prediction requested for {symbol} with model {model_path}")

    panel = MLPanel(root, on_train_start=on_train, on_predict_click=on_predict)

    root.mainloop()
