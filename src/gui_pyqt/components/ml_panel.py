"""
ML Panel Component - AI Model Training and Prediction Interface

Provides PyQt6 interface for:
- Dataset configuration and selection
- Model hyperparameter configuration
- Training progress with loss curves
- Prediction and inference
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QProgressBar, QGroupBox, QLineEdit, QDoubleSpinBox,
    QSpinBox, QTextEdit, QSplitter, QFrame, QGridLayout, QCheckBox,
    QScrollArea, QSlider, QRadioButton, QButtonGroup, QFileDialog,
    QMessageBox, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from pathlib import Path
from typing import Dict, Optional, List, Any
import json

from src.gui_pyqt.styles import COLORS, get_chart_colors
from src.core.config import AppConfig


class MLPanel(QWidget):
    """
    Machine Learning Panel for model training and prediction.

    Sections:
    1. Dataset Configuration - Select datasets and train/val/test split
    2. Model Configuration - Hyperparameters and settings
    3. Training Progress - Live metrics, loss curves, logs
    4. Prediction & Inference - Generate and visualize predictions

    Signals:
        train_requested(dict, list): Config dict and selected symbols when training starts
        predict_requested(str, str): Model path and symbol for prediction
    """

    train_requested = pyqtSignal(dict, list)
    predict_requested = pyqtSignal(str, str)
    model_test_requested = pyqtSignal(str, str)
    stop_training_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._training_active = False
        self._available_datasets = {}
        self._train_loss_history = []
        self._val_loss_history = []
        self._epochs = []
        self._setup_ui()
        self._scan_datasets()

    def _setup_ui(self):
        """Initialize the UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Scrollable area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Header
        header = QLabel("AI Model Training & Prediction")
        header.setProperty("class", "title")
        layout.addWidget(header)

        subtitle = QLabel("RankNet AI for cryptocurrency direction prediction")
        subtitle.setProperty("class", "secondary")
        layout.addWidget(subtitle)

        # Section 1: Dataset Configuration
        dataset_section = self._create_dataset_section()
        layout.addWidget(dataset_section)

        # Section 2: Model Configuration
        config_section = self._create_config_section()
        layout.addWidget(config_section)

        # Section 3: Training Progress
        progress_section = self._create_progress_section()
        layout.addWidget(progress_section)

        # Section 4: Prediction
        prediction_section = self._create_prediction_section()
        layout.addWidget(prediction_section)

        layout.addStretch()

        scroll.setWidget(content)
        main_layout.addWidget(scroll)

    def _create_dataset_section(self) -> QGroupBox:
        """Create dataset configuration section"""
        group = QGroupBox("Dataset Configuration")
        layout = QVBoxLayout(group)

        # Dataset selection row with dropdown + refresh
        select_row = QHBoxLayout()
        select_row.addWidget(QLabel("Select Dataset for Training:"))

        self.dataset_combo = QComboBox()
        self.dataset_combo.setMinimumWidth(300)
        self.dataset_combo.setPlaceholderText("No datasets found")
        self.dataset_combo.setEditable(False)
        select_row.addWidget(self.dataset_combo, stretch=1)

        self.dataset_refresh_btn = QPushButton("Refresh")
        self.dataset_refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_primary']};
                font-weight: bold;
                padding: 6px 14px;
                border-radius: 4px;
                border: 1px solid {COLORS['border']};
            }}
            QPushButton:hover {{
                background-color: {COLORS['border']};
            }}
        """)
        self.dataset_refresh_btn.clicked.connect(self._refresh_datasets)
        select_row.addWidget(self.dataset_refresh_btn)

        layout.addLayout(select_row)

        # Dataset info label
        self.dataset_count_label = QLabel("0 datasets available")
        self.dataset_count_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        layout.addWidget(self.dataset_count_label)

        # Split ratios
        split_label = QLabel("Data Split Ratios:")
        split_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(split_label)

        # Train split
        train_row = QHBoxLayout()
        train_row.addWidget(QLabel("Train:"))
        self.train_split_slider = QSlider(Qt.Orientation.Horizontal)
        self.train_split_slider.setRange(50, 80)
        self.train_split_slider.setValue(70)
        self.train_split_slider.valueChanged.connect(self._update_split_labels)
        train_row.addWidget(self.train_split_slider, stretch=1)
        self.train_split_label = QLabel("70%")
        self.train_split_label.setStyleSheet(f"color: {COLORS['accent']};")
        self.train_split_label.setMinimumWidth(50)
        train_row.addWidget(self.train_split_label)
        layout.addLayout(train_row)

        # Validation split
        val_row = QHBoxLayout()
        val_row.addWidget(QLabel("Validation:"))
        self.val_split_slider = QSlider(Qt.Orientation.Horizontal)
        self.val_split_slider.setRange(10, 30)
        self.val_split_slider.setValue(15)
        self.val_split_slider.valueChanged.connect(self._update_split_labels)
        val_row.addWidget(self.val_split_slider, stretch=1)
        self.val_split_label = QLabel("15%")
        self.val_split_label.setStyleSheet(f"color: {COLORS['accent']};")
        self.val_split_label.setMinimumWidth(50)
        val_row.addWidget(self.val_split_label)
        layout.addLayout(val_row)

        # Test split (auto-calculated)
        test_row = QHBoxLayout()
        test_row.addWidget(QLabel("Test:"))
        test_row.addStretch()
        self.test_split_label = QLabel("15%")
        self.test_split_label.setStyleSheet(f"color: {COLORS['accent']};")
        self.test_split_label.setMinimumWidth(50)
        test_row.addWidget(self.test_split_label)
        layout.addLayout(test_row)

        return group

    def _create_config_section(self) -> QGroupBox:
        """Create model configuration section"""
        group = QGroupBox("Model Configuration")
        layout = QVBoxLayout(group)

        # GPU Status
        gpu_status = self._detect_gpu()
        self.gpu_status_label = QLabel(f"GPU Status: {gpu_status}")
        self.gpu_status_label.setProperty("class", "secondary")
        layout.addWidget(self.gpu_status_label)

        # Use GPU checkbox
        self.use_gpu_cb = QCheckBox("Use GPU for Training")
        self.use_gpu_cb.setChecked(self._gpu_available)
        self.use_gpu_cb.setEnabled(self._gpu_available)
        layout.addWidget(self.use_gpu_cb)

        # Hyperparameters grid
        params_layout = QGridLayout()
        self.config_inputs = {}

        params = [
            ('hidden_size', 'Hidden Size', 128, 8, 512),
            ('num_blocks', 'Num Blocks', 3, 1, 8),
            ('dropout', 'Dropout', 0.1, 0.0, 0.9),
            ('max_epochs', 'Max Epochs', 20, 1, 500),
            ('learning_rate', 'Learning Rate', 0.001, 0.0001, 0.01),
        ]

        for i, (key, label, default, min_val, max_val) in enumerate(params):
            row = i // 2
            col = (i % 2) * 2

            params_layout.addWidget(QLabel(f"{label}:"), row, col)

            if key in ['dropout', 'learning_rate']:
                spin = QDoubleSpinBox()
                spin.setRange(min_val, max_val)
                spin.setValue(default)
                spin.setSingleStep(0.01 if key == 'dropout' else 0.0001)
                spin.setDecimals(4 if key == 'learning_rate' else 2)
            else:
                spin = QSpinBox()
                spin.setRange(int(min_val), int(max_val))
                spin.setValue(int(default))

            self.config_inputs[key] = spin
            params_layout.addWidget(spin, row, col + 1)

        layout.addLayout(params_layout)

        # Config buttons
        config_btn_layout = QHBoxLayout()

        defaults_btn = QPushButton("Use Defaults")
        defaults_btn.clicked.connect(self._load_defaults)
        config_btn_layout.addWidget(defaults_btn)

        load_btn = QPushButton("Load Config")
        load_btn.clicked.connect(self._load_config)
        config_btn_layout.addWidget(load_btn)

        save_btn = QPushButton("Save Config")
        save_btn.clicked.connect(self._save_config)
        config_btn_layout.addWidget(save_btn)

        config_btn_layout.addStretch()
        layout.addLayout(config_btn_layout)


        # Training Mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Training Mode:"))

        self.mode_group = QButtonGroup(self)
        self.scratch_radio = QRadioButton("From Scratch")
        self.scratch_radio.setChecked(True)
        self.finetune_radio = QRadioButton("Fine-tune")
        self.mode_group.addButton(self.scratch_radio)
        self.mode_group.addButton(self.finetune_radio)
        self.scratch_radio.toggled.connect(self._update_training_mode_ui)

        mode_layout.addWidget(self.scratch_radio)
        mode_layout.addWidget(self.finetune_radio)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # Pre-trained model path (for fine-tuning)
        self.pretrained_row = QHBoxLayout()
        self.pretrained_row.addWidget(QLabel("Pre-trained Model:"))
        self.pretrained_input = QLineEdit()
        self.pretrained_input.setPlaceholderText("Select model checkpoint...")
        self.pretrained_row.addWidget(self.pretrained_input, stretch=1)
        self.pretrained_browse_btn = QPushButton("Browse")
        self.pretrained_browse_btn.clicked.connect(self._browse_pretrained)
        self.pretrained_row.addWidget(self.pretrained_browse_btn)

        self.pretrained_widget = QWidget()
        self.pretrained_widget.setLayout(self.pretrained_row)
        self.pretrained_widget.setVisible(False)
        layout.addWidget(self.pretrained_widget)

        # Action buttons
        action_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Training")
        self.start_btn.setStyleSheet(f"background-color: {COLORS['success']}; color: white; font-weight: bold; padding: 12px 24px;")
        self.start_btn.clicked.connect(self._start_training)
        action_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.setStyleSheet(f"background-color: {COLORS['danger']}; color: white; font-weight: bold; padding: 12px 24px;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training)
        action_layout.addWidget(self.stop_btn)

        action_layout.addStretch()
        layout.addLayout(action_layout)

        return group

    def _create_progress_section(self) -> QGroupBox:
        """Create training progress section"""
        group = QGroupBox("Training Progress")
        layout = QVBoxLayout(group)

        # Status and progress
        self.progress_status = QLabel("Ready to train")
        layout.addWidget(self.progress_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # ICE (Information Coefficient) Display - Prominent metrics box
        ice_frame = QFrame()
        ice_frame.setFrameShape(QFrame.Shape.StyledPanel)
        ice_frame.setStyleSheet(f"background-color: {COLORS['bg_medium']}; border-radius: 8px; padding: 8px;")
        ice_layout = QGridLayout(ice_frame)

        # IC Label
        ice_layout.addWidget(QLabel("Information Coefficient (IC):"), 0, 0)
        self.ic_value_label = QLabel("--")
        self.ic_value_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {COLORS['accent']};")
        ice_layout.addWidget(self.ic_value_label, 0, 1)

        # IC Rating
        self.ic_rating_label = QLabel("Train a model to see IC")
        self.ic_rating_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-style: italic;")
        ice_layout.addWidget(self.ic_rating_label, 0, 2)

        # Sharpe Ratio
        ice_layout.addWidget(QLabel("Sharpe Ratio:"), 1, 0)
        self.sharpe_value_label = QLabel("--")
        self.sharpe_value_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {COLORS['accent']};")
        ice_layout.addWidget(self.sharpe_value_label, 1, 1)

        # Accuracy
        ice_layout.addWidget(QLabel("Test Accuracy:"), 1, 2)
        self.test_acc_label = QLabel("--")
        self.test_acc_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {COLORS['accent']};")
        ice_layout.addWidget(self.test_acc_label, 1, 3)

        layout.addWidget(ice_frame)

        # Loss chart
        colors = get_chart_colors()
        self.loss_figure = Figure(figsize=(8, 3), facecolor=colors['background'])
        self.loss_ax = self.loss_figure.add_subplot(111)
        self.loss_ax.set_facecolor(colors['background'])
        self.loss_ax.set_xlabel('Epoch', color=colors['text'])
        self.loss_ax.set_ylabel('Loss', color=colors['text'])
        self.loss_ax.set_title('Training & Validation Loss', color=colors['text'])
        self.loss_ax.tick_params(colors=colors['text'])
        self.loss_ax.grid(True, alpha=0.3, color=colors['grid'])

        self.loss_canvas = FigureCanvas(self.loss_figure)
        self.loss_canvas.setMinimumHeight(200)
        layout.addWidget(self.loss_canvas)

        # Metrics display
        self.metrics_label = QLabel("Metrics: Waiting for training...")
        self.metrics_label.setProperty("class", "secondary")
        layout.addWidget(self.metrics_label)

        # Training log
        log_label = QLabel("Training Logs:")
        log_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(log_label)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(150)
        layout.addWidget(self.log_output)

        return group

    def _create_prediction_section(self) -> QGroupBox:
        """Create prediction section with AI direction recommendation"""
        group = QGroupBox("AI Prediction - Investment Direction")
        layout = QVBoxLayout(group)

        # Symbol selection row
        select_row = QHBoxLayout()
        select_row.addWidget(QLabel("Select Crypto:"))
        self.predict_symbol_combo = QComboBox()
        self.predict_symbol_combo.setMinimumWidth(150)
        self.predict_symbol_combo.currentIndexChanged.connect(self._check_model_availability)
        select_row.addWidget(self.predict_symbol_combo)

        select_row.addWidget(QLabel("Timeframe:"))
        self.predict_timeframe_combo = QComboBox()
        self.predict_timeframe_combo.setMinimumWidth(80)
        timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']
        for tf in timeframes:
            self.predict_timeframe_combo.addItem(tf, tf)
        self.predict_timeframe_combo.setCurrentText('1h')
        self.predict_timeframe_combo.currentIndexChanged.connect(self._check_model_availability)
        select_row.addWidget(self.predict_timeframe_combo)

        self.predict_btn = QPushButton("Get AI Prediction")
        self.predict_btn.setStyleSheet(f"background-color: {COLORS['primary']}; color: white; font-weight: bold; padding: 12px 24px;")
        self.predict_btn.clicked.connect(self._generate_ai_prediction)
        select_row.addWidget(self.predict_btn)

        self.refresh_data_btn = QPushButton("Refresh Data")
        self.refresh_data_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_primary']};
                font-weight: bold;
                padding: 12px 16px;
                border-radius: 4px;
                border: 1px solid {COLORS['border']};
            }}
            QPushButton:hover {{
                background-color: {COLORS['border']};
            }}
        """)
        self.refresh_data_btn.clicked.connect(self._on_refresh_data)
        select_row.addWidget(self.refresh_data_btn)

        select_row.addStretch()
        layout.addLayout(select_row)

        # Model availability status
        self.model_status_label = QLabel("Select a symbol and timeframe")
        self.model_status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px; font-style: italic;")
        layout.addWidget(self.model_status_label)

        # Main prediction display - big direction indicator
        self.direction_frame = QFrame()
        self.direction_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.direction_frame.setStyleSheet(f"background-color: {COLORS['bg_medium']}; border-radius: 8px; padding: 16px;")
        direction_layout = QVBoxLayout(self.direction_frame)

        # Direction arrow and text
        self.direction_label = QLabel("--")
        self.direction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.direction_label.setStyleSheet("font-size: 48px; font-weight: bold;")
        direction_layout.addWidget(self.direction_label)

        self.direction_text = QLabel("Select a crypto and click 'Get AI Prediction'")
        self.direction_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.direction_text.setStyleSheet("font-size: 18px;")
        direction_layout.addWidget(self.direction_text)

        # Confidence bar
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Confidence:"))
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #555; border-radius: 4px; text-align: center; }
            QProgressBar::chunk { background-color: #4CAF50; }
        """)
        conf_row.addWidget(self.confidence_bar, stretch=1)
        self.confidence_label = QLabel("--")
        self.confidence_label.setMinimumWidth(50)
        conf_row.addWidget(self.confidence_label)
        direction_layout.addLayout(conf_row)

        layout.addWidget(self.direction_frame)

        # Prediction details
        details_frame = QFrame()
        details_frame.setFrameShape(QFrame.Shape.StyledPanel)
        details_layout = QGridLayout(details_frame)

        details_layout.addWidget(QLabel("Prediction Horizon:"), 0, 0)
        self.horizon_label = QLabel("4 hours ahead")
        self.horizon_label.setStyleSheet(f"color: {COLORS['accent']};")
        details_layout.addWidget(self.horizon_label, 0, 1)

        details_layout.addWidget(QLabel("Model Accuracy:"), 0, 2)
        self.accuracy_label = QLabel("--")
        self.accuracy_label.setStyleSheet(f"color: {COLORS['accent']};")
        details_layout.addWidget(self.accuracy_label, 0, 3)

        details_layout.addWidget(QLabel("Model IC:"), 1, 0)
        self.ic_label = QLabel("--")
        self.ic_label.setStyleSheet(f"color: {COLORS['accent']};")
        details_layout.addWidget(self.ic_label, 1, 1)

        details_layout.addWidget(QLabel("Last Updated:"), 1, 2)
        self.last_update_label = QLabel("--")
        self.last_update_label.setStyleSheet(f"color: {COLORS['accent']};")
        details_layout.addWidget(self.last_update_label, 1, 3)

        layout.addWidget(details_frame)

        # Signal Readiness scrollable panel
        readiness_group = QGroupBox("Signal Readiness")
        readiness_layout = QVBoxLayout(readiness_group)

        readiness_scroll = QScrollArea()
        readiness_scroll.setWidgetResizable(True)
        readiness_scroll.setMinimumHeight(120)
        readiness_scroll.setMaximumHeight(250)
        readiness_scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: {COLORS['bg_dark']};
            }}
        """)

        self._readiness_container = QWidget()
        self._readiness_grid = QGridLayout(self._readiness_container)
        self._readiness_grid.setContentsMargins(8, 8, 8, 8)
        self._readiness_grid.setSpacing(6)

        # Header row
        hdr_symbol = QLabel("Symbol")
        hdr_symbol.setStyleSheet("font-weight: bold; font-size: 11px;")
        hdr_signal = QLabel("Signal")
        hdr_signal.setStyleSheet("font-weight: bold; font-size: 11px;")
        hdr_conf = QLabel("Confidence")
        hdr_conf.setStyleSheet("font-weight: bold; font-size: 11px;")
        hdr_status = QLabel("Status")
        hdr_status.setStyleSheet("font-weight: bold; font-size: 11px;")
        self._readiness_grid.addWidget(hdr_symbol, 0, 0)
        self._readiness_grid.addWidget(hdr_signal, 0, 1)
        self._readiness_grid.addWidget(hdr_conf, 0, 2)
        self._readiness_grid.addWidget(hdr_status, 0, 3)

        # Populated dynamically by _scan_datasets / _on_refresh_data
        self._signal_rows = {}  # symbol -> (signal_label, conf_label, status_label)

        readiness_scroll.setWidget(self._readiness_container)
        readiness_layout.addWidget(readiness_scroll)
        layout.addWidget(readiness_group)

        # Disclaimer
        disclaimer = QLabel("Note: AI predictions are based on historical patterns. Past performance does not guarantee future results. Always do your own research.")
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px; font-style: italic;")
        layout.addWidget(disclaimer)

        # Hidden fields for model path (used by browse)
        self.model_path_input = QLineEdit()
        self.model_path_input.setVisible(False)
        layout.addWidget(self.model_path_input)

        return group


    def _detect_gpu(self) -> str:
        """Detect GPU availability"""
        self._gpu_available = False
        self._gpu_count = 0

        try:
            import torch
            if torch.cuda.is_available():
                self._gpu_available = True
                self._gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if self._gpu_count > 0 else "Unknown"
                return f"{self._gpu_count} GPU(s) available: {gpu_name}"
            else:
                return "CUDA not available - CPU training only"
        except ImportError:
            return "PyTorch not installed"
        except Exception as e:
            return f"GPU detection failed: {str(e)}"

    def _scan_datasets(self):
        """Scan for available datasets"""
        try:
            dataset_dir = Path(AppConfig.DATASET_DIR)

            if not dataset_dir.exists():
                self._update_dataset_dropdown()
                return

            for symbol_dir in dataset_dir.iterdir():
                if symbol_dir.is_dir() and symbol_dir.name not in ['.', '..']:
                    # Find subdirs that have metadata.json or CSV files
                    valid_dirs = []
                    for d in symbol_dir.iterdir():
                        if d.is_dir():
                            has_meta = (d / 'metadata.json').exists()
                            has_csv = any(d.glob("*.csv"))
                            if has_meta or has_csv:
                                valid_dirs.append(d)

                    if not valid_dirs:
                        continue

                    # Pick latest by name (timestamp-based dir names sort correctly)
                    latest = sorted(valid_dirs)[-1]
                    metadata_path = latest / 'metadata.json'

                    candle_count = 0
                    metadata = {}
                    if metadata_path.exists():
                        with open(metadata_path) as f:
                            metadata = json.load(f)
                        candle_count = metadata.get('candle_count', 0)
                    else:
                        # Count rows from largest CSV as fallback
                        csv_files = list(latest.glob("*.csv"))
                        if csv_files:
                            largest = max(csv_files, key=lambda f: f.stat().st_size)
                            try:
                                import pandas as pd
                                candle_count = len(pd.read_csv(largest, nrows=0).columns) and sum(1 for _ in open(largest)) - 1
                            except Exception:
                                candle_count = 0

                    self._available_datasets[symbol_dir.name] = {
                        'path': latest,
                        'candles': candle_count,
                        'metadata': metadata
                    }

            self._update_dataset_dropdown()
            self._update_symbol_combo()
            self._populate_signal_readiness()

        except Exception as e:
            print(f"Failed to scan datasets: {e}")

    def _update_dataset_dropdown(self):
        """Update dataset dropdown with available datasets"""
        self.dataset_combo.clear()

        if not self._available_datasets:
            self.dataset_count_label.setText("0 datasets found")
            return

        for symbol in sorted(self._available_datasets.keys()):
            info = self._available_datasets[symbol]
            self.dataset_combo.addItem(f"{symbol} ({info['candles']:,} candles)", symbol)

        self.dataset_count_label.setText(
            f"{self.dataset_combo.count()} dataset(s) available"
        )

    def _get_selected_datasets(self) -> list:
        """Get the single selected dataset symbol from the dropdown."""
        symbol = self.dataset_combo.currentData()
        if symbol:
            return [symbol]
        return []

    def _refresh_datasets(self):
        """Refresh button handler - rescan datasets."""
        self._available_datasets.clear()
        self._scan_datasets()
        self.log_message("[REFRESH] Dataset scan complete")

    def _update_symbol_combo(self):
        """Update prediction symbol dropdown with trained models first"""
        self.predict_symbol_combo.clear()

        # Get symbols with trained models (prioritize these)
        from pathlib import Path
        trained_symbols = set()
        checkpoint_dir = Path("models/checkpoints")
        if checkpoint_dir.exists():
            for d in checkpoint_dir.iterdir():
                if d.is_dir() and any(d.glob("*.pt")):
                    trained_symbols.add(d.name)

        # Get all available symbols from Binance
        try:
            from src.gui_pyqt.utils.symbols import get_all_usdt_symbols
            all_symbols = set(get_all_usdt_symbols())
        except Exception:
            # Fallback to datasets only
            all_symbols = set(self._available_datasets.keys())

        # Also include symbols from local datasets
        all_symbols.update(self._available_datasets.keys())

        # Combine: trained models first, then others alphabetically
        trained_list = sorted(trained_symbols)
        other_list = sorted(all_symbols - trained_symbols)

        # Add with markers
        for sym in trained_list:
            self.predict_symbol_combo.addItem(f"{sym} [AI Ready]", sym)

        for sym in other_list:
            self.predict_symbol_combo.addItem(sym, sym)

    def _update_split_labels(self):
        """Update split ratio labels"""
        train = self.train_split_slider.value()
        val = self.val_split_slider.value()
        test = 100 - train - val

        self.train_split_label.setText(f"{train}%")
        self.val_split_label.setText(f"{val}%")
        self.test_split_label.setText(f"{test}%")

    def _update_training_mode_ui(self, checked):
        """Show/hide pretrained model selection"""
        self.pretrained_widget.setVisible(not self.scratch_radio.isChecked())

    def _load_defaults(self):
        """Load default configuration"""
        defaults = {
            'hidden_size': 128,
            'num_blocks': 3,
            'dropout': 0.1,
            'max_epochs': 20,
            'learning_rate': 0.001,
        }

        for key, value in defaults.items():
            if key in self.config_inputs:
                self.config_inputs[key].setValue(value)

        self.log_message("[CONFIG] Loaded default configuration")

    def _load_config(self):
        """Load configuration from file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Model Configuration",
            "", "JSON files (*.json);;All files (*.*)"
        )

        if filepath:
            try:
                with open(filepath) as f:
                    config = json.load(f)

                for key, value in config.items():
                    if key in self.config_inputs:
                        self.config_inputs[key].setValue(float(value) if '.' in str(value) else int(value))

                self.log_message(f"[CONFIG] Loaded configuration from {Path(filepath).name}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load config: {e}")

    def _save_config(self):
        """Save configuration to file"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Model Configuration",
            "", "JSON files (*.json);;All files (*.*)"
        )

        if filepath:
            try:
                config = {key: spin.value() for key, spin in self.config_inputs.items()}

                with open(filepath, 'w') as f:
                    json.dump(config, f, indent=2)

                self.log_message(f"[CONFIG] Saved configuration to {Path(filepath).name}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save config: {e}")

    def _browse_pretrained(self):
        """Browse for pretrained model"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Pre-trained Model",
            "models/checkpoints", "Model files (*.ckpt *.pt *.pth);;All files (*.*)"
        )
        if filepath:
            self.pretrained_input.setText(filepath)

    def _browse_model(self):
        """Browse for model to load"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Model",
            "models/checkpoints", "Model files (*.ckpt *.pt *.pth);;All files (*.*)"
        )
        if filepath:
            self.model_path_input.setText(filepath)

    def _get_config(self) -> Dict:
        """Get current configuration"""
        config = {key: spin.value() for key, spin in self.config_inputs.items()}
        config['use_gpu'] = self.use_gpu_cb.isChecked()
        config['architecture'] = 'lstm'
        config['training_mode'] = 'scratch' if self.scratch_radio.isChecked() else 'finetune'
        config['pretrained_path'] = self.pretrained_input.text() if not self.scratch_radio.isChecked() else None
        config['train_split'] = self.train_split_slider.value() / 100.0
        config['val_split'] = self.val_split_slider.value() / 100.0
        return config

    def _start_training(self):
        """Start model training on the selected dataset"""
        selected = self._get_selected_datasets()

        if not selected:
            QMessageBox.warning(self, "No Dataset", "Please select a dataset for training.")
            return

        config = self._get_config()

        self._training_active = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_status.setText("Initializing training...")

        # Clear history
        self._train_loss_history = []
        self._val_loss_history = []
        self._epochs = []

        self.log_message(f"[TRAINING] Starting training on {selected[0]}")
        self.train_requested.emit(config, selected)

    def _stop_training(self):
        """Stop model training"""
        self._training_active = False
        self.log_message("[TRAINING] Stop requested...")
        self.stop_training_requested.emit()

    def _generate_prediction(self):
        """Generate prediction"""
        model_path = self.model_path_input.text()
        symbol = self.predict_symbol_combo.currentData() or self.predict_symbol_combo.currentText().replace(" [AI Ready]", "")

        if not model_path or model_path == "No model loaded":
            QMessageBox.warning(self, "No Model", "Please load a model first.")
            return

        if not symbol:
            QMessageBox.warning(self, "No Symbol", "Please select a symbol.")
            return

        self.predict_requested.emit(model_path, symbol)

    def _generate_quick_signal(self):
        """Generate quick trading signal based on technical indicators"""
        symbol = self.predict_symbol_combo.currentData() or self.predict_symbol_combo.currentText().replace(" [AI Ready]", "")
        if not symbol:
            return
        # Redirect to AI prediction
        self._generate_ai_prediction()

    def _check_model_availability(self):
        """Check if a trained model exists for the selected symbol and timeframe."""
        symbol = self.predict_symbol_combo.currentData()
        if symbol is None:
            symbol = self.predict_symbol_combo.currentText().replace(" [AI Ready]", "")
        timeframe = self.predict_timeframe_combo.currentData() or '1h'

        if not symbol:
            self.model_status_label.setText("Select a symbol and timeframe")
            self.model_status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px; font-style: italic;")
            self.predict_btn.setEnabled(True)
            return

        from pathlib import Path
        model_path = Path(f"models/checkpoints/{symbol}/ranknet_{timeframe}.pt")
        legacy_path = Path(f"models/checkpoints/{symbol}/ranknet_model.pt")

        if model_path.exists():
            self.model_status_label.setText(f"Model available for {symbol} @ {timeframe}")
            self.model_status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 11px; font-weight: bold;")
            self.predict_btn.setEnabled(True)
        elif timeframe == '1h' and legacy_path.exists():
            self.model_status_label.setText(f"Model available for {symbol} @ 1h (legacy)")
            self.model_status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 11px; font-weight: bold;")
            self.predict_btn.setEnabled(True)
        else:
            # List available timeframes for this symbol
            checkpoint_dir = Path(f"models/checkpoints/{symbol}")
            available_tfs = []
            if checkpoint_dir.exists():
                for f in checkpoint_dir.glob("ranknet_*.pt"):
                    tf = f.stem.replace("ranknet_", "")
                    if tf != "model":
                        available_tfs.append(tf)
                # Check legacy model
                if (checkpoint_dir / "ranknet_model.pt").exists():
                    available_tfs.append("1h (legacy)")

            if available_tfs:
                self.model_status_label.setText(
                    f"No model for {symbol} @ {timeframe}. Available: {', '.join(available_tfs)}"
                )
            else:
                self.model_status_label.setText(f"No model for {symbol}. Train a model first.")
            self.model_status_label.setStyleSheet(f"color: {COLORS['danger']}; font-size: 11px;")
            self.predict_btn.setEnabled(False)

    def _resolve_model_path(self, symbol: str, timeframe: str):
        """Resolve model path for a symbol and timeframe. Returns Path or None."""
        from pathlib import Path
        model_path = Path(f"models/checkpoints/{symbol}/ranknet_{timeframe}.pt")
        if model_path.exists():
            return model_path
        # Legacy fallback for 1h models
        if timeframe == '1h':
            legacy_path = Path(f"models/checkpoints/{symbol}/ranknet_model.pt")
            if legacy_path.exists():
                return legacy_path
        return None

    def _generate_ai_prediction(self):
        """Generate AI prediction for investment direction using RankNet.

        Fetches live/recent candlestick data from the exchange API at the
        selected timeframe and runs inference through the pre-trained model.
        """
        # Get actual symbol (not display text which may include "[AI Ready]")
        symbol = self.predict_symbol_combo.currentData()
        if symbol is None:
            symbol = self.predict_symbol_combo.currentText().replace(" [AI Ready]", "")
        if not symbol:
            QMessageBox.warning(self, "No Symbol", "Please select a cryptocurrency.")
            return

        timeframe = self.predict_timeframe_combo.currentData() or '1h'

        try:
            from datetime import datetime
            import numpy as np
            import pandas as pd
            from src.ml.models.ranknet_model import RankNetPredictor

            # --- Step 1: Resolve and load the pre-trained model ---
            model_path = self._resolve_model_path(symbol, timeframe)
            if model_path is None:
                self.direction_label.setText("?")
                self.direction_text.setText(
                    f"No trained model for {symbol} @ {timeframe}. "
                    "Train a model on this timeframe first."
                )
                self.direction_label.setStyleSheet(
                    f"font-size: 48px; font-weight: bold; color: {COLORS['text_secondary']};"
                )
                return

            try:
                predictor = RankNetPredictor.load(str(model_path))
                self.log_message(f"[RANKNET] Loaded model for {symbol} @ {timeframe}: {model_path.name}")
            except Exception as e:
                self.direction_label.setText("!")
                self.direction_text.setText(f"Failed to load model: {e}")
                self.direction_label.setStyleSheet(
                    f"font-size: 48px; font-weight: bold; color: {COLORS['danger']};"
                )
                return

            # --- Step 2: Fetch live/recent candlestick data from API ---
            self.log_message(f"[RANKNET] Fetching live {timeframe} data for {symbol} from Binance...")
            self.direction_text.setText("Fetching live market data...")
            QApplication.processEvents()  # Allow UI to update

            try:
                from src.api.ccxt_client import get_ccxt_client
                client = get_ccxt_client('binance')
                ccxt_symbol = client.convert_symbol_format(symbol, to_ccxt=True)
                df = client.get_ohlcv(ccxt_symbol, timeframe, 500)
            except Exception as e:
                self.direction_label.setText("!")
                self.direction_text.setText(f"Failed to fetch live data: {e}")
                self.direction_label.setStyleSheet(
                    f"font-size: 48px; font-weight: bold; color: {COLORS['danger']};"
                )
                self.log_message(f"[RANKNET] API error: {e}")
                return

            if df is None or len(df) < 300:
                count = len(df) if df is not None else 0
                self.direction_label.setText("?")
                self.direction_text.setText(
                    f"Insufficient live data ({count} candles, need 300+). "
                    "Try a larger timeframe or check connectivity."
                )
                return

            self.log_message(f"[RANKNET] Fetched {len(df)} live {timeframe} candles for {symbol}")

            # --- Step 3: Run prediction on live data ---
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            missing = [c for c in ohlcv_cols if c not in df.columns]
            if missing:
                self.direction_label.setText("?")
                self.direction_text.setText(f"Live data missing columns: {missing}")
                return

            result = predictor.predict(df[ohlcv_cols])

            direction = result['direction']
            confidence_val = result['confidence']
            metrics = result['metrics']

            # --- Step 4: Display results ---
            if direction == "UP":
                arrow = "▲"
                color = COLORS['success']
                recommendation = f"RankNet AI suggests {symbol} may go UP ({timeframe})"
            elif direction == "DOWN":
                arrow = "▼"
                color = COLORS['danger']
                recommendation = f"RankNet AI suggests {symbol} may go DOWN ({timeframe})"
            else:
                arrow = "◆"
                color = COLORS['text_secondary']
                recommendation = f"RankNet AI is uncertain about {symbol} direction ({timeframe})"

            self.direction_label.setText(f"{arrow} {direction}")
            self.direction_label.setStyleSheet(f"font-size: 48px; font-weight: bold; color: {color};")
            self.direction_text.setText(recommendation)
            self.direction_text.setStyleSheet(f"font-size: 18px; color: {color};")

            # Confidence
            confidence_pct = confidence_val * 100
            self.confidence_bar.setValue(int(min(confidence_pct, 100)))
            self.confidence_label.setText(f"{confidence_pct:.0f}%")

            bar_colors = {"UP": "#4CAF50", "DOWN": "#F44336", "NEUTRAL": "#888"}
            bar_color = bar_colors.get(direction, "#888")
            self.confidence_bar.setStyleSheet(f"""
                QProgressBar {{ border: 1px solid #555; border-radius: 4px; text-align: center; }}
                QProgressBar::chunk {{ background-color: {bar_color}; }}
            """)

            # Update details
            self.horizon_label.setText(f"30 candles ahead ({timeframe})")
            self.accuracy_label.setText(f"{metrics.get('directional_accuracy', 0) * 100:.1f}%")
            self.ic_label.setText(f"{metrics.get('ic', 0):.4f}")
            self.last_update_label.setText(datetime.now().strftime("%H:%M:%S"))

            # Update signal readiness panel
            self._update_signal_readiness(symbol, direction, confidence_val)

            self.log_message(
                f"[RANKNET] {symbol} @ {timeframe}: {direction} (conf={confidence_pct:.0f}%, "
                f"IC={metrics.get('ic', 0):.4f}, "
                f"Sharpe={metrics.get('sharpe', 0):.2f}, "
                f"WinRate={metrics.get('win_rate', 0) * 100:.1f}%)"
            )

        except Exception as e:
            self.direction_label.setText("!")
            self.direction_text.setText(f"Error: {str(e)}")
            self.direction_label.setStyleSheet(f"font-size: 48px; font-weight: bold; color: {COLORS['danger']};")
            import traceback
            traceback.print_exc()


    def _on_refresh_data(self):
        """Refresh datasets and update signal readiness panel."""
        self._available_datasets.clear()
        self._scan_datasets()
        self._populate_signal_readiness()
        self.log_message("[REFRESH] Dataset scan complete")

    def _populate_signal_readiness(self):
        """Populate signal readiness rows from available datasets."""
        # Clear existing rows (keep header at row 0)
        for symbol, (sig_lbl, conf_lbl, stat_lbl) in self._signal_rows.items():
            sig_lbl.deleteLater()
            conf_lbl.deleteLater()
            stat_lbl.deleteLater()
        # Also remove the symbol name labels (column 0, rows 1+)
        for i in reversed(range(self._readiness_grid.count())):
            item = self._readiness_grid.itemAt(i)
            if item and item.widget():
                pos = self._readiness_grid.getItemPosition(i)
                if pos[0] > 0:  # Skip header row
                    item.widget().deleteLater()
        self._signal_rows.clear()

        # Get all symbols with data
        symbols = sorted(self._available_datasets.keys())
        if not symbols:
            row = 1
            no_data = QLabel("No datasets found. Use Data Pipeline tab to download data.")
            no_data.setStyleSheet(f"color: {COLORS['text_secondary']}; font-style: italic; font-size: 11px;")
            self._readiness_grid.addWidget(no_data, row, 0, 1, 4)
            return

        # Check which have trained models
        checkpoint_dir = Path("models/checkpoints")

        for i, symbol in enumerate(symbols):
            row = i + 1

            sym_label = QLabel(symbol)
            sym_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 11px;")
            self._readiness_grid.addWidget(sym_label, row, 0)

            signal_label = QLabel("--")
            signal_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
            self._readiness_grid.addWidget(signal_label, row, 1)

            conf_label = QLabel("--")
            conf_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
            self._readiness_grid.addWidget(conf_label, row, 2)

            # Check if any model exists (timeframe-specific or legacy)
            symbol_model_dir = checkpoint_dir / symbol
            has_model = False
            model_timeframes = []
            if symbol_model_dir.exists():
                for mf in symbol_model_dir.glob("ranknet_*.pt"):
                    tf = mf.stem.replace("ranknet_", "")
                    if tf != "model":
                        model_timeframes.append(tf)
                    else:
                        model_timeframes.append("1h")
                    has_model = True

            if has_model:
                tf_str = ", ".join(model_timeframes)
                status_label = QLabel(f"Model Ready ({tf_str})")
                status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 11px; font-weight: bold;")
            else:
                status_label = QLabel("No Model")
                status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px; font-style: italic;")

            self._readiness_grid.addWidget(status_label, row, 3)
            self._signal_rows[symbol] = (signal_label, conf_label, status_label)

    def _update_signal_readiness(self, symbol: str, direction: str, confidence: float):
        """Update a single row in the signal readiness panel after prediction."""
        if symbol not in self._signal_rows:
            return

        signal_label, conf_label, status_label = self._signal_rows[symbol]

        dir_colors = {
            'UP': COLORS['success'],
            'DOWN': COLORS['danger'],
            'NEUTRAL': COLORS['text_secondary'],
        }
        dir_arrows = {'UP': '▲ UP', 'DOWN': '▼ DOWN', 'NEUTRAL': '◆ NEUTRAL'}
        color = dir_colors.get(direction, COLORS['text_secondary'])

        signal_label.setText(dir_arrows.get(direction, direction))
        signal_label.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold;")

        conf_label.setText(f"{confidence * 100:.0f}%")
        conf_label.setStyleSheet(f"color: {color}; font-size: 11px;")

        status_label.setText("Ready")
        status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 11px; font-weight: bold;")

    # Public methods for external updates

    def log_message(self, message: str):
        """Append message to log"""
        self.log_output.append(message)
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_progress(self, current: int, total: int, message: str):
        """Update training progress"""
        if total > 0:
            self.progress_bar.setValue(int((current / total) * 100))
        self.progress_status.setText(message)

    def update_loss_curve(self, epoch: int, train_loss: float, val_loss: float):
        """Update loss curve chart"""
        self._epochs.append(epoch)
        self._train_loss_history.append(train_loss)
        self._val_loss_history.append(val_loss)

        colors = get_chart_colors()
        self.loss_ax.clear()
        self.loss_ax.set_facecolor(colors['background'])
        self.loss_ax.set_xlabel('Epoch', color=colors['text'])
        self.loss_ax.set_ylabel('Loss', color=colors['text'])
        self.loss_ax.set_title('Training & Validation Loss', color=colors['text'])
        self.loss_ax.tick_params(colors=colors['text'])
        self.loss_ax.grid(True, alpha=0.3, color=colors['grid'])

        self.loss_ax.plot(self._epochs, self._train_loss_history,
                         label='Train Loss', color=colors['primary'], linewidth=2)
        self.loss_ax.plot(self._epochs, self._val_loss_history,
                         label='Val Loss', color=colors['accent'], linewidth=2)

        self.loss_ax.legend(facecolor=colors['background'], edgecolor=colors['grid'],
                           labelcolor=colors['text'])

        self.loss_figure.tight_layout()
        self.loss_canvas.draw()

    def update_metrics(self, metrics: Dict):
        """Update displayed metrics"""
        parts = []
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                parts.append(f"{k}: {v:.4f}")
            else:
                parts.append(f"{k}: {v}")
        metrics_text = " | ".join(parts)
        self.metrics_label.setText(f"Metrics: {metrics_text}")

    def set_training_complete(self, success: bool, message: str, results: Optional[Dict] = None):
        """Handle training completion and display ICE metrics"""
        self._training_active = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100 if success else 0)
        self.progress_status.setText(message)
        self.log_message(f"[TRAINING] {'Completed' if success else 'Failed'}: {message}")

        # Update ICE display if results provided
        if success and results:
            self._update_ice_display(results)

    def _update_ice_display(self, results: Dict):
        """Update the ICE (Information Coefficient) display after training"""
        # Extract metrics from results
        avg_ic = results.get('avg_ic', 0)
        avg_sharpe = results.get('avg_sharpe_ratio', 0)
        avg_acc = results.get('avg_accuracy', 0)

        # Update IC value
        self.ic_value_label.setText(f"{avg_ic:.4f}")

        # Color and rate IC
        if avg_ic > 0.10:
            ic_color = COLORS['success']
            ic_rating = "EXCELLENT - Strong predictive signal"
        elif avg_ic > 0.05:
            ic_color = COLORS['chart_green']
            ic_rating = "GOOD - Meaningful predictive power"
        elif avg_ic > 0.02:
            ic_color = COLORS['accent']
            ic_rating = "MODERATE - Some predictive signal"
        elif avg_ic > 0:
            ic_color = COLORS['text_secondary']
            ic_rating = "WEAK - Marginal signal"
        else:
            ic_color = COLORS['danger']
            ic_rating = "NONE - No predictive power"

        self.ic_value_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {ic_color};")
        self.ic_rating_label.setText(ic_rating)
        self.ic_rating_label.setStyleSheet(f"color: {ic_color};")

        # Update Sharpe
        self.sharpe_value_label.setText(f"{avg_sharpe:.2f}")
        sharpe_color = COLORS['success'] if avg_sharpe > 1 else (COLORS['accent'] if avg_sharpe > 0 else COLORS['danger'])
        self.sharpe_value_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {sharpe_color};")

        # Update Accuracy
        self.test_acc_label.setText(f"{avg_acc:.1%}")
        acc_color = COLORS['success'] if avg_acc > 0.55 else (COLORS['accent'] if avg_acc > 0.50 else COLORS['danger'])
        self.test_acc_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {acc_color};")

        # Log detailed ICE info
        self.log_message(f"\n{'='*60}")
        self.log_message(f"TRAINING COMPLETE - ICE METRICS")
        self.log_message(f"{'='*60}")
        self.log_message(f"  Information Coefficient (IC): {avg_ic:.4f} [{ic_rating.split(' - ')[0]}]")
        self.log_message(f"  Sharpe Ratio:                 {avg_sharpe:.2f}")
        self.log_message(f"  Test Accuracy:                {avg_acc:.1%}")
        self.log_message(f"{'='*60}")

    def display_prediction(self, prediction_data: Dict):
        """Display prediction results with trading signals and risk management"""
        colors = get_chart_colors()

        self.pred_ax.clear()
        self.pred_ax.set_facecolor(colors['background'])
        self.pred_ax.set_title('10-Hour Price Prediction', color=colors['text'])
        self.pred_ax.set_xlabel('Hours Ahead', color=colors['text'])
        self.pred_ax.set_ylabel('Price (USD)', color=colors['text'])
        self.pred_ax.tick_params(colors=colors['text'])
        self.pred_ax.grid(True, alpha=0.3, color=colors['grid'])

        if 'predictions' in prediction_data:
            prices = prediction_data['predictions']
            hours = list(range(1, len(prices) + 1))
            self.pred_ax.plot(hours, prices,
                            color=colors['primary'], linewidth=2, marker='o')

            if 'confidence' in prediction_data:
                conf = prediction_data['confidence']
                self.pred_ax.fill_between(hours,
                                         [p - c for p, c in zip(prices, conf)],
                                         [p + c for p, c in zip(prices, conf)],
                                         alpha=0.2, color=colors['primary'])

            # Add price labels for first and last points
            if len(prices) >= 2:
                self.pred_ax.annotate(f'${prices[0]:,.0f}',
                                    (1, prices[0]),
                                    textcoords="offset points",
                                    xytext=(0, 10),
                                    ha='center',
                                    color=colors['text'],
                                    fontsize=9)
                self.pred_ax.annotate(f'${prices[-1]:,.0f}',
                                    (len(prices), prices[-1]),
                                    textcoords="offset points",
                                    xytext=(0, 10),
                                    ha='center',
                                    color=colors['text'],
                                    fontsize=9)

        self.pred_figure.tight_layout()
        self.pred_canvas.draw()

        # Extract prediction data
        current_price = prediction_data.get('current_price', 0)
        predicted_price = prediction_data.get('predicted_price', 0)
        price_change_pct = prediction_data.get('price_change_pct', 0)
        signal = prediction_data.get('signal', 'HOLD')
        symbol = prediction_data.get('symbol', '')

        # Update prediction label
        if current_price > 0 and predicted_price > 0:
            self.prediction_label.setText(
                f"{symbol}: ${current_price:,.2f} -> ${predicted_price:,.2f} ({price_change_pct:+.1f}%)"
            )

        # Update trading signal display
        self._update_trading_signal(signal, price_change_pct)

        # Update risk management levels
        self._update_risk_levels(current_price, predicted_price, price_change_pct)

    def _update_trading_signal(self, signal: str, price_change_pct: float):
        """Update trading signal display based on signal type"""
        signal_configs = {
            'STRONG_BUY': ("[BUY] STRONG BUY", COLORS['chart_green'], "High confidence upward trend"),
            'BUY': ("[BUY] BUY", "#81c784", "Moderate upward trend"),
            'STRONG_SELL': ("[SELL] STRONG SELL", COLORS['chart_red'], "High confidence downward trend"),
            'SELL': ("[SELL] SELL", "#e57373", "Moderate downward trend"),
            'HOLD': ("[NEUTRAL] HOLD", COLORS['text_secondary'], "Low movement expected"),
        }

        config = signal_configs.get(signal, signal_configs['HOLD'])
        signal_text, signal_color, recommendation = config

        # Update signal label
        display_text = f"{signal_text} - Expected: {price_change_pct:+.1f}%"
        self.signal_label.setText(display_text)
        self.signal_label.setStyleSheet(
            f"color: {signal_color}; font-size: 16px; font-weight: bold;"
        )

        self.log_message(f"[SIGNAL] {signal_text}: {recommendation} ({price_change_pct:+.1f}%)")

    def _update_risk_levels(self, current_price: float, predicted_price: float, price_change_pct: float):
        """Calculate and display risk management levels"""
        if current_price <= 0:
            self.risk_levels_label.setText("Risk levels: N/A (no price data)")
            return

        # Conservative strategy: -2% stop loss, take 95% of predicted gain
        stop_loss = current_price * 0.98

        if predicted_price > current_price:
            # Long position (price expected to rise)
            take_profit = current_price + (predicted_price - current_price) * 0.95
            position_type = "LONG"
        else:
            # Short position (price expected to fall)
            take_profit = current_price - (current_price - predicted_price) * 0.95
            position_type = "SHORT"

        # Calculate risk/reward ratio
        risk_amount = abs(current_price - stop_loss)
        reward_amount = abs(take_profit - current_price)
        risk_reward = reward_amount / risk_amount if risk_amount > 0 else 0

        # Calculate percentages
        stop_loss_pct = ((stop_loss - current_price) / current_price * 100)
        take_profit_pct = ((take_profit - current_price) / current_price * 100)

        # Format display text
        levels_text = (
            f"Risk Management ({position_type} Position):\n"
            f"  Stop-Loss: ${stop_loss:,.2f} ({stop_loss_pct:.1f}%)\n"
            f"  Take-Profit: ${take_profit:,.2f} ({take_profit_pct:+.1f}%)\n"
            f"  Risk/Reward: 1:{risk_reward:.2f}"
        )

        self.risk_levels_label.setText(levels_text)
        self.log_message(f"[RISK] Stop: ${stop_loss:.2f}, Target: ${take_profit:.2f}, R/R 1:{risk_reward:.2f}")


    def refresh_datasets(self):
        """Refresh available datasets"""
        self._available_datasets = {}
        self._scan_datasets()

    def clear(self):
        """Clear the panel state"""
        self._training_active = False
        self._train_loss_history = []
        self._val_loss_history = []
        self._epochs = []

        self.progress_bar.setValue(0)
        self.progress_status.setText("Ready to train")
        self.log_output.clear()

        # Clear charts
        colors = get_chart_colors()
        self.loss_ax.clear()
        self.loss_ax.set_facecolor(colors['background'])
        self.loss_canvas.draw()

        self.pred_ax.clear()
        self.pred_ax.set_facecolor(colors['background'])
        self.pred_canvas.draw()
