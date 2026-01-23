"""
ML Panel Component - AI Model Training and Prediction Interface

Provides PyQt6 interface for:
- Dataset configuration and selection
- Model hyperparameter configuration
- Training progress with loss curves
- Prediction and inference
- Model testing and validation
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QProgressBar, QGroupBox, QLineEdit, QDoubleSpinBox,
    QSpinBox, QTextEdit, QSplitter, QFrame, QGridLayout, QCheckBox,
    QScrollArea, QSlider, QRadioButton, QButtonGroup, QFileDialog,
    QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
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
    5. Model Testing - Validation metrics and visualization

    Signals:
        train_requested(dict, list): Config dict and selected symbols when training starts
        predict_requested(str, str): Model path and symbol for prediction
        model_test_requested(str, str): Model path and symbol for testing
    """

    train_requested = pyqtSignal(dict, list)
    predict_requested = pyqtSignal(str, str)
    model_test_requested = pyqtSignal(str, str)
    stop_training_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._training_active = False
        self._available_datasets = {}
        self._dataset_vars = {}
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

        subtitle = QLabel("LSTM-based AI for cryptocurrency direction prediction")
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

        # Section 5: Model Testing
        testing_section = self._create_testing_section()
        layout.addWidget(testing_section)

        layout.addStretch()

        scroll.setWidget(content)
        main_layout.addWidget(scroll)

    def _create_dataset_section(self) -> QGroupBox:
        """Create dataset configuration section"""
        group = QGroupBox("Dataset Configuration")
        layout = QVBoxLayout(group)

        # Dataset selection label
        layout.addWidget(QLabel("Select Datasets for Training:"))

        # Dataset checkboxes container
        self.datasets_container = QWidget()
        self.datasets_layout = QVBoxLayout(self.datasets_container)
        self.datasets_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.datasets_container)

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
            ('hidden_size', 'Hidden Size', 32, 8, 512),
            ('lstm_layers', 'LSTM Layers', 1, 1, 8),
            ('dropout', 'Dropout', 0.4, 0.0, 0.9),
            ('batch_size', 'Batch Size', 128, 16, 512),
            ('max_epochs', 'Max Epochs', 50, 1, 500),
            ('learning_rate', 'Learning Rate', 0.0005, 0.0001, 0.01),
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
        select_row.addWidget(self.predict_symbol_combo)

        self.predict_btn = QPushButton("Get AI Prediction")
        self.predict_btn.setStyleSheet(f"background-color: {COLORS['primary']}; color: white; font-weight: bold; padding: 12px 24px;")
        self.predict_btn.clicked.connect(self._generate_ai_prediction)
        select_row.addWidget(self.predict_btn)
        select_row.addStretch()
        layout.addLayout(select_row)

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

    def _create_testing_section(self) -> QGroupBox:
        """Create model testing section"""
        group = QGroupBox("Model Testing & Validation")
        layout = QVBoxLayout(group)

        # Test controls
        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("Test on historical data:"))

        test_btn = QPushButton("Run Model Test")
        test_btn.setStyleSheet(f"background-color: {COLORS['warning']}; color: white; font-weight: bold; padding: 10px 24px;")
        test_btn.clicked.connect(self._run_model_test)
        controls_row.addWidget(test_btn)
        controls_row.addStretch()
        layout.addLayout(controls_row)

        # Metrics grid
        metrics_frame = QFrame()
        metrics_frame.setFrameShape(QFrame.Shape.StyledPanel)
        metrics_layout = QGridLayout(metrics_frame)

        self.test_metrics = {}
        metrics = [
            ('mae', 'MAE'),
            ('rmse', 'RMSE'),
            ('dir_acc', 'Dir Accuracy'),
            ('test_loss', 'Test Loss'),
        ]

        for i, (key, label) in enumerate(metrics):
            metrics_layout.addWidget(QLabel(f"{label}:"), i // 2, (i % 2) * 2)
            val_label = QLabel("--")
            val_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
            self.test_metrics[key] = val_label
            metrics_layout.addWidget(val_label, i // 2, (i % 2) * 2 + 1)

        layout.addWidget(metrics_frame)

        # Test results chart with multiple subplots
        colors = get_chart_colors()
        self.test_figure = Figure(figsize=(14, 10), facecolor=colors['background'])
        gs = GridSpec(3, 2, figure=self.test_figure, hspace=0.3, wspace=0.3)

        # Actual vs Predicted
        self.test_ax1 = self.test_figure.add_subplot(gs[0, :])
        self.test_ax1.set_facecolor(colors['background'])
        self.test_ax1.set_title('Actual vs Predicted (Test Set)', color=colors['text'])
        self.test_ax1.tick_params(colors=colors['text'])
        self.test_ax1.grid(True, alpha=0.3, color=colors['grid'])

        # Direction accuracy over time
        self.test_ax2 = self.test_figure.add_subplot(gs[1, 0])
        self.test_ax2.set_facecolor(colors['background'])
        self.test_ax2.tick_params(colors=colors['text'])

        # Error distribution
        self.test_ax3 = self.test_figure.add_subplot(gs[1, 1])
        self.test_ax3.set_facecolor(colors['background'])
        self.test_ax3.tick_params(colors=colors['text'])

        # Confusion matrix
        self.test_ax4 = self.test_figure.add_subplot(gs[2, 0])
        self.test_ax4.set_facecolor(colors['background'])
        self.test_ax4.tick_params(colors=colors['text'])

        # Metrics summary
        self.test_ax5 = self.test_figure.add_subplot(gs[2, 1])
        self.test_ax5.set_facecolor(colors['background'])

        self.test_canvas = FigureCanvas(self.test_figure)
        self.test_canvas.setMinimumHeight(500)
        layout.addWidget(self.test_canvas)

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
                self._update_dataset_checkboxes()
                return

            for symbol_dir in dataset_dir.iterdir():
                if symbol_dir.is_dir() and symbol_dir.name not in ['.', '..']:
                    datasets = [d for d in symbol_dir.iterdir() if d.is_dir()]

                    if datasets:
                        latest = sorted(datasets)[-1]
                        metadata_path = latest / 'metadata.json'

                        if metadata_path.exists():
                            with open(metadata_path) as f:
                                metadata = json.load(f)

                            self._available_datasets[symbol_dir.name] = {
                                'path': latest,
                                'candles': metadata.get('candle_count', 0),
                                'metadata': metadata
                            }

            self._update_dataset_checkboxes()
            self._update_symbol_combo()

        except Exception as e:
            print(f"Failed to scan datasets: {e}")

    def _update_dataset_checkboxes(self):
        """Update dataset checkboxes"""
        # Clear existing
        while self.datasets_layout.count():
            item = self.datasets_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._available_datasets:
            no_data = QLabel("No datasets found. Use Data Pipeline to download cryptocurrency data.")
            no_data.setProperty("class", "secondary")
            no_data.setStyleSheet("font-style: italic;")
            self.datasets_layout.addWidget(no_data)
            return

        self._dataset_vars = {}
        for symbol, info in self._available_datasets.items():
            cb = QCheckBox(f"{symbol} ({info['candles']:,} candles)")
            cb.setChecked(True)
            self._dataset_vars[symbol] = cb
            self.datasets_layout.addWidget(cb)

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

        # Get all available symbols from datasets
        all_symbols = set(self._available_datasets.keys())

        # Combine: trained models first, then others
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
            'hidden_size': 32,
            'lstm_layers': 1,
            'dropout': 0.4,
            'batch_size': 128,
            'max_epochs': 50,
            'learning_rate': 0.0005,
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
        """Start model training"""
        selected = [symbol for symbol, cb in self._dataset_vars.items() if cb.isChecked()]

        if not selected:
            QMessageBox.warning(self, "No Datasets", "Please select at least one dataset for training.")
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

        self.log_message(f"[TRAINING] Starting training with {len(selected)} symbols")
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

    def _generate_ai_prediction(self):
        """Generate AI prediction for investment direction"""
        # Get actual symbol (not display text which may include "[AI Ready]")
        symbol = self.predict_symbol_combo.currentData()
        if symbol is None:
            symbol = self.predict_symbol_combo.currentText().replace(" [AI Ready]", "")
        if not symbol:
            QMessageBox.warning(self, "No Symbol", "Please select a cryptocurrency.")
            return

        try:
            from pathlib import Path
            from datetime import datetime
            import numpy as np
            import pandas as pd

            # Find model for this symbol
            model_path = Path(f"models/checkpoints/{symbol}/simple_model.pt")

            if not model_path.exists():
                # Try to find any trained model
                checkpoint_dir = Path("models/checkpoints")
                symbol_dirs = list(checkpoint_dir.glob(f"{symbol}*"))
                if symbol_dirs:
                    possible_models = list(symbol_dirs[0].glob("*.pt"))
                    if possible_models:
                        model_path = possible_models[0]

            if not model_path.exists():
                self.direction_label.setText("?")
                self.direction_text.setText(f"No trained model found for {symbol}. Train a model first.")
                self.direction_label.setStyleSheet("font-size: 48px; font-weight: bold; color: #888;")
                return

            # Load model
            from src.ml.models.simple_model import SimpleTrainer

            trainer = SimpleTrainer.load(str(model_path))

            # Load recent data for this symbol
            from src.core.config import AppConfig
            dataset_dir = Path(AppConfig.DATASET_DIR)
            symbol_data_dirs = list(dataset_dir.glob(f"{symbol}/*"))

            if not symbol_data_dirs:
                self.direction_label.setText("?")
                self.direction_text.setText(f"No data found for {symbol}.")
                return

            # Find largest/most recent data file
            csv_files = []
            for d in symbol_data_dirs:
                csv_files.extend(list(d.glob("*.csv")))

            if not csv_files:
                self.direction_label.setText("?")
                self.direction_text.setText(f"No CSV data found for {symbol}.")
                return

            largest_file = max(csv_files, key=lambda f: f.stat().st_size)
            df = pd.read_csv(largest_file)

            # Use last portion for prediction
            df = df.tail(500)  # Last 500 rows

            # Prepare features
            from src.ml.models.simple_model import create_features, get_feature_columns

            df = create_features(df)
            feature_cols = get_feature_columns(df)
            df = df.dropna()

            if len(df) < trainer.config.sequence_length + 10:
                self.direction_label.setText("?")
                self.direction_text.setText("Insufficient data for prediction.")
                return

            # Get features for last sequence
            features = df[feature_cols].values.astype(np.float32)

            # Handle inf/nan
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            features = np.clip(features, -10, 10)

            # Normalize with saved scaler
            features = trainer.scaler.transform(features)

            # Get last sequence
            seq_len = trainer.config.sequence_length
            last_seq = features[-seq_len:]

            # Predict
            import torch
            trainer.model.eval()
            with torch.no_grad():
                x = torch.FloatTensor(last_seq).unsqueeze(0)
                logits = trainer.model(x)
                prob = torch.sigmoid(logits).item()

            # Determine direction
            if prob > 0.55:
                direction = "UP"
                arrow = "▲"
                color = COLORS['success']
                recommendation = f"AI suggests {symbol} may go UP in next 4 hours"
            elif prob < 0.45:
                direction = "DOWN"
                arrow = "▼"
                color = COLORS['danger']
                recommendation = f"AI suggests {symbol} may go DOWN in next 4 hours"
            else:
                direction = "NEUTRAL"
                arrow = "◆"
                color = COLORS['text_secondary']
                recommendation = f"AI is uncertain about {symbol} direction"

            # Update UI
            self.direction_label.setText(f"{arrow} {direction}")
            self.direction_label.setStyleSheet(f"font-size: 48px; font-weight: bold; color: {color};")
            self.direction_text.setText(recommendation)
            self.direction_text.setStyleSheet(f"font-size: 18px; color: {color};")

            # Confidence (distance from 0.5)
            confidence = abs(prob - 0.5) * 200  # Scale to 0-100
            self.confidence_bar.setValue(int(confidence))
            self.confidence_label.setText(f"{confidence:.0f}%")

            # Update confidence bar color based on direction
            if direction == "UP":
                self.confidence_bar.setStyleSheet("""
                    QProgressBar { border: 1px solid #555; border-radius: 4px; text-align: center; }
                    QProgressBar::chunk { background-color: #4CAF50; }
                """)
            elif direction == "DOWN":
                self.confidence_bar.setStyleSheet("""
                    QProgressBar { border: 1px solid #555; border-radius: 4px; text-align: center; }
                    QProgressBar::chunk { background-color: #F44336; }
                """)
            else:
                self.confidence_bar.setStyleSheet("""
                    QProgressBar { border: 1px solid #555; border-radius: 4px; text-align: center; }
                    QProgressBar::chunk { background-color: #888; }
                """)

            # Update details
            self.horizon_label.setText("4 hours ahead")
            self.accuracy_label.setText(f"~51-53%")
            self.ic_label.setText(f"~0.02-0.06")
            self.last_update_label.setText(datetime.now().strftime("%H:%M:%S"))

            self.log_message(f"[PREDICTION] {symbol}: {direction} (prob={prob:.3f}, conf={confidence:.0f}%)")

        except Exception as e:
            self.direction_label.setText("!")
            self.direction_text.setText(f"Error: {str(e)}")
            self.direction_label.setStyleSheet(f"font-size: 48px; font-weight: bold; color: {COLORS['danger']};")
            import traceback
            traceback.print_exc()

    def _run_model_test(self):
        """Run model test"""
        model_path = self.model_path_input.text()
        symbol = self.predict_symbol_combo.currentData() or self.predict_symbol_combo.currentText().replace(" [AI Ready]", "")

        if not model_path or model_path == "No model loaded":
            QMessageBox.warning(self, "No Model", "Please load a model first.")
            return

        if not symbol:
            QMessageBox.warning(self, "No Symbol", "Please select a symbol.")
            return

        self.model_test_requested.emit(model_path, symbol)

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

    def set_training_complete(self, success: bool, message: str):
        """Handle training completion"""
        self._training_active = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100 if success else 0)
        self.progress_status.setText(message)
        self.log_message(f"[TRAINING] {'Completed' if success else 'Failed'}: {message}")

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

    def display_test_results(self, results: Dict):
        """Display model test results"""
        # Update metrics labels
        for key, label in self.test_metrics.items():
            if key in results:
                label.setText(f"{results[key]:.4f}")

        colors = get_chart_colors()

        # Clear all axes
        for ax in [self.test_ax1, self.test_ax2, self.test_ax3, self.test_ax4, self.test_ax5]:
            ax.clear()
            ax.set_facecolor(colors['background'])
            ax.tick_params(colors=colors['text'])

        # Plot actual vs predicted
        if 'actual' in results and 'predicted' in results:
            self.test_ax1.plot(results['actual'], label='Actual',
                              color=colors['positive'], linewidth=1.5)
            self.test_ax1.plot(results['predicted'], label='Predicted',
                              color=colors['primary'], linewidth=1.5)
            self.test_ax1.set_title('Actual vs Predicted', color=colors['text'])
            self.test_ax1.legend(facecolor=colors['background'], labelcolor=colors['text'])
            self.test_ax1.grid(True, alpha=0.3, color=colors['grid'])

        # Direction accuracy over time
        if 'dir_acc_rolling' in results:
            self.test_ax2.plot(results['dir_acc_rolling'], color=colors['accent'])
            self.test_ax2.axhline(0.5, linestyle='--', color=colors['grid'], alpha=0.7)
            self.test_ax2.set_title('Direction Accuracy (Rolling)', color=colors['text'])
            self.test_ax2.set_ylabel('Accuracy', color=colors['text'])
            self.test_ax2.grid(True, alpha=0.3, color=colors['grid'])

        # Error distribution
        if 'errors' in results:
            self.test_ax3.hist(results['errors'], bins=50, color=colors['primary'], alpha=0.7)
            self.test_ax3.set_title('Prediction Error Distribution', color=colors['text'])
            self.test_ax3.set_xlabel('Error', color=colors['text'])
            self.test_ax3.grid(True, alpha=0.3, color=colors['grid'])

        self.test_figure.tight_layout()
        self.test_canvas.draw()

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
