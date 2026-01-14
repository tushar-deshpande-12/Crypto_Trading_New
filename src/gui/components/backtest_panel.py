"""
Backtesting Panel Component
GUI for running and visualizing backtesting results
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from pathlib import Path
from typing import Optional, Dict, Any

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.core.config import AppConfig
from src.gui.styles import get_color


class BacktestPanel:
    """
    Backtesting Panel for model evaluation

    Features:
    - Load trained model
    - Configure backtest parameters
    - Run backtest on test data
    - Display performance metrics
    - Visualize equity curve and trade statistics
    """

    def __init__(
        self,
        parent: tk.Frame,
        on_run_backtest: Optional[callable] = None
    ):
        """
        Initialize Backtest Panel

        Args:
            parent: Parent frame
            on_run_backtest: Callback when backtest starts (model_path, config)
        """
        print(f"\n[BACKTEST_PANEL] Initializing Backtest Panel...")

        self.parent = parent
        self.on_run_backtest = on_run_backtest

        # State
        self.model_path_str = tk.StringVar(value="No model loaded")
        self.initial_capital_var = tk.StringVar(value="1000")
        self.trade_fee_var = tk.StringVar(value="0.1")
        self.confidence_threshold_var = tk.StringVar(value="1.0")

        self.backtest_running = False
        self.results = None

        # Create main frame
        self.frame = tk.Frame(parent, bg=get_color('bg_dark'))
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Create UI
        self._create_ui()

        print(f"[BACKTEST_PANEL] [OK] Backtest Panel initialized")

    def _create_ui(self):
        """Create all UI components"""
        print(f"[BACKTEST_PANEL] Creating UI...")

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

        # Configuration section
        self._create_config_section(scrollable_frame)

        # Results section
        self._create_results_section(scrollable_frame)

        print(f"[BACKTEST_PANEL] [OK] UI created")

    def _create_header(self, parent):
        """Create panel header"""
        header_frame = tk.Frame(parent, bg=get_color('bg_medium'))
        header_frame.pack(fill=tk.X, padx=0, pady=0)

        title = tk.Label(
            header_frame,
            text="📊 Model Backtesting",
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_HEADER, 'bold'),
            bg=get_color('bg_medium'),
            fg=get_color('text_primary')
        )
        title.pack(pady=(20, 5), padx=20)

        subtitle = tk.Label(
            header_frame,
            text="Test your model on historical data with realistic trading simulation",
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL),
            bg=get_color('bg_medium'),
            fg=get_color('text_secondary')
        )
        subtitle.pack(pady=(0, 20), padx=20)

    def _create_config_section(self, parent):
        """Create configuration section"""
        section = tk.LabelFrame(
            parent,
            text=" ⚙️ Backtest Configuration ",
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_LARGE, 'bold'),
            bg=get_color('bg_medium'),
            fg=get_color('text_primary'),
            padx=20,
            pady=15,
            borderwidth=0,
            relief='flat'
        )
        section.pack(fill=tk.X, padx=15, pady=(0, 12))

        # Model selection
        model_frame = tk.Frame(section, bg=get_color('bg_medium'))
        model_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            model_frame,
            text="Model:",
            bg=get_color('bg_medium'),
            fg=get_color('text_primary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold')
        ).pack(side=tk.LEFT, padx=(0, 10))

        tk.Entry(
            model_frame,
            textvariable=self.model_path_str,
            state='readonly',
            bg=get_color('bg_light'),
            fg=get_color('text_primary'),
            width=50
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        tk.Button(
            model_frame,
            text="Browse",
            command=self._browse_model,
            bg=get_color('bg_light'),
            fg=get_color('text_primary'),
            padx=10,
            pady=5
        ).pack(side=tk.LEFT)

        # Parameters
        params_frame = tk.Frame(section, bg=get_color('bg_medium'))
        params_frame.pack(fill=tk.X, pady=(15, 5))

        tk.Label(
            params_frame,
            text="Trading Parameters:",
            bg=get_color('bg_medium'),
            fg=get_color('text_primary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold')
        ).pack(anchor=tk.W, pady=(0, 10))

        # Initial capital
        param_grid = tk.Frame(params_frame, bg=get_color('bg_medium'))
        param_grid.pack(fill=tk.X)

        tk.Label(
            param_grid,
            text="Initial Capital ($):",
            width=20,
            anchor=tk.W,
            bg=get_color('bg_medium'),
            fg=get_color('text_primary')
        ).grid(row=0, column=0, sticky=tk.W, pady=5)

        tk.Entry(
            param_grid,
            textvariable=self.initial_capital_var,
            width=15,
            bg=get_color('bg_light'),
            fg=get_color('text_primary')
        ).grid(row=0, column=1, sticky=tk.W, padx=(0, 20))

        # Trade fee
        tk.Label(
            param_grid,
            text="Trade Fee (%):",
            width=20,
            anchor=tk.W,
            bg=get_color('bg_medium'),
            fg=get_color('text_primary')
        ).grid(row=1, column=0, sticky=tk.W, pady=5)

        tk.Entry(
            param_grid,
            textvariable=self.trade_fee_var,
            width=15,
            bg=get_color('bg_light'),
            fg=get_color('text_primary')
        ).grid(row=1, column=1, sticky=tk.W, padx=(0, 20))

        # Confidence threshold
        tk.Label(
            param_grid,
            text="Trade Threshold (%):",
            width=20,
            anchor=tk.W,
            bg=get_color('bg_medium'),
            fg=get_color('text_primary')
        ).grid(row=2, column=0, sticky=tk.W, pady=5)

        tk.Entry(
            param_grid,
            textvariable=self.confidence_threshold_var,
            width=15,
            bg=get_color('bg_light'),
            fg=get_color('text_primary')
        ).grid(row=2, column=1, sticky=tk.W, padx=(0, 20))

        # Help text
        help_text = tk.Label(
            params_frame,
            text="Trade threshold: Minimum predicted price change (%) to execute a trade",
            bg=get_color('bg_medium'),
            fg=get_color('text_secondary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_SMALL),
            wraplength=600,
            justify=tk.LEFT
        )
        help_text.pack(anchor=tk.W, pady=(10, 0))

        # Run button
        btn_frame = tk.Frame(section, bg=get_color('bg_medium'))
        btn_frame.pack(fill=tk.X, pady=(15, 0))

        self.run_btn = tk.Button(
            btn_frame,
            text="🚀 Run Backtest",
            command=self._run_backtest,
            bg=get_color('success'),
            fg='white',
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold'),
            padx=24,
            pady=12,
            relief=tk.FLAT,
            cursor="hand2"
        )
        self.run_btn.pack(side=tk.LEFT)

    def _create_results_section(self, parent):
        """Create results display section"""
        section = tk.LabelFrame(
            parent,
            text=" 📈 Backtest Results ",
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_LARGE, 'bold'),
            bg=get_color('bg_medium'),
            fg=get_color('text_primary'),
            padx=20,
            pady=15,
            borderwidth=0,
            relief='flat'
        )
        section.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        # Metrics display
        metrics_frame = tk.Frame(section, bg=get_color('bg_light'))
        metrics_frame.pack(fill=tk.X, pady=(0, 15), padx=10)

        # Create metric labels
        self.metric_labels = {}

        # Row 1: Portfolio Performance
        row1 = tk.Frame(metrics_frame, bg=get_color('bg_light'))
        row1.pack(fill=tk.X, pady=5, padx=10)

        tk.Label(
            row1,
            text="PORTFOLIO PERFORMANCE",
            bg=get_color('bg_light'),
            fg=get_color('text_primary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold')
        ).pack(anchor=tk.W, pady=(5, 5))

        for metric in ['Initial Capital', 'Final Capital', 'Total Return', 'Return %']:
            self._create_metric_label(row1, metric)

        # Row 2: Trading Statistics
        row2 = tk.Frame(metrics_frame, bg=get_color('bg_light'))
        row2.pack(fill=tk.X, pady=5, padx=10)

        tk.Label(
            row2,
            text="TRADING STATISTICS",
            bg=get_color('bg_light'),
            fg=get_color('text_primary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold')
        ).pack(anchor=tk.W, pady=(5, 5))

        for metric in ['Total Trades', 'Win Rate', 'P/L Ratio']:
            self._create_metric_label(row2, metric)

        # Row 3: Prediction Accuracy
        row3 = tk.Frame(metrics_frame, bg=get_color('bg_light'))
        row3.pack(fill=tk.X, pady=5, padx=10)

        tk.Label(
            row3,
            text="PREDICTION ACCURACY",
            bg=get_color('bg_light'),
            fg=get_color('text_primary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold')
        ).pack(anchor=tk.W, pady=(5, 5))

        for metric in ['MAE', 'RMSE', 'R² Score', 'Direction Accuracy']:
            self._create_metric_label(row3, metric)

        # Charts
        chart_frame = tk.Frame(section, bg=get_color('bg_medium'))
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8), facecolor=get_color('bg_medium'))
        self.canvas = FigureCanvasTkAgg(self.figure, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initial empty plot
        self._show_placeholder()

    def _create_metric_label(self, parent, metric_name):
        """Create a metric label"""
        frame = tk.Frame(parent, bg=get_color('bg_light'))
        frame.pack(fill=tk.X, pady=2)

        tk.Label(
            frame,
            text=f"{metric_name}:",
            width=20,
            anchor=tk.W,
            bg=get_color('bg_light'),
            fg=get_color('text_primary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_SMALL)
        ).pack(side=tk.LEFT)

        value_label = tk.Label(
            frame,
            text="--",
            anchor=tk.W,
            bg=get_color('bg_light'),
            fg=get_color('accent'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_SMALL, 'bold')
        )
        value_label.pack(side=tk.LEFT)

        self.metric_labels[metric_name] = value_label

    def _show_placeholder(self):
        """Show placeholder when no results"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Run backtest to see results',
                ha='center', va='center',
                transform=ax.transAxes,
                fontsize=16, color=get_color('text_secondary'))
        ax.axis('off')
        self.canvas.draw()

    def _browse_model(self):
        """Browse for model file"""
        path = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[
                ("Model files", "*.ckpt *.pth *.pt"),
                ("All files", "*.*")
            ],
            initialdir="models/checkpoints"
        )
        if path:
            self.model_path_str.set(path)

    def _run_backtest(self):
        """Run backtest"""
        if self.model_path_str.get() == "No model loaded":
            messagebox.showwarning("No Model", "Please select a model first")
            return

        try:
            config = {
                'model_path': self.model_path_str.get(),
                'initial_capital': float(self.initial_capital_var.get()),
                'trade_fee': float(self.trade_fee_var.get()) / 100,
                'confidence_threshold': float(self.confidence_threshold_var.get()) / 100
            }

            self.run_btn.config(state=tk.DISABLED)
            self.backtest_running = True

            if self.on_run_backtest:
                self.on_run_backtest(config)

        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please check your input values:\n{e}")
            self.run_btn.config(state=tk.NORMAL)

    def display_results(self, results: Dict[str, Any]):
        """
        Display backtest results

        Args:
            results: Dictionary with backtest results
        """
        # Update metrics
        self.metric_labels['Initial Capital'].config(
            text=f"${results.get('initial_capital', 0):,.2f}"
        )
        self.metric_labels['Final Capital'].config(
            text=f"${results.get('final_capital', 0):,.2f}"
        )
        self.metric_labels['Total Return'].config(
            text=f"${results.get('total_return', 0):+,.2f}"
        )

        return_pct = results.get('total_return_pct', 0)
        self.metric_labels['Return %'].config(
            text=f"{return_pct:+.2f}%",
            fg=get_color('success') if return_pct > 0 else get_color('danger')
        )

        self.metric_labels['Total Trades'].config(
            text=str(results.get('num_trades', 0))
        )
        self.metric_labels['Win Rate'].config(
            text=f"{results.get('win_rate', 0):.1f}%"
        )
        self.metric_labels['P/L Ratio'].config(
            text=f"{results.get('profit_loss_ratio', 0):.2f}"
        )

        self.metric_labels['MAE'].config(
            text=f"{results.get('prediction_mae', 0):.4f}"
        )
        self.metric_labels['RMSE'].config(
            text=f"{results.get('prediction_rmse', 0):.4f}"
        )
        self.metric_labels['R² Score'].config(
            text=f"{results.get('prediction_r2', 0):.4f}"
        )
        self.metric_labels['Direction Accuracy'].config(
            text=f"{results.get('direction_accuracy', 0):.1f}%"
        )

        # Update chart (plot will be generated by the backend)
        self.run_btn.config(state=tk.NORMAL)
        self.backtest_running = False

    def update_chart(self, figure):
        """Update chart with new figure"""
        self.figure = figure
        self.canvas.figure = figure
        self.canvas.draw()

    def update_chart_from_file(self, image_path: str):
        """Update chart by loading image from file"""
        try:
            from PIL import Image
            import matplotlib.pyplot as plt
            from pathlib import Path

            # Check if file exists
            if not Path(image_path).exists():
                print(f"[BACKTEST_PANEL] Warning: Chart image not found: {image_path}")
                return

            # Clear current figure
            self.figure.clear()

            # Load image and display
            img = Image.open(image_path)
            ax = self.figure.add_subplot(111)
            ax.imshow(img)
            ax.axis('off')  # Hide axes

            # Redraw canvas
            self.canvas.draw()

            print(f"[BACKTEST_PANEL] Chart updated from {image_path}")

        except Exception as e:
            print(f"[BACKTEST_PANEL] Error updating chart from file: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Test
    root = tk.Tk()
    root.title("Backtest Panel Test")
    root.geometry("1000x800")

    panel = BacktestPanel(root)

    root.mainloop()
