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
        self.backtest_mode = tk.StringVar(value="signals")  # "signals" or "model"
        self.symbol_var = tk.StringVar(value="BTCUSDT")

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
            text=" Backtest Configuration ",
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_LARGE, 'bold'),
            bg=get_color('bg_medium'),
            fg=get_color('text_primary'),
            padx=20,
            pady=15,
            borderwidth=0,
            relief='flat'
        )
        section.pack(fill=tk.X, padx=15, pady=(0, 12))

        # MODE SELECTOR (NEW!)
        mode_frame = tk.Frame(section, bg=get_color('bg_medium'))
        mode_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(
            mode_frame,
            text="Backtest Mode:",
            bg=get_color('bg_medium'),
            fg=get_color('text_primary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold')
        ).pack(side=tk.LEFT, padx=(0, 15))

        tk.Radiobutton(
            mode_frame,
            text="Technical Signals (No AI)",
            variable=self.backtest_mode,
            value="signals",
            bg=get_color('bg_medium'),
            fg=get_color('text_primary'),
            selectcolor=get_color('bg_light'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL),
            command=self._update_mode_ui
        ).pack(side=tk.LEFT, padx=(0, 20))

        tk.Radiobutton(
            mode_frame,
            text="AI Model Predictions",
            variable=self.backtest_mode,
            value="model",
            bg=get_color('bg_medium'),
            fg=get_color('text_primary'),
            selectcolor=get_color('bg_light'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL),
            command=self._update_mode_ui
        ).pack(side=tk.LEFT)

        # Symbol selection (for signal mode)
        self.symbol_frame = tk.Frame(section, bg=get_color('bg_medium'))
        self.symbol_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            self.symbol_frame,
            text="Symbol:",
            bg=get_color('bg_medium'),
            fg=get_color('text_primary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold')
        ).pack(side=tk.LEFT, padx=(0, 10))

        symbol_combo = ttk.Combobox(
            self.symbol_frame,
            textvariable=self.symbol_var,
            values=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"],
            state='readonly',
            width=20
        )
        symbol_combo.pack(side=tk.LEFT)

        # Model selection (hidden in signals mode)
        self.model_frame = tk.Frame(section, bg=get_color('bg_medium'))
        self.model_frame.pack(fill=tk.X, pady=5)
        self.model_frame.pack_forget()  # Hide by default (signals mode)

        tk.Label(
            self.model_frame,
            text="Model:",
            bg=get_color('bg_medium'),
            fg=get_color('text_primary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold')
        ).pack(side=tk.LEFT, padx=(0, 10))

        tk.Entry(
            self.model_frame,
            textvariable=self.model_path_str,
            state='readonly',
            bg=get_color('bg_light'),
            fg=get_color('text_primary'),
            width=50
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        tk.Button(
            self.model_frame,
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

    def _update_mode_ui(self):
        """Show/hide UI elements based on mode"""
        if self.backtest_mode.get() == "signals":
            self.symbol_frame.pack(fill=tk.X, pady=5)
            self.model_frame.pack_forget()
        else:
            self.symbol_frame.pack_forget()
            self.model_frame.pack(fill=tk.X, pady=5)

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
        mode = self.backtest_mode.get()

        if mode == "model" and self.model_path_str.get() == "No model loaded":
            messagebox.showwarning("No Model", "Please select a model first")
            return

        try:
            config = {
                'mode': mode,
                'symbol': self.symbol_var.get(),
                'model_path': self.model_path_str.get() if mode == "model" else None,
                'initial_capital': float(self.initial_capital_var.get()),
                'trade_fee': float(self.trade_fee_var.get()) / 100,
                'confidence_threshold': float(self.confidence_threshold_var.get()) / 100
            }

            self.run_btn.config(state=tk.DISABLED)
            self.backtest_running = True

            if mode == "signals":
                # Run signal-based backtest locally
                self._run_signal_backtest(config)
            else:
                # Run model-based backtest via callback
                if self.on_run_backtest:
                    self.on_run_backtest(config)

        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please check your input values:\n{e}")
            self.run_btn.config(state=tk.NORMAL)

    def _run_signal_backtest(self, config: Dict[str, Any]):
        """Run backtest using technical indicator signals (NO AI REQUIRED)"""
        import threading

        def backtest_thread():
            try:
                print(f"\n[BACKTEST_PANEL] Starting signal backtest for {config['symbol']}...")

                from src.ml.preprocessing.preprocessor import CryptoPreprocessor
                from src.utils.debug_logger import debug_log
                import numpy as np
                import pandas as pd

                # Load data
                preprocessor = CryptoPreprocessor("dataset")
                df = preprocessor.load_symbol_data(config['symbol'])

                if df is None or len(df) < 200:
                    self.parent.after(0, lambda: messagebox.showerror("Error", "Insufficient data for backtest"))
                    self.parent.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
                    return

                df = df.sort_values('datetime').tail(1000).reset_index(drop=True)
                print(f"[BACKTEST_PANEL]   - Using {len(df)} candles for backtest")

                # Calculate technical indicators
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = -delta.where(delta < 0, 0).rolling(14).mean()
                rs = gain / loss.replace(0, np.nan)
                df['rsi'] = 100 - (100 / (1 + rs))

                ema_12 = df['close'].ewm(span=12).mean()
                ema_26 = df['close'].ewm(span=26).mean()
                df['macd'] = ema_12 - ema_26
                df['macd_signal'] = df['macd'].ewm(span=9).mean()

                df['sma_10'] = df['close'].rolling(10).mean()
                df['sma_20'] = df['close'].rolling(20).mean()

                # Generate signals
                df = df.dropna().reset_index(drop=True)
                signals = []

                for i in range(len(df)):
                    row = df.iloc[i]
                    score = 0

                    # RSI
                    if row['rsi'] < 30:
                        score += 2
                    elif row['rsi'] < 40:
                        score += 1
                    elif row['rsi'] > 70:
                        score -= 2
                    elif row['rsi'] > 60:
                        score -= 1

                    # MACD
                    if row['macd'] > 0 and row['macd'] > row['macd_signal']:
                        score += 2
                    elif row['macd'] > row['macd_signal']:
                        score += 1
                    elif row['macd'] < 0 and row['macd'] < row['macd_signal']:
                        score -= 2
                    elif row['macd'] < row['macd_signal']:
                        score -= 1

                    # Trend
                    if row['sma_10'] > row['sma_20'] * 1.01:
                        score += 2
                    elif row['sma_10'] > row['sma_20']:
                        score += 1
                    elif row['sma_10'] < row['sma_20'] * 0.99:
                        score -= 2
                    elif row['sma_10'] < row['sma_20']:
                        score -= 1

                    signals.append(score)

                df['signal_score'] = signals

                # Run backtest simulation
                capital = config['initial_capital']
                position = 0  # 0 = no position, 1 = long
                entry_price = 0
                trades = []
                equity_curve = [capital]

                for i in range(1, len(df)):
                    current_price = df['close'].iloc[i]
                    score = df['signal_score'].iloc[i]

                    # Trading logic
                    if position == 0:  # No position
                        if score >= 3:  # Strong buy signal
                            position = 1
                            entry_price = current_price
                            capital -= capital * config['trade_fee']
                            trades.append({'type': 'BUY', 'price': current_price, 'idx': i})
                    else:  # In position
                        if score <= -2:  # Sell signal
                            pnl = (current_price - entry_price) / entry_price
                            capital = capital * (1 + pnl) * (1 - config['trade_fee'])
                            trades.append({'type': 'SELL', 'price': current_price, 'idx': i,
                                          'pnl': pnl * 100, 'win': pnl > 0})
                            position = 0

                    equity_curve.append(capital if position == 0 else capital * (1 + (current_price - entry_price) / entry_price))

                # Close final position if open
                if position == 1:
                    final_price = df['close'].iloc[-1]
                    pnl = (final_price - entry_price) / entry_price
                    capital = capital * (1 + pnl) * (1 - config['trade_fee'])
                    trades.append({'type': 'SELL', 'price': final_price, 'idx': len(df)-1,
                                  'pnl': pnl * 100, 'win': pnl > 0})

                # Calculate metrics
                sell_trades = [t for t in trades if t['type'] == 'SELL']
                wins = [t for t in sell_trades if t.get('win', False)]
                losses = [t for t in sell_trades if not t.get('win', True)]

                results = {
                    'initial_capital': config['initial_capital'],
                    'final_capital': capital,
                    'total_return': capital - config['initial_capital'],
                    'total_return_pct': ((capital / config['initial_capital']) - 1) * 100,
                    'num_trades': len(sell_trades),
                    'win_rate': (len(wins) / len(sell_trades) * 100) if sell_trades else 0,
                    'profit_loss_ratio': (np.mean([t['pnl'] for t in wins]) / abs(np.mean([t['pnl'] for t in losses]))) if wins and losses else 0,
                    'prediction_mae': 0,  # N/A for signal backtest
                    'prediction_rmse': 0,
                    'prediction_r2': 0,
                    'direction_accuracy': (len(wins) / len(sell_trades) * 100) if sell_trades else 0,
                    'equity_curve': equity_curve,
                    'trades': trades,
                    'prices': df['close'].tolist()
                }

                print(f"[BACKTEST_PANEL]   - Final capital: ${capital:,.2f}")
                print(f"[BACKTEST_PANEL]   - Return: {results['total_return_pct']:.2f}%")
                print(f"[BACKTEST_PANEL]   - Trades: {len(sell_trades)}, Win rate: {results['win_rate']:.1f}%")

                # Log to debug
                debug_log("signal_backtest", "complete", {
                    "symbol": config['symbol'],
                    "initial_capital": config['initial_capital'],
                    "final_capital": capital,
                    "return_pct": results['total_return_pct'],
                    "num_trades": len(sell_trades),
                    "win_rate": results['win_rate']
                })

                # Update UI on main thread
                self.parent.after(0, lambda: self.display_results(results))
                self.parent.after(0, lambda: self._plot_signal_backtest_results(results, df))
                self.parent.after(0, lambda: self.run_btn.config(state=tk.NORMAL))

            except Exception as e:
                print(f"[BACKTEST_PANEL] [X] Signal backtest failed: {e}")
                import traceback
                traceback.print_exc()
                self.parent.after(0, lambda: messagebox.showerror("Backtest Error", str(e)))
                self.parent.after(0, lambda: self.run_btn.config(state=tk.NORMAL))

        # Run in background thread
        thread = threading.Thread(target=backtest_thread, daemon=True)
        thread.start()

    def _plot_signal_backtest_results(self, results: Dict[str, Any], df):
        """Plot signal backtest results"""
        try:
            import numpy as np

            self.figure.clear()

            # Create subplots
            ax1 = self.figure.add_subplot(2, 2, 1)
            ax2 = self.figure.add_subplot(2, 2, 2)
            ax3 = self.figure.add_subplot(2, 2, 3)
            ax4 = self.figure.add_subplot(2, 2, 4)

            # 1. Equity Curve
            equity = results.get('equity_curve', [])
            ax1.plot(equity, color='#4fc3f7', linewidth=2)
            ax1.axhline(y=results['initial_capital'], color='gray', linestyle='--', alpha=0.5)
            ax1.set_title('Equity Curve', color=get_color('text_primary'), fontweight='bold')
            ax1.set_xlabel('Time', color=get_color('text_primary'))
            ax1.set_ylabel('Capital ($)', color=get_color('text_primary'))
            ax1.grid(True, alpha=0.3)
            ax1.set_facecolor(get_color('bg_light'))

            # 2. Price with trade markers
            prices = results.get('prices', df['close'].tolist() if df is not None else [])
            trades = results.get('trades', [])

            ax2.plot(prices, color='#9e9e9e', linewidth=1, alpha=0.7, label='Price')
            for trade in trades:
                idx = trade.get('idx', 0)
                if idx < len(prices):
                    if trade['type'] == 'BUY':
                        ax2.scatter(idx, prices[idx], color='#4caf50', s=50, marker='^', zorder=5)
                    else:
                        color = '#4caf50' if trade.get('win', False) else '#f44336'
                        ax2.scatter(idx, prices[idx], color=color, s=50, marker='v', zorder=5)

            ax2.set_title('Price & Trades', color=get_color('text_primary'), fontweight='bold')
            ax2.set_xlabel('Time', color=get_color('text_primary'))
            ax2.set_ylabel('Price ($)', color=get_color('text_primary'))
            ax2.grid(True, alpha=0.3)
            ax2.set_facecolor(get_color('bg_light'))

            # 3. Trade P&L Distribution
            sell_trades = [t for t in trades if t['type'] == 'SELL']
            if sell_trades:
                pnls = [t.get('pnl', 0) for t in sell_trades]
                colors = ['#4caf50' if p > 0 else '#f44336' for p in pnls]
                ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
                ax3.axhline(y=0, color='white', linewidth=1)
            ax3.set_title('Trade P&L (%)', color=get_color('text_primary'), fontweight='bold')
            ax3.set_xlabel('Trade #', color=get_color('text_primary'))
            ax3.set_ylabel('P&L (%)', color=get_color('text_primary'))
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.set_facecolor(get_color('bg_light'))

            # 4. Summary Text
            ax4.axis('off')
            summary = f"""
SIGNAL BACKTEST RESULTS
{'='*35}

Initial Capital:  ${results['initial_capital']:,.2f}
Final Capital:    ${results['final_capital']:,.2f}
Total Return:     {results['total_return_pct']:+.2f}%

Total Trades:     {results['num_trades']}
Win Rate:         {results['win_rate']:.1f}%
P/L Ratio:        {results['profit_loss_ratio']:.2f}

Strategy:         Technical Signals
  - RSI (14)
  - MACD (12/26/9)
  - SMA Crossover (10/20)
"""
            ax4.text(0.1, 0.5, summary, fontsize=10, family='monospace',
                    verticalalignment='center', color=get_color('text_primary'))
            ax4.set_facecolor(get_color('bg_light'))

            self.figure.tight_layout()
            self.canvas.draw()

            print(f"[BACKTEST_PANEL] [OK] Results plotted")

        except Exception as e:
            print(f"[BACKTEST_PANEL] [X] Failed to plot results: {e}")
            import traceback
            traceback.print_exc()

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
