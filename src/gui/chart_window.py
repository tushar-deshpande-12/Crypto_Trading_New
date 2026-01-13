"""
Candlestick Chart Window
Professional OHLCV visualization for cryptocurrency trading analysis
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from datetime import datetime
import threading
import logging
import pandas as pd
import mplfinance as mpf

logger = logging.getLogger(__name__)


class CandlestickChartWindow:
    """
    Professional candlestick chart viewer
    Displays OHLCV data with volume bars and interactive controls
    """

    # Time interval options
    INTERVALS = {
        "1 Minute": "1m",
        "5 Minutes": "5m",
        "15 Minutes": "15m",
        "30 Minutes": "30m",
        "1 Hour": "1h",
        "4 Hours": "4h",
        "1 Day": "1d",
        "1 Week": "1w"
    }

    # Candle count options
    CANDLE_LIMITS = {
        "50 Candles": 50,
        "100 Candles": 100,
        "200 Candles": 200,
        "500 Candles": 500
    }

    def __init__(self, parent, api_client, symbol: str, base_asset: str):
        """
        Initialize chart window

        Args:
            parent: Parent window
            api_client: BinanceAPIClient instance
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            base_asset: Base asset name (e.g., "BTC")
        """
        self.parent = parent
        self.api_client = api_client
        self.symbol = symbol
        self.base_asset = base_asset

        # Create window
        self.window = tk.Toplevel(parent)
        self.window.title(f"{base_asset}/USDT Chart | Binance")
        self.window.geometry("1200x800")
        self.window.configure(bg="#1e1e1e")

        # Chart data
        self.klines_data = None
        self.current_interval = "1h"
        self.current_limit = 200

        # Build UI
        self._create_widgets()

        # Load initial data
        self._load_chart_data()

    def _create_widgets(self):
        """Build the chart window UI"""

        # Top control panel
        control_frame = tk.Frame(self.window, bg="#2d2d2d", pady=10)
        control_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        # Symbol label
        tk.Label(
            control_frame,
            text=f"{self.base_asset}/USDT",
            bg="#2d2d2d",
            fg="#4caf50",
            font=('Arial', 16, 'bold')
        ).pack(side=tk.LEFT, padx=20)

        # Interval selector
        tk.Label(
            control_frame,
            text="Timeframe:",
            bg="#2d2d2d",
            fg="#e0e0e0",
            font=('Arial', 10)
        ).pack(side=tk.LEFT, padx=(20, 5))

        self.interval_var = tk.StringVar(value="1 Hour")
        interval_combo = ttk.Combobox(
            control_frame,
            textvariable=self.interval_var,
            values=list(self.INTERVALS.keys()),
            state='readonly',
            width=12,
            font=('Arial', 9)
        )
        interval_combo.pack(side=tk.LEFT, padx=5)
        interval_combo.bind('<<ComboboxSelected>>', self._on_interval_change)

        # Candle count selector
        tk.Label(
            control_frame,
            text="Candles:",
            bg="#2d2d2d",
            fg="#e0e0e0",
            font=('Arial', 10)
        ).pack(side=tk.LEFT, padx=(20, 5))

        self.limit_var = tk.StringVar(value="200 Candles")
        limit_combo = ttk.Combobox(
            control_frame,
            textvariable=self.limit_var,
            values=list(self.CANDLE_LIMITS.keys()),
            state='readonly',
            width=12,
            font=('Arial', 9)
        )
        limit_combo.pack(side=tk.LEFT, padx=5)
        limit_combo.bind('<<ComboboxSelected>>', self._on_limit_change)

        # Refresh button
        refresh_btn = tk.Button(
            control_frame,
            text="[R] Refresh",
            command=self._load_chart_data,
            bg="#0d47a1",
            fg="white",
            font=('Arial', 10, 'bold'),
            padx=15,
            pady=5,
            relief=tk.FLAT,
            cursor="hand2"
        )
        refresh_btn.pack(side=tk.LEFT, padx=20)

        # Status label
        self.status_label = tk.Label(
            control_frame,
            text="Loading...",
            bg="#2d2d2d",
            fg="#9e9e9e",
            font=('Arial', 9)
        )
        self.status_label.pack(side=tk.RIGHT, padx=20)

        # Chart container
        self.chart_frame = tk.Frame(self.window, bg="#1e1e1e")
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _load_chart_data(self):
        """Fetch candlestick data from API"""
        self.status_label.config(text="Loading chart data...", fg="#ffa726")

        def fetch():
            try:
                klines = self.api_client.get_klines_formatted(
                    self.symbol,
                    self.current_interval,
                    self.current_limit
                )
                self.window.after(0, lambda: self._update_chart(klines))
            except Exception as e:
                logger.error(f"Error fetching chart data: {e}")
                self.window.after(0, lambda: self._show_error(str(e)))

        thread = threading.Thread(target=fetch, daemon=True)
        thread.start()

    def _update_chart(self, klines_data):
        """Update chart with new data"""
        if not klines_data:
            self.status_label.config(text="Failed to load chart data", fg="#f44336")
            messagebox.showerror("Error", f"Could not fetch chart data for {self.symbol}")
            return

        self.klines_data = klines_data
        self._draw_chart()
        self.status_label.config(
            text=f"Loaded {len(klines_data)} candles | Last update: {datetime.now().strftime('%H:%M:%S')}",
            fg="#4caf50"
        )

    def _draw_chart(self):
        """Draw the candlestick chart with volume"""
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        if not self.klines_data:
            return

        # Convert to DataFrame
        df = pd.DataFrame(self.klines_data)
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]

        # Calculate price stats
        current_price = df['close'].iloc[-1]
        price_change = df['close'].iloc[-1] - df['open'].iloc[0]
        price_change_pct = (price_change / df['open'].iloc[0]) * 100

        # Style configuration
        mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit',
                                    wick={'up':'#26a69a', 'down':'#ef5350'},
                                    volume={'up':'#26a69a', 'down':'#ef5350'})

        s = mpf.make_mpf_style(marketcolors=mc, gridcolor='#424242',
                               gridstyle='--', gridaxis='both',
                               facecolor='#1e1e1e', figcolor='#1e1e1e',
                               edgecolor='#424242', rc={'axes.labelcolor': '#e0e0e0',
                                                        'xtick.color': '#e0e0e0',
                                                        'ytick.color': '#e0e0e0'})

        # Plot using mplfinance
        title_color = '#26a69a' if price_change >= 0 else '#ef5350'
        fig, axes = mpf.plot(df, type='candle', style=s, volume=True,
                             title=f'{self.base_asset}/USDT  |  {current_price:,.8f}  |  {price_change:+,.8f} ({price_change_pct:+.2f}%)',
                             ylabel='Price (USDT)', ylabel_lower='Volume',
                             figsize=(12, 8), returnfig=True, tight_layout=True)

        # Apply title color
        axes[0].set_title(axes[0].get_title(), color=title_color, fontsize=12, fontweight='bold')

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add interactive toolbar
        toolbar = NavigationToolbar2Tk(canvas, self.chart_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def _on_interval_change(self, event=None):
        """Handle interval selection change"""
        interval_name = self.interval_var.get()
        self.current_interval = self.INTERVALS[interval_name]
        self._load_chart_data()

    def _on_limit_change(self, event=None):
        """Handle candle count change"""
        limit_name = self.limit_var.get()
        self.current_limit = self.CANDLE_LIMITS[limit_name]
        self._load_chart_data()

    def _show_error(self, error_msg):
        """Display error message"""
        self.status_label.config(text=f"Error: {error_msg}", fg="#f44336")
