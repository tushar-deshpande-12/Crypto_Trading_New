"""
Chart Panel Component
Displays candlestick charts for cryptocurrency data
"""

import tkinter as tk
from tkinter import messagebox
import logging
from typing import Optional, Dict
from datetime import datetime

from src.core.config import AppConfig
from src.gui.styles import get_color

logger = logging.getLogger(__name__)


class ChartPanel:
    """
    Chart display panel with candlestick visualization
    """

    def __init__(self, parent):
        """
        Initialize chart panel

        Args:
            parent: Parent widget
        """
        self.parent = parent
        self.current_symbol = None
        self.chart_data = []

        self._create_widgets()

    def _create_widgets(self):
        """Build the chart UI"""
        # Container frame
        self.frame = tk.Frame(self.parent, bg=get_color('bg_dark'))

        # Header
        header_frame = tk.Frame(self.frame, bg=get_color('bg_medium'))
        header_frame.pack(fill=tk.X, padx=0, pady=0)

        self.title_label = tk.Label(
            header_frame,
            text="Select a symbol to view chart",
            bg=get_color('bg_medium'),
            fg=get_color('text_primary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_LARGE, 'bold')
        )
        self.title_label.pack(pady=20, padx=20)

        # Chart canvas area
        self.canvas_frame = tk.Frame(self.frame, bg=get_color('bg_dark'))
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # Placeholder message
        self.placeholder = tk.Label(
            self.canvas_frame,
            text="📊\n\nChart will appear here\n\nDouble-click a symbol or click 'Chart' to view",
            bg=get_color('bg_dark'),
            fg=get_color('text_secondary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL),
            justify=tk.CENTER
        )
        self.placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def show_chart(self, symbol_data: Dict, chart_data: list):
        """
        Display chart for symbol

        Args:
            symbol_data: Symbol information dictionary
            chart_data: OHLCV candlestick data
        """
        try:
            symbol = symbol_data.get('symbol', 'Unknown')
            self.current_symbol = symbol
            self.chart_data = chart_data

            # Update title
            self.title_label.config(text=f"{symbol} - Candlestick Chart")

            # Hide placeholder if it exists
            try:
                if self.placeholder and self.placeholder.winfo_exists():
                    self.placeholder.place_forget()
            except:
                pass

            # Try to use matplotlib if available
            try:
                self._render_matplotlib_chart()
            except ImportError:
                self._render_simple_chart()

        except Exception as e:
            logger.error(f"Error showing chart: {e}", exc_info=True)
            messagebox.showerror("Chart Error", f"Failed to display chart: {str(e)}")

    def _render_matplotlib_chart(self):
        """Render chart using matplotlib"""
        import matplotlib
        matplotlib.use('TkAgg')
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        import matplotlib.dates as mdates

        # Clear previous chart
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        # Create figure
        fig = Figure(figsize=(10, 6), facecolor=get_color('bg_dark'))
        ax = fig.add_subplot(111)

        # Prepare data
        if not self.chart_data:
            return

        dates = [d['timestamp'] for d in self.chart_data]
        opens = [d['open'] for d in self.chart_data]
        highs = [d['high'] for d in self.chart_data]
        lows = [d['low'] for d in self.chart_data]
        closes = [d['close'] for d in self.chart_data]

        # Plot candlesticks
        for i in range(len(dates)):
            color = '#4caf50' if closes[i] >= opens[i] else '#f44336'

            # Wick (high-low line)
            ax.plot([dates[i], dates[i]], [lows[i], highs[i]],
                   color=color, linewidth=1, alpha=0.8)

            # Body (open-close rectangle)
            height = abs(closes[i] - opens[i])
            bottom = min(opens[i], closes[i])
            ax.bar(dates[i], height, bottom=bottom, width=0.0007,
                  color=color, alpha=0.8)

        # Style axes
        ax.set_facecolor(get_color('bg_dark'))
        ax.tick_params(colors=get_color('text_primary'))
        ax.spines['bottom'].set_color(get_color('text_secondary'))
        ax.spines['top'].set_color(get_color('text_secondary'))
        ax.spines['left'].set_color(get_color('text_secondary'))
        ax.spines['right'].set_color(get_color('text_secondary'))
        ax.xaxis.label.set_color(get_color('text_primary'))
        ax.yaxis.label.set_color(get_color('text_primary'))
        ax.grid(True, alpha=0.2, color=get_color('text_secondary'))

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        fig.autofmt_xdate()

        ax.set_ylabel('Price (USDT)', color=get_color('text_primary'))
        ax.set_title(f'{self.current_symbol} - Last {len(dates)} Candles',
                    color=get_color('text_primary'), pad=20)

        fig.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _render_simple_chart(self):
        """Render simple text-based chart when matplotlib unavailable"""
        # Clear previous content
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        text = tk.Text(
            self.canvas_frame,
            bg=get_color('bg_dark'),
            fg=get_color('text_primary'),
            font=(AppConfig.FONT_MONO, AppConfig.FONT_SIZE_SMALL),
            wrap=tk.NONE
        )
        text.pack(fill=tk.BOTH, expand=True)

        # Display data
        text.insert('1.0', f"Chart Data for {self.current_symbol}\n")
        text.insert('end', "=" * 80 + "\n\n")
        text.insert('end', f"{'Time':<20} {'Open':>12} {'High':>12} {'Low':>12} {'Close':>12} {'Volume':>15}\n")
        text.insert('end', "-" * 80 + "\n")

        for candle in self.chart_data[-50:]:  # Last 50 candles
            time_str = candle.get('timestamp', '').strftime('%Y-%m-%d %H:%M') if isinstance(candle.get('timestamp'), datetime) else str(candle.get('timestamp', ''))[:19]
            text.insert('end',
                f"{time_str:<20} "
                f"{candle.get('open', 0):>12.4f} "
                f"{candle.get('high', 0):>12.4f} "
                f"{candle.get('low', 0):>12.4f} "
                f"{candle.get('close', 0):>12.4f} "
                f"{candle.get('volume', 0):>15.2f}\n"
            )

        text.config(state=tk.DISABLED)

    def clear(self):
        """Clear the chart"""
        self.current_symbol = None
        self.chart_data = []
        self.title_label.config(text="Select a symbol to view chart")

        # Clear canvas
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        # Show placeholder
        self.placeholder = tk.Label(
            self.canvas_frame,
            text="📊\n\nChart will appear here\n\nDouble-click a symbol or click 'Chart' to view",
            bg=get_color('bg_dark'),
            fg=get_color('text_secondary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL),
            justify=tk.CENTER
        )
        self.placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def pack(self, **kwargs):
        """Pack the component"""
        self.frame.pack(**kwargs)

    def grid(self, **kwargs):
        """Grid the component"""
        self.frame.grid(**kwargs)
