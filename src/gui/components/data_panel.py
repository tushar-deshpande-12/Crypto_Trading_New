"""
Data Panel Component
Controls for downloading and managing cryptocurrency datasets
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import logging
from typing import Optional, Dict, Callable
from datetime import datetime

from src.core.config import AppConfig
from src.gui.styles import get_color

logger = logging.getLogger(__name__)


class DataPanel:
    """
    Data fetching and management panel
    """

    def __init__(
        self,
        parent,
        on_fetch_click: Optional[Callable] = None
    ):
        """
        Initialize data panel

        Args:
            parent: Parent widget
            on_fetch_click: Callback(symbol, max_candles) when fetch clicked
        """
        self.parent = parent
        self.on_fetch_click = on_fetch_click
        self.is_fetching = False
        self.current_symbol = None

        self._create_widgets()

    def _create_widgets(self):
        """Build the panel UI"""
        # Container frame
        self.frame = tk.Frame(self.parent, bg=get_color('bg_dark'))

        # Header
        header_frame = tk.Frame(self.frame, bg=get_color('primary'))
        header_frame.pack(fill=tk.X, padx=0, pady=0)

        tk.Label(
            header_frame,
            text="📥 Data Pipeline",
            bg=get_color('primary'),
            fg="white",
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_HEADER, 'bold')
        ).pack(pady=(20, 5), padx=20)

        tk.Label(
            header_frame,
            text="Fetch historical 1h OHLCV data for AI prediction",
            bg=get_color('primary'),
            fg=get_color('text_primary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_SMALL)
        ).pack(pady=(0, 20), padx=20)

        # Content area
        content = tk.Frame(self.frame, bg=get_color('bg_dark'))
        content.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Symbol display
        symbol_frame = tk.LabelFrame(
            content,
            text=" Target Symbol ",
            bg=get_color('bg_medium'),
            fg=get_color('text_primary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold'),
            padx=20,
            pady=15,
            borderwidth=0,
            relief='flat'
        )
        symbol_frame.pack(fill=tk.X, pady=(0, 15))

        self.symbol_label = tk.Label(
            symbol_frame,
            text="No symbol selected",
            bg=get_color('bg_medium'),
            fg=get_color('accent'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_LARGE, 'bold')
        )
        self.symbol_label.pack(pady=5)

        # Configuration
        config_frame = tk.LabelFrame(
            content,
            text=" Fetch Configuration ",
            bg=get_color('bg_medium'),
            fg=get_color('text_primary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold'),
            padx=20,
            pady=15,
            borderwidth=0,
            relief='flat'
        )
        config_frame.pack(fill=tk.X, pady=(0, 15))

        # Candles selection
        tk.Label(
            config_frame,
            text="Maximum Candles:",
            bg=get_color('bg_medium'),
            fg=get_color('text_primary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL)
        ).pack(anchor=tk.W, pady=(5, 2))

        self.candles_var = tk.StringVar(value="10000")
        candles_combo = ttk.Combobox(
            config_frame,
            textvariable=self.candles_var,
            values=[
                "1000 (~41 days)",
                "5000 (~208 days)",
                "10000 (~13.7 months)",
                "20000 (~2.3 years)",
                "50000 (~5.7 years)"
            ],
            state="readonly",
            width=25
        )
        candles_combo.pack(fill=tk.X, pady=(0, 10))
        candles_combo.current(2)

        # Fetch button
        self.fetch_btn = tk.Button(
            config_frame,
            text="🚀 Start Fetching Data",
            command=self._on_fetch_click,
            bg=get_color('success'),
            fg="white",
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold'),
            padx=20,
            pady=12,
            relief=tk.FLAT,
            borderwidth=0,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.fetch_btn.pack(fill=tk.X, pady=(10, 5))

        # Progress section
        progress_frame = tk.LabelFrame(
            content,
            text=" Progress ",
            bg=get_color('bg_medium'),
            fg=get_color('text_primary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold'),
            padx=20,
            pady=15,
            borderwidth=0,
            relief='flat'
        )
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 0))

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate',
            style="Crypto.Horizontal.TProgressbar"
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 8))

        # Status label
        self.status_label = tk.Label(
            progress_frame,
            text="Ready",
            bg=get_color('bg_medium'),
            fg=get_color('text_secondary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_SMALL)
        )
        self.status_label.pack(pady=(0, 5))

        # Log output
        self.log_text = scrolledtext.ScrolledText(
            progress_frame,
            height=12,
            bg=get_color('bg_dark'),
            fg=get_color('text_primary'),
            font=(AppConfig.FONT_MONO, AppConfig.FONT_SIZE_SMALL),
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def set_symbol(self, symbol_data: Dict):
        """
        Set the target symbol for data fetching

        Args:
            symbol_data: Symbol information dictionary
        """
        self.current_symbol = symbol_data
        symbol = symbol_data.get('symbol', 'Unknown')
        base = symbol_data.get('base_asset', symbol[:-4] if symbol.endswith('USDT') else symbol)

        self.symbol_label.config(text=f"{base} ({symbol})")
        self.fetch_btn.config(state=tk.NORMAL)

        self._log(f"Target set: {symbol}")

    def _on_fetch_click(self):
        """Handle fetch button click"""
        if not self.current_symbol:
            messagebox.showwarning("No Symbol", "Please select a symbol first")
            return

        if self.is_fetching:
            messagebox.showwarning("Already Fetching", "Data fetch is already in progress")
            return

        # Get max candles
        candles_str = self.candles_var.get()
        max_candles = int(candles_str.split()[0])

        # Confirm large fetch
        if max_candles > 20000:
            symbol = self.current_symbol.get('symbol', 'Unknown')
            if not messagebox.askyesno(
                "Large Dataset",
                f"Fetching {max_candles:,} candles for {symbol}.\n"
                f"This may take several minutes.\n\nContinue?"
            ):
                return

        # Call callback
        if self.on_fetch_click:
            self.on_fetch_click(self.current_symbol, max_candles)

    def start_fetch(self):
        """Mark fetch as started"""
        self.is_fetching = True
        self.fetch_btn.config(state=tk.DISABLED, text="⏳ Fetching...")
        self.progress_var.set(0)

    def update_progress(self, current: int, total: int, message: str):
        """Update progress display"""
        progress_pct = (current / total * 100) if total > 0 else 0
        self.progress_var.set(progress_pct)
        self.status_label.config(text=message)
        self._log(message)

    def complete_fetch(self, success: bool, message: str = ""):
        """Mark fetch as complete"""
        self.is_fetching = False
        self.fetch_btn.config(state=tk.NORMAL, text="🚀 Start Fetching Data")

        if success:
            self.status_label.config(text="✓ Complete!", fg=get_color('success'))
            self.progress_var.set(100)
            self._log(f"✓ SUCCESS: {message}")
        else:
            self.status_label.config(text="✗ Failed", fg=get_color('danger'))
            self.progress_var.set(0)
            self._log(f"✗ FAILED: {message}")

    def _log(self, message: str):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def clear_log(self):
        """Clear the log"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete('1.0', tk.END)
        self.log_text.config(state=tk.DISABLED)

    def pack(self, **kwargs):
        """Pack the component"""
        self.frame.pack(**kwargs)

    def grid(self, **kwargs):
        """Grid the component"""
        self.frame.grid(**kwargs)
