"""
Data Fetching GUI Window
Professional interface for fetching and managing cryptocurrency datasets
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import logging
from datetime import datetime
from pathlib import Path

from ..data.manager import DataManager
from ..data.config import DataPipelineConfig, ConfigPresets

logger = logging.getLogger(__name__)


class DataFetchWindow:
    """
    Professional data fetching interface
    Allows users to fetch historical OHLCV data with real-time progress tracking
    """

    def __init__(self, parent=None):
        """
        Initialize the data fetch window

        Args:
            parent: Parent window (optional)
        """
        self.window = tk.Toplevel(parent) if parent else tk.Tk()
        self.window.title("AI Prediction Data Pipeline | Fetch Historical Data")
        self.window.geometry("900x700")
        self.window.configure(bg="#1e1e1e")

        # Data manager
        self.config = DataPipelineConfig()
        self.data_manager = DataManager(
            storage_dir=self.config.storage_base_dir,
            interval=self.config.default_interval
        )

        # State
        self.is_fetching = False
        self.available_symbols = []

        # UI Setup
        self._setup_styles()
        self._create_widgets()
        self._load_available_symbols()

    def _setup_styles(self):
        """Configure professional styling matching main app"""
        style = ttk.Style()
        style.theme_use('clam')

        # Button styling
        style.configure("Fetch.TButton",
                       background="#0d47a1",
                       foreground="white",
                       font=('Arial', 10, 'bold'),
                       padding=10)

        style.configure("Stop.TButton",
                       background="#c62828",
                       foreground="white",
                       font=('Arial', 10, 'bold'),
                       padding=10)

        # Frame styling
        style.configure("Dark.TFrame",
                       background="#1e1e1e")

        style.configure("DarkLabel.TLabel",
                       background="#2d2d2d",
                       foreground="#e0e0e0",
                       font=('Arial', 10))

    def _create_widgets(self):
        """Build the UI components"""

        # Header
        header_frame = tk.Frame(self.window, bg="#0d47a1", height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="📈 Cryptocurrency Data Pipeline",
            bg="#0d47a1",
            fg="white",
            font=('Arial', 20, 'bold')
        )
        title_label.pack(pady=10)

        subtitle_label = tk.Label(
            header_frame,
            text="Fetch maximum historical 1-hour OHLCV data for AI prediction models",
            bg="#0d47a1",
            fg="#e0e0e0",
            font=('Arial', 10)
        )
        subtitle_label.pack()

        # Main content area
        content_frame = tk.Frame(self.window, bg="#1e1e1e")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # === Symbol Selection Section ===
        symbol_frame = tk.LabelFrame(
            content_frame,
            text=" Symbol Selection ",
            bg="#2d2d2d",
            fg="#e0e0e0",
            font=('Arial', 11, 'bold'),
            padx=15,
            pady=15
        )
        symbol_frame.pack(fill=tk.X, pady=(0, 15))

        # Symbol input row
        input_row = tk.Frame(symbol_frame, bg="#2d2d2d")
        input_row.pack(fill=tk.X)

        tk.Label(
            input_row,
            text="Symbol:",
            bg="#2d2d2d",
            fg="#e0e0e0",
            font=('Arial', 10)
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.symbol_var = tk.StringVar(value="BTC")
        symbol_entry = tk.Entry(
            input_row,
            textvariable=self.symbol_var,
            width=15,
            font=('Arial', 12, 'bold'),
            bg="#1e1e1e",
            fg="#4fc3f7",
            insertbackground="#4fc3f7"
        )
        symbol_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(
            input_row,
            text="USDT",
            bg="#2d2d2d",
            fg="#e0e0e0",
            font=('Arial', 10)
        ).pack(side=tk.LEFT, padx=5)

        # Popular symbols quick select
        tk.Label(
            input_row,
            text="Quick:",
            bg="#2d2d2d",
            fg="#e0e0e0",
            font=('Arial', 9)
        ).pack(side=tk.LEFT, padx=(30, 5))

        for symbol in ["BTC", "ETH", "BNB", "SOL", "XRP"]:
            btn = tk.Button(
                input_row,
                text=symbol,
                command=lambda s=symbol: self.symbol_var.set(s),
                bg="#0d47a1",
                fg="white",
                font=('Arial', 8, 'bold'),
                padx=8,
                pady=2,
                relief=tk.FLAT,
                cursor="hand2"
            )
            btn.pack(side=tk.LEFT, padx=2)

        # === Configuration Section ===
        config_frame = tk.LabelFrame(
            content_frame,
            text=" Fetch Configuration ",
            bg="#2d2d2d",
            fg="#e0e0e0",
            font=('Arial', 11, 'bold'),
            padx=15,
            pady=15
        )
        config_frame.pack(fill=tk.X, pady=(0, 15))

        # Max candles selection
        candles_row = tk.Frame(config_frame, bg="#2d2d2d")
        candles_row.pack(fill=tk.X, pady=5)

        tk.Label(
            candles_row,
            text="Maximum Candles:",
            bg="#2d2d2d",
            fg="#e0e0e0",
            font=('Arial', 10)
        ).pack(side=tk.LEFT)

        self.candles_var = tk.StringVar(value="10000")
        candles_combo = ttk.Combobox(
            candles_row,
            textvariable=self.candles_var,
            values=["1000 (~41 days)", "5000 (~208 days)", "10000 (~13.7 months)",
                   "20000 (~2.3 years)", "50000 (~5.7 years)"],
            width=25,
            state="readonly"
        )
        candles_combo.pack(side=tk.LEFT, padx=10)
        candles_combo.current(2)  # Default to 10000

        # Info label
        self.info_label = tk.Label(
            candles_row,
            text="Estimated: ~416 days of 1h data",
            bg="#2d2d2d",
            fg="#4fc3f7",
            font=('Arial', 9, 'italic')
        )
        self.info_label.pack(side=tk.LEFT, padx=10)

        # === Action Buttons ===
        button_frame = tk.Frame(content_frame, bg="#1e1e1e")
        button_frame.pack(fill=tk.X, pady=(0, 15))

        self.fetch_btn = tk.Button(
            button_frame,
            text="🚀 Fetch Data",
            command=self._start_fetch,
            bg="#0d47a1",
            fg="white",
            font=('Arial', 12, 'bold'),
            padx=30,
            pady=10,
            relief=tk.FLAT,
            cursor="hand2"
        )
        self.fetch_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(
            button_frame,
            text="⏹ Stop",
            command=self._stop_fetch,
            bg="#c62828",
            fg="white",
            font=('Arial', 12, 'bold'),
            padx=30,
            pady=10,
            relief=tk.FLAT,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        view_btn = tk.Button(
            button_frame,
            text="📁 View Datasets",
            command=self._view_datasets,
            bg="#2d2d2d",
            fg="white",
            font=('Arial', 12, 'bold'),
            padx=30,
            pady=10,
            relief=tk.FLAT,
            cursor="hand2"
        )
        view_btn.pack(side=tk.LEFT, padx=5)

        # === Progress Section ===
        progress_frame = tk.LabelFrame(
            content_frame,
            text=" Progress ",
            bg="#2d2d2d",
            fg="#e0e0e0",
            font=('Arial', 11, 'bold'),
            padx=15,
            pady=15
        )
        progress_frame.pack(fill=tk.BOTH, expand=True)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate',
            length=400
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))

        # Status label
        self.status_label = tk.Label(
            progress_frame,
            text="Ready to fetch data",
            bg="#2d2d2d",
            fg="#4fc3f7",
            font=('Arial', 10, 'bold')
        )
        self.status_label.pack(pady=5)

        # Log output
        self.log_text = scrolledtext.ScrolledText(
            progress_frame,
            height=15,
            bg="#1e1e1e",
            fg="#e0e0e0",
            font=('Consolas', 9),
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # === Storage Statistics ===
        stats_frame = tk.Frame(content_frame, bg="#1e1e1e")
        stats_frame.pack(fill=tk.X, pady=(10, 0))

        self.stats_label = tk.Label(
            stats_frame,
            text="Storage: Loading...",
            bg="#1e1e1e",
            fg="#9e9e9e",
            font=('Arial', 9)
        )
        self.stats_label.pack()

        # Update storage stats
        self._update_storage_stats()

    def _load_available_symbols(self):
        """Load available symbols in background"""
        def load():
            try:
                self._log("Loading available symbols from Binance...")
                symbols = self.data_manager.get_available_symbols()
                self.available_symbols = symbols
                self._log(f"Loaded {len(symbols)} available USDT pairs")
            except Exception as e:
                logger.error(f"Failed to load symbols: {e}")

        thread = threading.Thread(target=load, daemon=True)
        thread.start()

    def _start_fetch(self):
        """Start fetching data in background thread"""
        if self.is_fetching:
            messagebox.showwarning("Already Fetching", "Data fetch is already in progress")
            return

        symbol = self.symbol_var.get().strip().upper()
        if not symbol:
            messagebox.showerror("Invalid Symbol", "Please enter a symbol")
            return

        # Parse max candles from selection
        candles_str = self.candles_var.get()
        max_candles = int(candles_str.split()[0])

        # Confirm large fetch
        if max_candles > 20000:
            confirm = messagebox.askyesno(
                "Large Dataset",
                f"You are about to fetch {max_candles:,} candles for {symbol}USDT.\n"
                f"This may take several minutes.\n\nContinue?"
            )
            if not confirm:
                return

        self.is_fetching = True
        self.fetch_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        self._log("=" * 60)
        self._log(f"Starting data fetch for {symbol}USDT")
        self._log(f"Target: {max_candles:,} candles (~{max_candles/24:.0f} days)")
        self._log("=" * 60)

        def fetch_thread():
            try:
                def progress_callback(current, total, message):
                    if not self.is_fetching:
                        return

                    progress_pct = (current / total * 100) if total > 0 else 0
                    self.window.after(0, lambda: self.progress_var.set(progress_pct))
                    self.window.after(0, lambda: self.status_label.config(text=message))
                    self.window.after(0, lambda: self._log(message))

                dataset_path = self.data_manager.fetch_and_save(
                    symbol=symbol,
                    max_candles=max_candles,
                    progress_callback=progress_callback
                )

                if dataset_path and self.is_fetching:
                    self.window.after(0, lambda: self._log("=" * 60))
                    self.window.after(0, lambda: self._log(f"✓ SUCCESS! Data saved to:"))
                    self.window.after(0, lambda: self._log(f"  {dataset_path}"))
                    self.window.after(0, lambda: self._log("=" * 60))
                    self.window.after(0, lambda: messagebox.showinfo(
                        "Success",
                        f"Data fetched successfully!\n\n"
                        f"Saved to: {dataset_path.name}"
                    ))
                    self.window.after(0, self._update_storage_stats)
                else:
                    self.window.after(0, lambda: self._log("✗ Fetch failed or was cancelled"))

            except Exception as e:
                logger.error(f"Fetch error: {e}", exc_info=True)
                self.window.after(0, lambda: self._log(f"✗ ERROR: {str(e)}"))
                self.window.after(0, lambda: messagebox.showerror(
                    "Fetch Error",
                    f"Failed to fetch data:\n{str(e)}"
                ))

            finally:
                self.is_fetching = False
                self.window.after(0, lambda: self.fetch_btn.config(state=tk.NORMAL))
                self.window.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
                self.window.after(0, lambda: self.status_label.config(text="Ready"))
                self.window.after(0, lambda: self.progress_var.set(0))

        thread = threading.Thread(target=fetch_thread, daemon=True)
        thread.start()

    def _stop_fetch(self):
        """Stop the current fetch operation"""
        if self.is_fetching:
            self.is_fetching = False
            self._log("⏹ Stopping fetch operation...")
            self.status_label.config(text="Stopping...")

    def _view_datasets(self):
        """Open dataset browser window"""
        datasets = self.data_manager.list_all_datasets()

        if not datasets:
            messagebox.showinfo("No Datasets", "No datasets found in storage")
            return

        # Create browser window
        browser = tk.Toplevel(self.window)
        browser.title("Dataset Browser")
        browser.geometry("800x600")
        browser.configure(bg="#1e1e1e")

        # Header
        header = tk.Label(
            browser,
            text="📁 Stored Datasets",
            bg="#0d47a1",
            fg="white",
            font=('Arial', 16, 'bold'),
            pady=15
        )
        header.pack(fill=tk.X)

        # Dataset list
        list_frame = tk.Frame(browser, bg="#1e1e1e")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        listbox = tk.Listbox(
            list_frame,
            bg="#1e1e1e",
            fg="#e0e0e0",
            font=('Consolas', 9),
            yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        # Populate list
        for ds in datasets:
            symbol = ds.get('symbol', 'Unknown')
            candles = ds.get('candle_count', 0)
            timestamp = ds.get('fetch_timestamp', 'Unknown')
            listbox.insert(tk.END, f"{symbol:12} | {candles:6,} candles | {timestamp}")

        self._log(f"Opened dataset browser ({len(datasets)} datasets)")

    def _update_storage_stats(self):
        """Update storage statistics display"""
        try:
            stats = self.data_manager.get_storage_statistics()
            stats_text = (
                f"Storage: {stats['total_datasets']} datasets | "
                f"{stats['total_symbols']} symbols | "
                f"{stats['total_size_mb']} MB"
            )
            self.stats_label.config(text=stats_text)
        except Exception as e:
            logger.error(f"Failed to update stats: {e}")

    def _log(self, message):
        """Add message to log output"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def run(self):
        """Run the window main loop"""
        self.window.mainloop()

    def close(self):
        """Clean up resources"""
        if self.data_manager:
            self.data_manager.close()
        self.window.destroy()


def main():
    """Standalone entry point for data fetch window"""
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    app = DataFetchWindow()
    app.run()


if __name__ == "__main__":
    main()
