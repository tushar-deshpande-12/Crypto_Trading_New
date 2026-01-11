"""
Cryptocurrency Tracker Main GUI
Professional tkinter-based interface for monitoring Binance markets
Enhanced with candlestick chart viewer
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from datetime import datetime
import logging

from .chart_window import CandlestickChartWindow

logger = logging.getLogger(__name__)


class CryptoTrackerGUI:
    """
    Main GUI application for cryptocurrency market monitoring
    Designed with trader UX in mind - fast, clear, actionable
    """

    def __init__(self, root, api_client):
        self.root = root
        self.root.title("Crypto Market Tracker | Binance")
        self.root.geometry("1400x800")

        # API Client
        self.api_client = api_client
        self.market_data = []
        self.filtered_data = []
        self.sort_column = "quote_volume_24h"
        self.sort_reverse = True

        # Map tree item IDs to crypto data for chart viewing
        self.item_to_data = {}

        # UI Setup
        self._setup_styles()
        self._create_widgets()
        self._load_data()

    def _setup_styles(self):
        """Configure professional styling"""
        style = ttk.Style()
        style.theme_use('clam')

        # Treeview styling
        style.configure("Treeview",
                       background="#1e1e1e",
                       foreground="#e0e0e0",
                       fieldbackground="#1e1e1e",
                       borderwidth=0,
                       font=('Consolas', 9))

        style.configure("Treeview.Heading",
                       background="#2d2d2d",
                       foreground="#ffffff",
                       borderwidth=1,
                       font=('Consolas', 9, 'bold'))

        style.map('Treeview.Heading',
                 background=[('active', '#3d3d3d')])

        style.map('Treeview',
                 background=[('selected', '#0d47a1')],
                 foreground=[('selected', '#ffffff')])

    def _create_widgets(self):
        """Build the UI components"""

        # Top control panel
        control_frame = tk.Frame(self.root, bg="#2d2d2d")
        control_frame.pack(fill=tk.X, padx=0, pady=0)

        # Search box
        tk.Label(control_frame, text="Filter:", bg="#2d2d2d", fg="#e0e0e0",
                font=('Arial', 10)).pack(side=tk.LEFT, padx=(20, 8), pady=15)

        self.search_var = tk.StringVar()
        self.search_var.trace('w', lambda *args: self._apply_filter())

        search_entry = tk.Entry(control_frame, textvariable=self.search_var,
                               width=22, font=('Arial', 10),
                               relief='flat', borderwidth=1,
                               highlightthickness=1, highlightcolor="#0d47a1",
                               highlightbackground="#3d3d3d")
        search_entry.pack(side=tk.LEFT, padx=8, pady=15)

        # Refresh button
        refresh_btn = tk.Button(control_frame, text="↻ Refresh",
                               command=self._load_data,
                               bg="#0d47a1", fg="white",
                               font=('Arial', 10, 'bold'),
                               padx=20, pady=8,
                               relief=tk.FLAT,
                               borderwidth=0,
                               cursor="hand2")
        refresh_btn.pack(side=tk.LEFT, padx=15, pady=15)

        # Chart button
        chart_btn = tk.Button(control_frame, text="📊 View Chart",
                             command=self._open_chart,
                             bg="#2e7d32", fg="white",
                             font=('Arial', 10, 'bold'),
                             padx=20, pady=8,
                             relief=tk.FLAT,
                             borderwidth=0,
                             cursor="hand2")
        chart_btn.pack(side=tk.LEFT, padx=10, pady=15)

        # Market summary
        self.summary_label = tk.Label(control_frame, text="",
                                     bg="#2d2d2d", fg="#4caf50",
                                     font=('Arial', 10, 'bold'))
        self.summary_label.pack(side=tk.LEFT, padx=20, pady=15)

        # Timestamp
        self.timestamp_label = tk.Label(control_frame, text="",
                                       bg="#2d2d2d", fg="#9e9e9e",
                                       font=('Arial', 9))
        self.timestamp_label.pack(side=tk.RIGHT, padx=20, pady=15)

        # Main treeview frame
        tree_frame = tk.Frame(self.root, bg="#1e1e1e")
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Scrollbars
        y_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        x_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Treeview (table)
        columns = ('symbol', 'price', 'change_pct', 'change', 'volume_24h',
                  'quote_volume', 'high_24h', 'low_24h', 'trades')

        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings',
                                yscrollcommand=y_scrollbar.set,
                                xscrollcommand=x_scrollbar.set)

        y_scrollbar.config(command=self.tree.yview)
        x_scrollbar.config(command=self.tree.xview)

        # Column headers with sorting
        column_config = {
            'symbol': ('Symbol', 120, 'base_asset'),
            'price': ('Price (USDT)', 130, 'price'),
            'change_pct': ('24h Change %', 120, 'price_change_pct'),
            'change': ('24h Change', 120, 'price_change'),
            'volume_24h': ('Volume 24h', 150, 'volume_24h'),
            'quote_volume': ('Quote Vol (USDT)', 160, 'quote_volume_24h'),
            'high_24h': ('24h High', 130, 'high_24h'),
            'low_24h': ('24h Low', 130, 'low_24h'),
            'trades': ('Trades', 100, 'trades_count')
        }

        for col, (heading, width, data_key) in column_config.items():
            self.tree.heading(col, text=heading,
                            command=lambda c=data_key: self._sort_by_column(c))
            self.tree.column(col, width=width, anchor='center')

        self.tree.pack(fill=tk.BOTH, expand=True)

        # Bind double-click to open chart
        self.tree.bind('<Double-Button-1>', lambda e: self._open_chart())

        # Status bar
        status_frame = tk.Frame(self.root, bg="#2d2d2d")
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = tk.Label(status_frame, text="Ready • Double-click a row to view chart",
                                    bg="#2d2d2d", fg="#9e9e9e",
                                    font=('Arial', 9), anchor='w')
        self.status_label.pack(side=tk.LEFT, padx=20, pady=12)

    def _load_data(self):
        """Fetch data from Binance API in background thread"""
        self.status_label.config(text="Loading market data...", fg="#ffa726")

        def fetch():
            try:
                data = self.api_client.get_usdt_pairs_detailed()
                # Update UI from main thread
                self.root.after(0, lambda: self._update_display(data))
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                self.root.after(0, lambda: self._show_error(str(e)))

        thread = threading.Thread(target=fetch, daemon=True)
        thread.start()

    def _update_display(self, data):
        """Update the treeview with fresh market data"""
        if not data:
            self.status_label.config(text="Failed to load data", fg="#f44336")
            messagebox.showerror("Error", "Could not fetch market data from Binance")
            return

        self.market_data = data
        self._apply_filter()

        # Update timestamp
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.timestamp_label.config(text=f"Updated: {now}")

        # Update summary
        total_pairs = len(data)
        gainers = sum(1 for d in data if d['price_change_pct'] > 0)
        losers = sum(1 for d in data if d['price_change_pct'] < 0)

        self.summary_label.config(
            text=f"Pairs: {total_pairs} | ↑ {gainers} | ↓ {losers}"
        )

        self.status_label.config(
            text=f"Loaded {total_pairs} trading pairs | Double-click a row to view chart",
            fg="#4caf50"
        )

    def _apply_filter(self):
        """Apply search filter and refresh display"""
        search_term = self.search_var.get().upper()

        if search_term:
            self.filtered_data = [
                d for d in self.market_data
                if search_term in d['symbol'].upper() or search_term in d['base_asset'].upper()
            ]
        else:
            self.filtered_data = self.market_data.copy()

        self._refresh_tree()

    def _sort_by_column(self, column):
        """Sort data by clicked column"""
        # Toggle sort direction if same column
        if self.sort_column == column:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = column
            self.sort_reverse = True

        self._refresh_tree()

    def _refresh_tree(self):
        """Refresh treeview with sorted/filtered data"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Clear the mapping
        self.item_to_data = {}

        # Sort data
        sorted_data = sorted(self.filtered_data,
                           key=lambda x: x[self.sort_column],
                           reverse=self.sort_reverse)

        # Insert data
        for data in sorted_data:
            # Format values for display
            price_change_pct = data['price_change_pct']
            color_tag = 'gain' if price_change_pct > 0 else 'loss'

            values = (
                data['base_asset'],
                f"${data['price']:,.8f}".rstrip('0').rstrip('.'),
                f"{price_change_pct:+.2f}%",
                f"{data['price_change']:+,.8f}".rstrip('0').rstrip('.'),
                f"{data['volume_24h']:,.2f}",
                f"${data['quote_volume_24h']:,.0f}",
                f"${data['high_24h']:,.8f}".rstrip('0').rstrip('.'),
                f"${data['low_24h']:,.8f}".rstrip('0').rstrip('.'),
                f"{data['trades_count']:,}"
            )

            # Insert item into tree
            item_id = self.tree.insert('', tk.END, values=values, tags=(color_tag,))

            # Store the mapping from item_id to crypto data for chart access
            self.item_to_data[item_id] = data

        # Color coding for gains/losses
        self.tree.tag_configure('gain', foreground='#4caf50')
        self.tree.tag_configure('loss', foreground='#f44336')

    def _open_chart(self):
        """Open candlestick chart for selected cryptocurrency"""
        selected_items = self.tree.selection()

        if not selected_items:
            messagebox.showinfo(
                "No Selection",
                "Please select a cryptocurrency to view its chart.\n\n"
                "Tip: Double-click on any row to open the chart."
            )
            return

        # Get selected item
        item_id = selected_items[0]

        # Get the crypto data from our mapping
        crypto_data = self.item_to_data.get(item_id)

        if not crypto_data:
            messagebox.showerror("Error", "Could not find data for selected cryptocurrency")
            return

        # Open chart window
        try:
            CandlestickChartWindow(
                self.root,
                self.api_client,
                crypto_data['symbol'],
                crypto_data['base_asset']
            )
        except Exception as e:
            logger.error(f"Error opening chart: {e}")
            messagebox.showerror("Error", f"Failed to open chart: {str(e)}")

    def _show_error(self, error_msg):
        """Display error message"""
        self.status_label.config(text=f"Error: {error_msg}", fg="#f44336")

    def run(self):
        """Start the application"""
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        """Clean up before closing"""
        self.api_client.close()
        self.root.destroy()
