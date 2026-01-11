"""
Symbol Table Component
Displays cryptocurrency list with download and chart actions
"""

import tkinter as tk
from tkinter import ttk
import logging
from typing import List, Dict, Callable, Optional

from src.core.config import AppConfig
from src.gui.styles import get_color

logger = logging.getLogger(__name__)


class SymbolTable:
    """
    Symbol list table with integrated download and chart buttons
    """

    def __init__(
        self,
        parent,
        on_download_click: Optional[Callable] = None,
        on_chart_click: Optional[Callable] = None
    ):
        """
        Initialize symbol table

        Args:
            parent: Parent widget
            on_download_click: Callback(symbol_data) when download clicked
            on_chart_click: Callback(symbol_data) when chart clicked
        """
        self.parent = parent
        self.on_download_click = on_download_click
        self.on_chart_click = on_chart_click
        self.market_data = []
        self.filtered_data = []
        self.item_to_data = {}
        self.sort_column = "quote_volume_24h"
        self.sort_reverse = True

        self._create_widgets()

    def _create_widgets(self):
        """Build the table UI"""
        # Container frame
        self.frame = tk.Frame(self.parent, bg=get_color('bg_dark'))

        # Control bar
        control_frame = tk.Frame(self.frame, bg=get_color('bg_medium'))
        control_frame.pack(fill=tk.X, padx=0, pady=0)

        # Search box
        tk.Label(
            control_frame,
            text="Search:",
            bg=get_color('bg_medium'),
            fg=get_color('text_primary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL)
        ).pack(side=tk.LEFT, padx=(20, 8), pady=15)

        self.search_var = tk.StringVar()
        self.search_var.trace('w', lambda *args: self._apply_filter())

        search_entry = tk.Entry(
            control_frame,
            textvariable=self.search_var,
            width=18,
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL),
            bg=get_color('bg_dark'),
            fg=get_color('accent'),
            insertbackground=get_color('accent'),
            relief='flat',
            borderwidth=1,
            highlightthickness=1,
            highlightcolor=get_color('primary'),
            highlightbackground=get_color('bg_light')
        )
        search_entry.pack(side=tk.LEFT, padx=5, pady=15)

        # Refresh button
        refresh_btn = tk.Button(
            control_frame,
            text="↻ Refresh",
            command=self._on_refresh_click,
            bg=get_color('primary'),
            fg="white",
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold'),
            padx=18,
            pady=8,
            relief=tk.FLAT,
            borderwidth=0,
            cursor="hand2"
        )
        refresh_btn.pack(side=tk.LEFT, padx=15, pady=15)

        # Info label
        self.info_label = tk.Label(
            control_frame,
            text="Loading...",
            bg=get_color('bg_medium'),
            fg=get_color('text_secondary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_SMALL)
        )
        self.info_label.pack(side=tk.RIGHT, padx=20, pady=15)

        # Help label
        help_label = tk.Label(
            control_frame,
            text="Click [Download] or [Chart] • Right-click for menu • Double-click to chart",
            bg=get_color('bg_medium'),
            fg=get_color('text_secondary'),
            font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_SMALL),
            anchor=tk.W
        )
        help_label.pack(side=tk.LEFT, padx=(20, 10), pady=15)

        # Table container with scrollbar
        table_container = tk.Frame(self.frame, bg=get_color('bg_dark'))
        table_container.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Scrollbar
        scrollbar = tk.Scrollbar(table_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Treeview (Table)
        columns = ('Symbol', 'Price', 'Change%', 'Volume', 'Actions')
        self.tree = ttk.Treeview(
            table_container,
            columns=columns,
            show='headings',
            yscrollcommand=scrollbar.set,
            style="Crypto.Treeview",
            selectmode='browse'
        )

        # Configure columns
        self.tree.heading('Symbol', text='Symbol', command=lambda: self._sort_by('symbol'))
        self.tree.heading('Price', text='Price (USDT)', command=lambda: self._sort_by('price'))
        self.tree.heading('Change%', text='24h Change %', command=lambda: self._sort_by('price_change_pct'))
        self.tree.heading('Volume', text='Volume (USDT)', command=lambda: self._sort_by('quote_volume_24h'))
        self.tree.heading('Actions', text='Actions')

        self.tree.column('Symbol', width=120, anchor=tk.W)
        self.tree.column('Price', width=120, anchor=tk.E)
        self.tree.column('Change%', width=120, anchor=tk.E)
        self.tree.column('Volume', width=150, anchor=tk.E)
        self.tree.column('Actions', width=180, anchor=tk.CENTER)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.tree.yview)

        # Bind double-click to chart
        self.tree.bind('<Double-1>', lambda e: self._on_row_double_click())

        # Bind right-click for context menu
        self.tree.bind('<Button-3>', self._on_right_click)

        # Action buttons frame (overlay)
        self._create_action_overlay()

    def _create_action_overlay(self):
        """Create overlay for action buttons"""
        # We'll use tags and bind clicks instead of actual buttons in cells
        self.tree.tag_configure('positive', foreground='#4caf50')
        self.tree.tag_configure('negative', foreground='#f44336')

        # Bind single click for actions
        self.tree.bind('<Button-1>', self._on_tree_click)

    def _on_tree_click(self, event):
        """Handle tree click for action buttons"""
        try:
            region = self.tree.identify_region(event.x, event.y)
            if region != "cell":
                return

            column = self.tree.identify_column(event.x)
            row_id = self.tree.identify_row(event.y)

            if not row_id or column != '#5':  # Actions column
                return

            # Get symbol data
            symbol_data = self.item_to_data.get(row_id)
            if not symbol_data:
                return

            # Get bbox safely
            bbox = self.tree.bbox(row_id, column)
            if not bbox:
                return

            # Determine which action based on x position
            x_in_column = event.x - bbox[0]
            column_width = self.tree.column(column, 'width')

            # Download button is on left half, Chart on right half
            if x_in_column < column_width // 2:
                # Download action
                logger.info(f"Download clicked for {symbol_data.get('symbol')}")
                if self.on_download_click:
                    self.on_download_click(symbol_data)
            else:
                # Chart action
                logger.info(f"Chart clicked for {symbol_data.get('symbol')}")
                if self.on_chart_click:
                    self.on_chart_click(symbol_data)
        except Exception as e:
            logger.error(f"Error handling tree click: {e}", exc_info=True)

    def _on_row_double_click(self):
        """Handle double-click to show chart"""
        selection = self.tree.selection()
        if not selection:
            return

        symbol_data = self.item_to_data.get(selection[0])
        if symbol_data and self.on_chart_click:
            self.on_chart_click(symbol_data)

    def _on_right_click(self, event):
        """Handle right-click to show context menu"""
        try:
            # Select the item under cursor
            row_id = self.tree.identify_row(event.y)
            if not row_id:
                return

            self.tree.selection_set(row_id)
            symbol_data = self.item_to_data.get(row_id)
            if not symbol_data:
                return

            # Create context menu
            menu = tk.Menu(self.tree, tearoff=0, bg=get_color('bg_medium'), fg=get_color('text_primary'))

            symbol = symbol_data.get('symbol', 'Unknown')
            menu.add_command(
                label=f"📊 View Chart for {symbol}",
                command=lambda: self.on_chart_click(symbol_data) if self.on_chart_click else None
            )
            menu.add_command(
                label=f"📥 Download Data for {symbol}",
                command=lambda: self.on_download_click(symbol_data) if self.on_download_click else None
            )

            # Show menu
            menu.post(event.x_root, event.y_root)
        except Exception as e:
            logger.error(f"Error showing context menu: {e}", exc_info=True)

    def _on_refresh_click(self):
        """Handle refresh button click"""
        if self.on_chart_click:  # Using chart_click callback as general refresh
            self.refresh_data()

    def load_data(self, market_data: List[Dict]):
        """
        Load market data into the table

        Args:
            market_data: List of symbol dictionaries
        """
        self.market_data = market_data
        self._apply_filter()

        # Update info label
        self.info_label.config(
            text=f"Showing {len(self.filtered_data)} of {len(self.market_data)} symbols"
        )

    def _apply_filter(self):
        """Filter data based on search term"""
        search_term = self.search_var.get().upper().strip()

        if search_term:
            self.filtered_data = [
                item for item in self.market_data
                if search_term in item.get('symbol', '').upper()
                or search_term in item.get('base_asset', '').upper()
            ]
        else:
            self.filtered_data = self.market_data.copy()

        self._update_table()

    def _sort_by(self, column: str):
        """Sort table by column"""
        if self.sort_column == column:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = column
            self.sort_reverse = True

        self.filtered_data.sort(
            key=lambda x: x.get(column, 0),
            reverse=self.sort_reverse
        )

        self._update_table()

    def _update_table(self):
        """Update table display"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        self.item_to_data.clear()

        # Add filtered items
        for data in self.filtered_data:
            symbol = data.get('symbol', '')
            base = data.get('base_asset', symbol[:-4] if symbol.endswith('USDT') else symbol)
            price = data.get('price', 0)
            change_pct = data.get('price_change_pct', 0)
            volume = data.get('quote_volume_24h', 0)

            # Format values
            price_str = self._format_price(price)
            change_str = f"{change_pct:+.2f}%"
            volume_str = self._format_volume(volume)
            actions_str = "[Download] | [Chart]"

            # Determine tag for color
            tag = 'positive' if change_pct >= 0 else 'negative'

            item_id = self.tree.insert(
                '',
                'end',
                values=(base, price_str, change_str, volume_str, actions_str),
                tags=(tag,)
            )

            self.item_to_data[item_id] = data

        # Update info
        self.info_label.config(
            text=f"Showing {len(self.filtered_data)} of {len(self.market_data)} symbols"
        )

    def _format_price(self, price: float) -> str:
        """Format price with appropriate decimals"""
        if price >= 1000:
            return f"${price:,.2f}"
        elif price >= 1:
            return f"${price:.4f}"
        else:
            return f"${price:.8f}"

    def _format_volume(self, volume: float) -> str:
        """Format volume in M/B notation"""
        if volume >= 1_000_000_000:
            return f"${volume / 1_000_000_000:.2f}B"
        elif volume >= 1_000_000:
            return f"${volume / 1_000_000:.2f}M"
        elif volume >= 1_000:
            return f"${volume / 1_000:.2f}K"
        else:
            return f"${volume:.2f}"

    def get_selected_symbol(self) -> Optional[Dict]:
        """Get currently selected symbol data"""
        selection = self.tree.selection()
        if selection:
            return self.item_to_data.get(selection[0])
        return None

    def refresh_data(self):
        """Trigger data refresh (to be implemented by parent)"""
        logger.info("Refresh requested")

    def pack(self, **kwargs):
        """Pack the component"""
        self.frame.pack(**kwargs)

    def grid(self, **kwargs):
        """Grid the component"""
        self.frame.grid(**kwargs)
