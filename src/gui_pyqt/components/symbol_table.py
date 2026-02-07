"""
Symbol Table Component - QTableView with sorting and filtering

Displays cryptocurrency pairs with price, change, and volume.
Supports inline action buttons for chart and download.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableView, QLineEdit,
    QPushButton, QLabel, QHeaderView, QAbstractItemView, QStyledItemDelegate
)
from PyQt6.QtCore import (
    Qt, pyqtSignal, QSortFilterProxyModel, QAbstractTableModel,
    QModelIndex, QVariant
)
from PyQt6.QtGui import QColor, QBrush
from typing import List, Dict, Optional
from src.gui_pyqt.styles import COLORS


class SymbolTableModel(QAbstractTableModel):
    """
    Model for symbol table data.

    Columns: Symbol, Price, Change%, Volume, Actions
    Supports both crypto (USD) and stocks (INR) markets.
    """

    COLUMNS = ['Symbol', 'Price', 'Change %', 'Volume 24h', 'Actions']

    def __init__(self):
        super().__init__()
        self._data: List[Dict] = []
        self._market_type = "crypto"  # "crypto" or "stocks"

    def set_market_type(self, market_type: str):
        """Set the market type for currency formatting"""
        self._market_type = market_type
        self.layoutChanged.emit()

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(self.COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.COLUMNS[section]
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._data)):
            return None

        row_data = self._data[index.row()]
        col = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:  # Symbol
                return row_data.get('symbol', '')
            elif col == 1:  # Price
                price = row_data.get('price', 0)
                currency = "₹" if self._market_type == "stocks" else "$"
                if price >= 1000:
                    return f"{currency}{price:,.2f}"
                elif price >= 1:
                    return f"{currency}{price:.4f}"
                else:
                    return f"{currency}{price:.8f}"
            elif col == 2:  # Change %
                change = row_data.get('price_change_pct', 0)
                return f"{change:+.2f}%"
            elif col == 3:  # Volume
                vol = row_data.get('quote_volume_24h', 0)
                currency = "₹" if self._market_type == "stocks" else "$"
                if vol >= 1_000_000_000:
                    return f"{currency}{vol/1_000_000_000:.2f}B"
                elif vol >= 1_000_000:
                    return f"{currency}{vol/1_000_000:.2f}M"
                elif vol >= 1_000:
                    return f"{currency}{vol/1_000:.2f}K"
                else:
                    return f"{currency}{vol:.2f}"
            elif col == 4:  # Actions
                return "Chart | Download"

        elif role == Qt.ItemDataRole.ForegroundRole:
            if col == 2:  # Color for change %
                change = row_data.get('price_change_pct', 0)
                if change > 0:
                    return QBrush(QColor(COLORS['chart_green']))
                elif change < 0:
                    return QBrush(QColor(COLORS['chart_red']))

        elif role == Qt.ItemDataRole.TextAlignmentRole:
            if col in [1, 2, 3]:  # Right-align numbers
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            elif col == 4:  # Center actions
                return Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter

        elif role == Qt.ItemDataRole.UserRole:
            # Return full data dict for row
            return row_data

        return None

    def setData(self, data: List[Dict]):
        """Update table data"""
        self.beginResetModel()
        self._data = data
        self.endResetModel()

    def get_row_data(self, row: int) -> Optional[Dict]:
        """Get data for a specific row"""
        if 0 <= row < len(self._data):
            return self._data[row]
        return None


class SymbolFilterProxyModel(QSortFilterProxyModel):
    """
    Proxy model for filtering and sorting symbols.
    """

    def __init__(self):
        super().__init__()
        self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setFilterKeyColumn(0)  # Filter on Symbol column

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        """Custom sorting for numeric columns"""
        col = left.column()
        left_data = self.sourceModel().data(left, Qt.ItemDataRole.UserRole)
        right_data = self.sourceModel().data(right, Qt.ItemDataRole.UserRole)

        if left_data is None or right_data is None:
            return False

        if col == 1:  # Price
            return left_data.get('price', 0) < right_data.get('price', 0)
        elif col == 2:  # Change %
            return left_data.get('price_change_pct', 0) < right_data.get('price_change_pct', 0)
        elif col == 3:  # Volume
            return left_data.get('quote_volume_24h', 0) < right_data.get('quote_volume_24h', 0)

        return super().lessThan(left, right)


class SymbolTable(QWidget):
    """
    Symbol Table Widget

    Displays cryptocurrency pairs with filtering, sorting, and actions.

    Signals:
        chart_clicked(dict): Emitted when chart action clicked
        download_clicked(dict): Emitted when download action clicked
        row_double_clicked(dict): Emitted on double-click
    """

    chart_clicked = pyqtSignal(dict)
    download_clicked = pyqtSignal(dict)
    row_double_clicked = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Control bar
        control_layout = QHBoxLayout()

        # Search box
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search symbols...")
        self.search_input.textChanged.connect(self._on_search)
        control_layout.addWidget(self.search_input, stretch=1)

        # Refresh button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setFixedWidth(80)
        control_layout.addWidget(self.refresh_btn)

        # Info label
        self.info_label = QLabel("0 symbols")
        self.info_label.setProperty("class", "secondary")
        control_layout.addWidget(self.info_label)

        layout.addLayout(control_layout)

        # Table view
        self.table = QTableView()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setSortingEnabled(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)

        # Model setup
        self.model = SymbolTableModel()
        self.proxy_model = SymbolFilterProxyModel()
        self.proxy_model.setSourceModel(self.model)
        self.table.setModel(self.proxy_model)

        # Column sizing
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(4, 140)

        # Default sort by volume
        self.table.sortByColumn(3, Qt.SortOrder.DescendingOrder)

        # Connect signals
        self.table.doubleClicked.connect(self._on_double_click)
        self.table.clicked.connect(self._on_click)

        layout.addWidget(self.table)

    def _on_search(self, text: str):
        """Filter table by search text"""
        self.proxy_model.setFilterFixedString(text)
        self._update_info_label()

    def _on_double_click(self, index: QModelIndex):
        """Handle double-click - emit chart_clicked"""
        source_index = self.proxy_model.mapToSource(index)
        data = self.model.get_row_data(source_index.row())
        if data:
            self.row_double_clicked.emit(data)
            self.chart_clicked.emit(data)

    def _on_click(self, index: QModelIndex):
        """Handle click on actions column"""
        if index.column() == 4:  # Actions column
            source_index = self.proxy_model.mapToSource(index)
            data = self.model.get_row_data(source_index.row())
            if data:
                # Determine if Chart or Download based on click position
                # For simplicity, emit chart_clicked (download via right-click menu)
                self.chart_clicked.emit(data)

    def _update_info_label(self):
        """Update the info label with count"""
        visible = self.proxy_model.rowCount()
        total = self.model.rowCount()
        if visible == total:
            self.info_label.setText(f"{total} symbols")
        else:
            self.info_label.setText(f"{visible} / {total} symbols")

    def load_data(self, data: List[Dict]):
        """
        Load market data into the table.

        Args:
            data: List of symbol data dicts with keys:
                  symbol, price, price_change_pct, quote_volume_24h
        """
        self.model.setData(data)
        self._update_info_label()

    def get_selected_data(self) -> Optional[Dict]:
        """Get data for currently selected row"""
        indexes = self.table.selectionModel().selectedRows()
        if indexes:
            source_index = self.proxy_model.mapToSource(indexes[0])
            return self.model.get_row_data(source_index.row())
        return None

    def clear(self):
        """Clear all data"""
        self.model.setData([])
        self._update_info_label()

    def set_market_type(self, market_type: str):
        """Set market type for currency display (crypto/stocks)"""
        self.model.set_market_type(market_type)
