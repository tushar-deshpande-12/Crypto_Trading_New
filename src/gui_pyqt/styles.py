"""
PyQt6 Stylesheet - Dark Theme

Professional dark theme for the Crypto AI Predictor application.
"""

# Color Palette
COLORS = {
    'bg_dark': '#1e1e1e',
    'bg_medium': '#2d2d2d',
    'bg_light': '#3d3d3d',
    'bg_lighter': '#4d4d4d',
    'primary': '#0d47a1',
    'primary_hover': '#1565c0',
    'primary_pressed': '#0a3d91',
    'success': '#2e7d32',
    'success_hover': '#388e3c',
    'danger': '#c62828',
    'danger_hover': '#d32f2f',
    'warning': '#f57c00',
    'text_primary': '#e0e0e0',
    'text_secondary': '#9e9e9e',
    'text_disabled': '#666666',
    'accent': '#4fc3f7',
    'border': '#404040',
    'scrollbar': '#555555',
    'scrollbar_hover': '#666666',
    'chart_green': '#4caf50',
    'chart_red': '#f44336',
}

# Font Configuration
FONTS = {
    'family': 'Segoe UI, Arial, sans-serif',
    'mono': 'Consolas, Courier New, monospace',
    'size_small': 9,
    'size_normal': 10,
    'size_large': 12,
    'size_header': 14,
    'size_title': 16,
}

# Main Stylesheet
DARK_STYLESHEET = f"""
/* ============================================== */
/* GLOBAL STYLES                                  */
/* ============================================== */

QMainWindow {{
    background-color: {COLORS['bg_dark']};
}}

QWidget {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_primary']};
    font-family: {FONTS['family']};
    font-size: {FONTS['size_normal']}pt;
}}

/* ============================================== */
/* TAB WIDGET                                     */
/* ============================================== */

QTabWidget::pane {{
    border: none;
    background-color: {COLORS['bg_dark']};
    padding: 5px;
}}

QTabBar::tab {{
    background-color: {COLORS['bg_medium']};
    color: {COLORS['text_primary']};
    padding: 12px 24px;
    border: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
    font-size: {FONTS['size_normal']}pt;
}}

QTabBar::tab:selected {{
    background-color: {COLORS['bg_light']};
    font-weight: bold;
}}

QTabBar::tab:hover:!selected {{
    background-color: {COLORS['bg_lighter']};
}}

/* ============================================== */
/* BUTTONS                                        */
/* ============================================== */

QPushButton {{
    background-color: {COLORS['primary']};
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 4px;
    font-weight: bold;
    font-size: {FONTS['size_normal']}pt;
    min-width: 80px;
}}

QPushButton:hover {{
    background-color: {COLORS['primary_hover']};
}}

QPushButton:pressed {{
    background-color: {COLORS['primary_pressed']};
}}

QPushButton:disabled {{
    background-color: {COLORS['bg_light']};
    color: {COLORS['text_disabled']};
}}

QPushButton[class="success"] {{
    background-color: {COLORS['success']};
}}

QPushButton[class="success"]:hover {{
    background-color: {COLORS['success_hover']};
}}

QPushButton[class="danger"] {{
    background-color: {COLORS['danger']};
}}

QPushButton[class="danger"]:hover {{
    background-color: {COLORS['danger_hover']};
}}

/* ============================================== */
/* INPUT FIELDS                                   */
/* ============================================== */

QLineEdit {{
    background-color: {COLORS['bg_light']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 8px 12px;
    color: {COLORS['text_primary']};
    font-size: {FONTS['size_normal']}pt;
}}

QLineEdit:focus {{
    border-color: {COLORS['primary']};
}}

QLineEdit:disabled {{
    background-color: {COLORS['bg_medium']};
    color: {COLORS['text_disabled']};
}}

QSpinBox, QDoubleSpinBox {{
    background-color: {COLORS['bg_light']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 6px 10px;
    color: {COLORS['text_primary']};
}}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: {COLORS['bg_medium']};
    border: none;
    width: 20px;
}}

/* ============================================== */
/* COMBOBOX                                       */
/* ============================================== */

QComboBox {{
    background-color: {COLORS['bg_light']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 8px 12px;
    color: {COLORS['text_primary']};
    min-width: 120px;
}}

QComboBox:hover {{
    border-color: {COLORS['primary']};
}}

QComboBox::drop-down {{
    border: none;
    width: 30px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {COLORS['text_primary']};
    margin-right: 10px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS['bg_medium']};
    border: 1px solid {COLORS['border']};
    selection-background-color: {COLORS['primary']};
    selection-color: white;
}}

/* ============================================== */
/* TABLE VIEW                                     */
/* ============================================== */

QTableView {{
    background-color: {COLORS['bg_dark']};
    alternate-background-color: {COLORS['bg_medium']};
    gridline-color: {COLORS['border']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    selection-background-color: {COLORS['primary']};
    selection-color: white;
}}

QTableView::item {{
    padding: 8px;
    border-bottom: 1px solid {COLORS['border']};
}}

QTableView::item:selected {{
    background-color: {COLORS['primary']};
}}

QHeaderView::section {{
    background-color: {COLORS['bg_medium']};
    color: {COLORS['text_primary']};
    padding: 10px;
    border: none;
    border-bottom: 2px solid {COLORS['primary']};
    font-weight: bold;
}}

QHeaderView::section:hover {{
    background-color: {COLORS['bg_light']};
}}

/* ============================================== */
/* SCROLLBARS                                     */
/* ============================================== */

QScrollBar:vertical {{
    background-color: {COLORS['bg_dark']};
    width: 12px;
    border: none;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS['scrollbar']};
    border-radius: 6px;
    min-height: 30px;
    margin: 2px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {COLORS['scrollbar_hover']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar:horizontal {{
    background-color: {COLORS['bg_dark']};
    height: 12px;
    border: none;
}}

QScrollBar::handle:horizontal {{
    background-color: {COLORS['scrollbar']};
    border-radius: 6px;
    min-width: 30px;
    margin: 2px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {COLORS['scrollbar_hover']};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* ============================================== */
/* PROGRESS BAR                                   */
/* ============================================== */

QProgressBar {{
    background-color: {COLORS['bg_medium']};
    border: none;
    border-radius: 4px;
    text-align: center;
    height: 24px;
    font-weight: bold;
}}

QProgressBar::chunk {{
    background-color: {COLORS['primary']};
    border-radius: 4px;
}}

/* ============================================== */
/* LABELS                                         */
/* ============================================== */

QLabel {{
    color: {COLORS['text_primary']};
    background-color: transparent;
}}

QLabel[class="header"] {{
    font-size: {FONTS['size_header']}pt;
    font-weight: bold;
    color: {COLORS['accent']};
    padding: 10px 0px;
}}

QLabel[class="title"] {{
    font-size: {FONTS['size_title']}pt;
    font-weight: bold;
}}

QLabel[class="secondary"] {{
    color: {COLORS['text_secondary']};
}}

/* ============================================== */
/* CHECKBOX & RADIO                               */
/* ============================================== */

QCheckBox {{
    color: {COLORS['text_primary']};
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {COLORS['border']};
    border-radius: 4px;
    background-color: {COLORS['bg_light']};
}}

QCheckBox::indicator:checked {{
    background-color: {COLORS['primary']};
    border-color: {COLORS['primary']};
}}

QCheckBox::indicator:hover {{
    border-color: {COLORS['primary']};
}}

QRadioButton {{
    color: {COLORS['text_primary']};
    spacing: 8px;
}}

QRadioButton::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {COLORS['border']};
    border-radius: 9px;
    background-color: {COLORS['bg_light']};
}}

QRadioButton::indicator:checked {{
    background-color: {COLORS['primary']};
    border-color: {COLORS['primary']};
}}

/* ============================================== */
/* GROUP BOX                                      */
/* ============================================== */

QGroupBox {{
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 10px;
    font-weight: bold;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 10px;
    color: {COLORS['accent']};
}}

/* ============================================== */
/* TEXT EDIT                                      */
/* ============================================== */

QTextEdit, QPlainTextEdit {{
    background-color: {COLORS['bg_light']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 8px;
    color: {COLORS['text_primary']};
    font-family: {FONTS['mono']};
    font-size: {FONTS['size_small']}pt;
}}

/* ============================================== */
/* SLIDER                                         */
/* ============================================== */

QSlider::groove:horizontal {{
    background-color: {COLORS['bg_light']};
    height: 8px;
    border-radius: 4px;
}}

QSlider::handle:horizontal {{
    background-color: {COLORS['primary']};
    width: 18px;
    height: 18px;
    margin: -5px 0;
    border-radius: 9px;
}}

QSlider::handle:horizontal:hover {{
    background-color: {COLORS['primary_hover']};
}}

QSlider::sub-page:horizontal {{
    background-color: {COLORS['primary']};
    border-radius: 4px;
}}

/* ============================================== */
/* SPLITTER                                       */
/* ============================================== */

QSplitter::handle {{
    background-color: {COLORS['border']};
}}

QSplitter::handle:horizontal {{
    width: 4px;
}}

QSplitter::handle:vertical {{
    height: 4px;
}}

/* ============================================== */
/* TOOL TIP                                       */
/* ============================================== */

QToolTip {{
    background-color: {COLORS['bg_medium']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 6px;
}}

/* ============================================== */
/* STATUS BAR                                     */
/* ============================================== */

QStatusBar {{
    background-color: {COLORS['bg_medium']};
    color: {COLORS['text_secondary']};
    border-top: 1px solid {COLORS['border']};
}}

QStatusBar::item {{
    border: none;
}}

/* ============================================== */
/* MENU                                           */
/* ============================================== */

QMenuBar {{
    background-color: {COLORS['bg_medium']};
    color: {COLORS['text_primary']};
    padding: 4px;
}}

QMenuBar::item:selected {{
    background-color: {COLORS['bg_light']};
}}

QMenu {{
    background-color: {COLORS['bg_medium']};
    border: 1px solid {COLORS['border']};
}}

QMenu::item {{
    padding: 8px 30px;
}}

QMenu::item:selected {{
    background-color: {COLORS['primary']};
}}
"""


def apply_stylesheet(widget):
    """
    Apply the dark stylesheet to a widget or application.

    Args:
        widget: QWidget, QApplication, or QMainWindow instance
    """
    widget.setStyleSheet(DARK_STYLESHEET)


def get_color(name: str) -> str:
    """
    Get a color from the palette by name.

    Args:
        name: Color name (e.g., 'primary', 'bg_dark', 'success')

    Returns:
        Hex color string
    """
    return COLORS.get(name, COLORS['text_primary'])


def get_chart_colors() -> dict:
    """
    Get colors for matplotlib charts that match the theme.

    Returns:
        Dict with color values for charts
    """
    return {
        'background': COLORS['bg_dark'],
        'text': COLORS['text_primary'],
        'grid': COLORS['border'],
        'positive': COLORS['chart_green'],
        'negative': COLORS['chart_red'],
        'primary': COLORS['primary'],
        'accent': COLORS['accent'],
    }
