"""
PyQt6 Stylesheet - Minimalist Dark Theme

Clean, refined dark theme for the Crypto AI Predictor application.
"""

# Color Palette — Minimalist
COLORS = {
    # Backgrounds: deep, near-black with subtle layering
    'bg_dark': '#0e0e11',
    'bg_medium': '#17171c',
    'bg_light': '#21212a',
    'bg_lighter': '#2b2b35',

    # Primary: refined indigo
    'primary': '#635bff',
    'primary_hover': '#7a73ff',
    'primary_pressed': '#534cc9',

    # Semantic: soft, non-aggressive
    'success': '#3ecf8e',
    'success_hover': '#54d9a0',
    'danger': '#f87171',
    'danger_hover': '#fca5a5',
    'warning': '#eab308',

    # Text: clean hierarchy
    'text_primary': '#dfdfe3',
    'text_secondary': '#6e6e7a',
    'text_disabled': '#3e3e47',

    # Accent: soft periwinkle for headers & highlights
    'accent': '#a5b4fc',

    # Borders & chrome
    'border': '#24242e',
    'scrollbar': '#2e2e3a',
    'scrollbar_hover': '#3e3e4c',

    # Chart: clear but soft
    'chart_green': '#3ecf8e',
    'chart_red': '#f87171',
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
/* GLOBAL                                         */
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
/* TABS                                           */
/* ============================================== */

QTabWidget::pane {{
    border: none;
    background-color: {COLORS['bg_dark']};
    padding: 4px;
}}

QTabBar::tab {{
    background-color: transparent;
    color: {COLORS['text_secondary']};
    padding: 10px 22px;
    border: none;
    border-bottom: 2px solid transparent;
    margin-right: 2px;
    font-size: {FONTS['size_normal']}pt;
}}

QTabBar::tab:selected {{
    color: {COLORS['text_primary']};
    border-bottom: 2px solid {COLORS['primary']};
    font-weight: bold;
}}

QTabBar::tab:hover:!selected {{
    color: {COLORS['text_primary']};
    border-bottom: 2px solid {COLORS['border']};
}}

/* ============================================== */
/* BUTTONS                                        */
/* ============================================== */

QPushButton {{
    background-color: {COLORS['primary']};
    color: white;
    border: none;
    padding: 8px 18px;
    border-radius: 6px;
    font-weight: 600;
    font-size: {FONTS['size_normal']}pt;
    min-width: 72px;
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
    color: #0e0e11;
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
/* INPUTS                                         */
/* ============================================== */

QLineEdit {{
    background-color: {COLORS['bg_light']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 7px 12px;
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
    border-radius: 6px;
    padding: 5px 10px;
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
    border-radius: 6px;
    padding: 7px 12px;
    color: {COLORS['text_primary']};
    min-width: 120px;
}}

QComboBox:hover {{
    border-color: {COLORS['primary']};
}}

QComboBox::drop-down {{
    border: none;
    width: 28px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {COLORS['text_secondary']};
    margin-right: 10px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS['bg_medium']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    selection-background-color: {COLORS['primary']};
    selection-color: white;
    padding: 4px;
}}

/* ============================================== */
/* TABLES                                         */
/* ============================================== */

QTableView {{
    background-color: {COLORS['bg_dark']};
    alternate-background-color: {COLORS['bg_medium']};
    gridline-color: {COLORS['border']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    selection-background-color: {COLORS['primary']};
    selection-color: white;
}}

QTableView::item {{
    padding: 6px;
    border-bottom: 1px solid {COLORS['border']};
}}

QTableView::item:selected {{
    background-color: {COLORS['primary']};
}}

QHeaderView::section {{
    background-color: {COLORS['bg_medium']};
    color: {COLORS['text_secondary']};
    padding: 8px;
    border: none;
    border-bottom: 1px solid {COLORS['border']};
    font-weight: 600;
    font-size: {FONTS['size_small']}pt;
    text-transform: uppercase;
}}

QHeaderView::section:hover {{
    color: {COLORS['text_primary']};
}}

/* ============================================== */
/* SCROLLBARS                                     */
/* ============================================== */

QScrollBar:vertical {{
    background-color: transparent;
    width: 8px;
    border: none;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS['scrollbar']};
    border-radius: 4px;
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
    background-color: transparent;
    height: 8px;
    border: none;
}}

QScrollBar::handle:horizontal {{
    background-color: {COLORS['scrollbar']};
    border-radius: 4px;
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
    height: 20px;
    font-weight: 600;
    font-size: {FONTS['size_small']}pt;
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
    color: {COLORS['text_primary']};
    padding: 8px 0px;
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
    width: 16px;
    height: 16px;
    border: 1px solid {COLORS['border']};
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
    width: 16px;
    height: 16px;
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
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
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 10px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: {COLORS['text_secondary']};
    font-size: {FONTS['size_small']}pt;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* ============================================== */
/* TEXT EDIT                                       */
/* ============================================== */

QTextEdit, QPlainTextEdit {{
    background-color: {COLORS['bg_light']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
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
    height: 6px;
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background-color: {COLORS['primary']};
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background-color: {COLORS['primary_hover']};
}}

QSlider::sub-page:horizontal {{
    background-color: {COLORS['primary']};
    border-radius: 3px;
}}

/* ============================================== */
/* SPLITTER                                       */
/* ============================================== */

QSplitter::handle {{
    background-color: {COLORS['border']};
}}

QSplitter::handle:horizontal {{
    width: 1px;
}}

QSplitter::handle:vertical {{
    height: 1px;
}}

/* ============================================== */
/* TOOLTIP                                        */
/* ============================================== */

QToolTip {{
    background-color: {COLORS['bg_medium']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 6px 10px;
}}

/* ============================================== */
/* STATUS BAR                                     */
/* ============================================== */

QStatusBar {{
    background-color: {COLORS['bg_medium']};
    color: {COLORS['text_secondary']};
    border-top: 1px solid {COLORS['border']};
    font-size: {FONTS['size_small']}pt;
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
    border-radius: 4px;
}}

QMenu {{
    background-color: {COLORS['bg_medium']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 4px;
}}

QMenu::item {{
    padding: 6px 28px;
    border-radius: 4px;
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
