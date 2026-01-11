"""
GUI Styling Module
Centralized styling configuration for tkinter widgets
"""

from tkinter import ttk
from src.core.config import AppConfig


def setup_styles():
    """Configure and apply ttk styles"""
    style = ttk.Style()
    style.theme_use('clam')

    # Treeview (Table) Styling
    style.configure(
        "Crypto.Treeview",
        background=AppConfig.COLOR_BG_DARK,
        foreground=AppConfig.COLOR_TEXT_PRIMARY,
        fieldbackground=AppConfig.COLOR_BG_DARK,
        borderwidth=0,
        relief='flat',
        font=(AppConfig.FONT_MONO, AppConfig.FONT_SIZE_SMALL),
        rowheight=28
    )

    style.configure(
        "Crypto.Treeview.Heading",
        background=AppConfig.COLOR_BG_MEDIUM,
        foreground=AppConfig.COLOR_TEXT_PRIMARY,
        borderwidth=0,
        relief='flat',
        font=(AppConfig.FONT_MONO, AppConfig.FONT_SIZE_SMALL, 'bold')
    )

    style.map(
        'Crypto.Treeview.Heading',
        background=[('active', AppConfig.COLOR_BG_LIGHT)]
    )

    style.map(
        'Crypto.Treeview',
        background=[('selected', AppConfig.COLOR_PRIMARY)],
        foreground=[('selected', '#ffffff')]
    )

    # Button Styling
    style.configure(
        "Primary.TButton",
        background=AppConfig.COLOR_PRIMARY,
        foreground="white",
        font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold'),
        padding=(16, 10),
        borderwidth=0,
        relief='flat'
    )

    style.configure(
        "Success.TButton",
        background=AppConfig.COLOR_SUCCESS,
        foreground="white",
        font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold'),
        padding=(16, 10),
        borderwidth=0,
        relief='flat'
    )

    style.configure(
        "Danger.TButton",
        background=AppConfig.COLOR_DANGER,
        foreground="white",
        font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL, 'bold'),
        padding=(16, 10),
        borderwidth=0,
        relief='flat'
    )

    # Frame Styling
    style.configure(
        "Dark.TFrame",
        background=AppConfig.COLOR_BG_DARK,
        borderwidth=0,
        relief='flat'
    )

    style.configure(
        "Medium.TFrame",
        background=AppConfig.COLOR_BG_MEDIUM,
        borderwidth=0,
        relief='flat'
    )

    # Label Styling
    style.configure(
        "Header.TLabel",
        background=AppConfig.COLOR_PRIMARY,
        foreground="white",
        font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_HEADER, 'bold')
    )

    style.configure(
        "Dark.TLabel",
        background=AppConfig.COLOR_BG_DARK,
        foreground=AppConfig.COLOR_TEXT_PRIMARY,
        font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL)
    )

    style.configure(
        "Medium.TLabel",
        background=AppConfig.COLOR_BG_MEDIUM,
        foreground=AppConfig.COLOR_TEXT_PRIMARY,
        font=(AppConfig.FONT_FAMILY, AppConfig.FONT_SIZE_NORMAL)
    )

    # Progressbar
    style.configure(
        "Crypto.Horizontal.TProgressbar",
        background=AppConfig.COLOR_PRIMARY,
        troughcolor=AppConfig.COLOR_BG_MEDIUM,
        borderwidth=0,
        thickness=24,
        relief='flat'
    )


def get_color(color_name: str) -> str:
    """Get color value by name"""
    colors = {
        'bg_dark': AppConfig.COLOR_BG_DARK,
        'bg_medium': AppConfig.COLOR_BG_MEDIUM,
        'bg_light': AppConfig.COLOR_BG_LIGHT,
        'primary': AppConfig.COLOR_PRIMARY,
        'success': AppConfig.COLOR_SUCCESS,
        'danger': AppConfig.COLOR_DANGER,
        'text_primary': AppConfig.COLOR_TEXT_PRIMARY,
        'text_secondary': AppConfig.COLOR_TEXT_SECONDARY,
        'accent': AppConfig.COLOR_ACCENT,
    }
    return colors.get(color_name, AppConfig.COLOR_TEXT_PRIMARY)
