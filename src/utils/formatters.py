"""
Formatting utilities for displaying cryptocurrency data
Professional number formatting for traders
"""


def format_price(price: float, decimals: int = 8) -> str:
    """
    Format price with appropriate decimal places
    Removes trailing zeros for cleaner display

    Args:
        price: Price value
        decimals: Maximum decimal places

    Returns:
        Formatted price string with $ prefix
    """
    if price == 0:
        return "$0.00"

    # Use more decimals for very small prices
    if price < 0.01:
        decimals = 8
    elif price < 1:
        decimals = 6
    elif price < 100:
        decimals = 4
    else:
        decimals = 2

    formatted = f"${price:,.{decimals}f}"
    # Remove trailing zeros after decimal point
    if '.' in formatted:
        formatted = formatted.rstrip('0').rstrip('.')

    return formatted


def format_volume(volume: float) -> str:
    """
    Format large volume numbers with K/M/B suffixes

    Args:
        volume: Volume value

    Returns:
        Formatted volume string
    """
    if volume >= 1_000_000_000:
        return f"{volume / 1_000_000_000:,.2f}B"
    elif volume >= 1_000_000:
        return f"{volume / 1_000_000:,.2f}M"
    elif volume >= 1_000:
        return f"{volume / 1_000:,.2f}K"
    else:
        return f"{volume:,.2f}"


def format_percentage(pct: float) -> str:
    """
    Format percentage with +/- sign and color indication

    Args:
        pct: Percentage value

    Returns:
        Formatted percentage string with sign
    """
    return f"{pct:+.2f}%"


def format_number(num: float, decimals: int = 2) -> str:
    """
    Generic number formatter with thousand separators

    Args:
        num: Number to format
        decimals: Decimal places

    Returns:
        Formatted number string
    """
    return f"{num:,.{decimals}f}"
