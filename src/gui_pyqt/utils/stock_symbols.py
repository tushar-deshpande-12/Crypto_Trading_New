"""
Indian Stock Market Symbols

Provides functions to get stock symbols from NSE (National Stock Exchange of India).
Includes Nifty 50, Nifty Next 50, and sector-wise categorization.
"""

from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Cache for symbols
_cached_stock_symbols: Optional[List[str]] = None


# Nifty 50 stocks (Top 50 NSE companies by market cap)
NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "BAJFINANCE",
    "TITAN", "SUNPHARMA", "ULTRACEMCO", "NESTLEIND", "WIPRO",
    "TATAMOTORS", "M&M", "HCLTECH", "NTPC", "POWERGRID",
    "TATASTEEL", "JSWSTEEL", "ONGC", "ADANIENT", "ADANIPORTS",
    "COALINDIA", "BAJAJFINSV", "TECHM", "GRASIM", "DRREDDY",
    "INDUSINDBK", "HINDALCO", "BPCL", "CIPLA", "DIVISLAB",
    "APOLLOHOSP", "HEROMOTOCO", "EICHERMOT", "BRITANNIA", "UPL",
    "TATACONSUM", "SBILIFE", "HDFCLIFE", "BAJAJ-AUTO", "LTIM"
]

# Nifty Next 50 stocks
NIFTY_NEXT_50 = [
    "ADANIGREEN", "ADANIPOWER", "AMBUJACEM", "ATGL", "AUROPHARMA",
    "BANKBARODA", "BEL", "BERGEPAINT", "BOSCHLTD", "CANBK",
    "CHOLAFIN", "COLPAL", "CONCOR", "DALBHARAT", "DLF",
    "GAIL", "GODREJCP", "HAVELLS", "ICICIPRULI", "ICICIGI",
    "IDEA", "IDFCFIRSTB", "IGL", "INDHOTEL", "INDIGO",
    "IOC", "IRCTC", "IRFC", "JINDALSTEL", "JSWENERGY",
    "LICI", "LUPIN", "MANKIND", "MAXHEALTH", "MCDOWELL-N",
    "NHPC", "NMDC", "NYKAA", "OBEROIRLTY", "OFSS",
    "PAYTM", "PERSISTENT", "PGHH", "PIIND", "PNB",
    "POLYCAB", "SRF", "TATAPOWER", "TORNTPHARM", "TVSMOTOR"
]

# Sector-wise categorization
SECTORS: Dict[str, List[str]] = {
    "Banking": [
        "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
        "INDUSINDBK", "BANKBARODA", "PNB", "CANBK", "IDFCFIRSTB"
    ],
    "IT": [
        "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM",
        "LTIM", "PERSISTENT", "OFSS"
    ],
    "Pharma": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "AUROPHARMA",
        "LUPIN", "TORNTPHARM", "MANKIND", "APOLLOHOSP", "MAXHEALTH"
    ],
    "Auto": [
        "TATAMOTORS", "MARUTI", "M&M", "BAJAJ-AUTO", "HEROMOTOCO",
        "EICHERMOT", "TVSMOTOR", "BOSCHLTD"
    ],
    "FMCG": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM",
        "GODREJCP", "COLPAL", "PGHH", "MCDOWELL-N"
    ],
    "Energy": [
        "RELIANCE", "ONGC", "NTPC", "POWERGRID", "BPCL",
        "COALINDIA", "GAIL", "IOC", "ADANIGREEN", "ADANIPOWER",
        "TATAPOWER", "JSWENERGY", "NHPC"
    ],
    "Metals": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "JINDALSTEL", "NMDC"
    ],
    "Finance": [
        "BAJFINANCE", "BAJAJFINSV", "SBILIFE", "HDFCLIFE", "ICICIPRULI",
        "ICICIGI", "CHOLAFIN", "LICI", "IRFC"
    ],
    "Infrastructure": [
        "LT", "ADANIENT", "ADANIPORTS", "ULTRACEMCO", "GRASIM",
        "AMBUJACEM", "DLF", "OBEROIRLTY", "CONCOR"
    ],
    "Consumer": [
        "TITAN", "ASIANPAINT", "BERGEPAINT", "HAVELLS", "POLYCAB",
        "INDIGO", "INDHOTEL", "IRCTC", "NYKAA", "PAYTM"
    ]
}


def get_nifty50_symbols() -> List[str]:
    """
    Get Nifty 50 stock symbols.

    Returns:
        List of Nifty 50 symbol strings
    """
    return NIFTY_50.copy()


def get_nifty_next50_symbols() -> List[str]:
    """
    Get Nifty Next 50 stock symbols.

    Returns:
        List of Nifty Next 50 symbol strings
    """
    return NIFTY_NEXT_50.copy()


def get_all_stock_symbols(use_cache: bool = True) -> List[str]:
    """
    Get all Indian stock symbols (Nifty 50 + Nifty Next 50).

    Args:
        use_cache: If True, return cached symbols if available

    Returns:
        List of symbol strings
    """
    global _cached_stock_symbols

    if use_cache and _cached_stock_symbols is not None:
        return _cached_stock_symbols

    # Combine Nifty 50 and Nifty Next 50
    all_symbols = NIFTY_50 + NIFTY_NEXT_50

    # Remove duplicates while preserving order
    seen = set()
    unique_symbols = []
    for sym in all_symbols:
        if sym not in seen:
            seen.add(sym)
            unique_symbols.append(sym)

    _cached_stock_symbols = unique_symbols
    logger.info(f"Loaded {len(unique_symbols)} NSE stock symbols")
    return unique_symbols


def get_stocks_by_sector(sector: str) -> List[str]:
    """
    Get stock symbols for a specific sector.

    Args:
        sector: Sector name (e.g., "Banking", "IT", "Pharma")

    Returns:
        List of symbol strings in that sector
    """
    return SECTORS.get(sector, []).copy()


def get_all_sectors() -> List[str]:
    """
    Get list of all available sectors.

    Returns:
        List of sector names
    """
    return list(SECTORS.keys())


def get_popular_stocks() -> List[str]:
    """
    Get list of most popular/liquid stocks.

    Returns:
        List of popular symbol strings
    """
    return [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
        "TATAMOTORS", "BAJFINANCE", "TITAN", "AXISBANK", "LT",
        "MARUTI", "SUNPHARMA", "HCLTECH", "WIPRO", "NTPC"
    ]


def clear_stock_symbol_cache():
    """Clear the cached stock symbols to force a refresh."""
    global _cached_stock_symbols
    _cached_stock_symbols = None


def is_valid_nse_symbol(symbol: str) -> bool:
    """
    Check if a symbol is a valid NSE stock symbol.

    Args:
        symbol: Stock symbol to check

    Returns:
        True if valid NSE symbol, False otherwise
    """
    all_symbols = get_all_stock_symbols()
    return symbol.upper() in all_symbols
