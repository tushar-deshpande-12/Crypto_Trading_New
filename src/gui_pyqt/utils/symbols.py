"""
Shared symbol list for all panels.

Provides functions to get cryptocurrency symbols from Binance.
"""

from typing import List, Optional, Set
import logging

logger = logging.getLogger(__name__)

# Cache for symbols
_cached_symbols: Optional[List[str]] = None

# Known delisted or suspended symbols (no trading data) - 203 symbols
# Last updated: 2026-01-24 via automated scan
DELISTED_SYMBOLS: Set[str] = {
    '1INCHDOWNUSDT', '1INCHUPUSDT', 'AAVEDOWNUSDT', 'AAVEUPUSDT',
    'ADADOWNUSDT', 'ADAUPUSDT', 'AERGOUSDT', 'AGIXUSDT', 'AIONUSDT',
    'AKROUSDT', 'ALPACAUSDT', 'ALPHAUSDT', 'AMBUSDT', 'ANCUSDT',
    'ANTUSDT', 'ANYUSDT', 'ASTUSDT', 'AUDUSDT', 'AUTOUSDT',
    'BADGERUSDT', 'BAKEUSDT', 'BALUSDT', 'BCCUSDT', 'BCHABCUSDT',
    'BCHDOWNUSDT', 'BCHUPUSDT', 'BEAMUSDT', 'BEARUSDT', 'BETAUSDT',
    'BETHUSDT', 'BKRWUSDT', 'BLZUSDT', 'BNBBEARUSDT', 'BNBBULLUSDT',
    'BNBDOWNUSDT', 'BNBUPUSDT', 'BNXUSDT', 'BONDUSDT', 'BSVUSDT',
    'BSWUSDT', 'BTCDOWNUSDT', 'BTCSTUSDT', 'BTCUPUSDT', 'BTGUSDT',
    'BTSUSDT', 'BTTUSDT', 'BULLUSDT', 'BURGERUSDT', 'BUSDUSDT',
    'BZRXUSDT', 'CLVUSDT', 'COCOSUSDT', 'COMBOUSDT', 'CREAMUSDT',
    'CTXCUSDT', 'CVPUSDT', 'DAIUSDT', 'DARUSDT', 'DNTUSDT',
    'DOCKUSDT', 'DOTDOWNUSDT', 'DOTUPUSDT', 'DREPUSDT', 'ELFUSDT',
    'EOSBEARUSDT', 'EOSBULLUSDT', 'EOSDOWNUSDT', 'EOSUPUSDT', 'EOSUSDT',
    'EPSUSDT', 'EPXUSDT', 'ERDUSDT', 'ERNUSDT', 'ETHBEARUSDT',
    'ETHBULLUSDT', 'ETHDOWNUSDT', 'ETHUPUSDT', 'FILDOWNUSDT', 'FILUPUSDT',
    'FIROUSDT', 'FISUSDT', 'FLMUSDT', 'FORUSDT', 'FRONTUSDT',
    'FTMUSDT', 'FXSUSDT', 'GALUSDT', 'GBPUSDT', 'GFTUSDT',
    'GTOUSDT', 'GXSUSDT', 'HARDUSDT', 'HCUSDT', 'HIFIUSDT',
    'HNTUSDT', 'IRISUSDT', 'KDAUSDT', 'KEEPUSDT', 'KEYUSDT',
    'KLAYUSDT', 'KMDUSDT', 'KP3RUSDT', 'LENDUSDT', 'LEVERUSDT',
    'LINAUSDT', 'LINKDOWNUSDT', 'LINKUPUSDT', 'LITUSDT', 'LOKAUSDT',
    'LOOMUSDT', 'LTCDOWNUSDT', 'LTCUPUSDT', 'LTOUSDT', 'MATICUSDT',
    'MCOUSDT', 'MCUSDT', 'MDXUSDT', 'MFTUSDT', 'MIRUSDT',
    'MITHUSDT', 'MKRUSDT', 'MOBUSDT', 'MULTIUSDT', 'NANOUSDT',
    'NBSUSDT', 'NEBLUSDT', 'NPXSUSDT', 'NULSUSDT', 'NUUSDT',
    'OAXUSDT', 'OCEANUSDT', 'OMGUSDT', 'OMNIUSDT', 'OOKIUSDT',
    'ORNUSDT', 'PAXUSDT', 'PDAUSDT', 'PERLUSDT', 'PERPUSDT',
    'PLAUSDT', 'PNTUSDT', 'POLSUSDT', 'POLYUSDT', 'PROSUSDT',
    'RAMPUSDT', 'REEFUSDT', 'REIUSDT', 'RENUSDT', 'REPUSDT',
    'RGTUSDT', 'RNDRUSDT', 'SLFUSDT', 'SNTUSDT', 'SRMUSDT',
    'STMXUSDT', 'STORMUSDT', 'STPTUSDT', 'STRATUSDT', 'SUSDUSDT',
    'SUSHIDOWNUSDT', 'SUSHIUPUSDT', 'SXPDOWNUSDT', 'SXPUPUSDT', 'TCTUSDT',
    'TOMOUSDT', 'TORNUSDT', 'TRIBEUSDT', 'TROYUSDT', 'TRXDOWNUSDT',
    'TRXUPUSDT', 'TVKUSDT', 'UFTUSDT', 'UNFIUSDT', 'UNIDOWNUSDT',
    'UNIUPUSDT', 'USDSBUSDT', 'USDSUSDT', 'USTUSDT', 'VENUSDT',
    'VGXUSDT', 'VIBUSDT', 'VIDTUSDT', 'VITEUSDT', 'VOXELUSDT',
    'WAVESUSDT', 'WINGUSDT', 'WNXMUSDT', 'WRXUSDT', 'WTCUSDT',
    'XEMUSDT', 'XLMDOWNUSDT', 'XLMUPUSDT', 'XMRUSDT', 'XRPBEARUSDT',
    'XRPBULLUSDT', 'XRPDOWNUSDT', 'XRPUPUSDT', 'XTZDOWNUSDT', 'XTZUPUSDT',
    'XZCUSDT', 'YFIDOWNUSDT', 'YFIIUSDT', 'YFIUPUSDT',
}


def get_all_usdt_symbols(use_cache: bool = True) -> List[str]:
    """
    Get all USDT trading pairs from Binance.

    Args:
        use_cache: If True, return cached symbols if available

    Returns:
        List of symbol strings (e.g., ['BTCUSDT', 'ETHUSDT', ...])
    """
    global _cached_symbols

    if use_cache and _cached_symbols is not None:
        return _cached_symbols

    try:
        from src.api.ccxt_client import get_ccxt_client
        client = get_ccxt_client('binance')
        markets = client.exchange.load_markets()

        # Filter USDT spot pairs (exclude futures/margin denoted by ':')
        symbols = []
        for symbol in markets.keys():
            if symbol.endswith('/USDT') and ':' not in symbol:
                # Convert BTC/USDT to BTCUSDT
                sym = symbol.replace('/USDT', 'USDT')
                # Skip delisted/problematic symbols
                if sym not in DELISTED_SYMBOLS:
                    symbols.append(sym)

        # Sort alphabetically
        symbols.sort()

        # Move popular symbols to the top
        top_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'AVAXUSDT', 'LINKUSDT',
            'SHIBUSDT', 'LTCUSDT', 'UNIUSDT', 'ATOMUSDT', 'NEARUSDT',
            'ARBUSDT', 'OPUSDT', 'SUIUSDT', 'APTUSDT', 'POLUSDT',
            'PEPEUSDT', 'FETUSDT', 'RENDERUSDT', 'INJUSDT', 'TIAUSDT',
        ]

        # Reorder: top symbols first, then rest alphabetically
        ordered_symbols = []
        for sym in top_symbols:
            if sym in symbols:
                ordered_symbols.append(sym)
                symbols.remove(sym)
        ordered_symbols.extend(symbols)

        _cached_symbols = ordered_symbols
        logger.info(f"Loaded {len(ordered_symbols)} USDT symbols from Binance")
        return ordered_symbols

    except Exception as e:
        logger.error(f"Failed to fetch symbols: {e}")
        # Return default fallback list
        return get_default_symbols()


def get_default_symbols() -> List[str]:
    """
    Get default symbol list (fallback when API fails).

    Returns:
        List of common trading symbols
    """
    return [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
        'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'AVAXUSDT', 'LINKUSDT',
        'SHIBUSDT', 'LTCUSDT', 'UNIUSDT', 'ATOMUSDT', 'ETCUSDT',
        'POLUSDT', 'ARBUSDT', 'OPUSDT', 'SUIUSDT', 'APTUSDT',
        'NEARUSDT', 'SEIUSDT', 'TIAUSDT', 'STXUSDT', 'INJUSDT',
        'FETUSDT', 'RENDERUSDT', 'GRTUSDT', 'IMXUSDT', 'AXSUSDT',
        'SANDUSDT', 'MANAUSDT', 'GALAUSDT', 'PEPEUSDT', 'WIFUSDT',
        'AAVEUSDT', 'RUNEUSDT', 'FILUSDT', 'XLMUSDT', 'TRXUSDT',
    ]


def clear_symbol_cache():
    """Clear the cached symbols to force a refresh."""
    global _cached_symbols
    _cached_symbols = None
