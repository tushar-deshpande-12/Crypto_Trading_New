import pandas as pd
import numpy as np


def compute_atr(df, window=14):
    high = df['high']
    low = df['low']
    close = df['close']

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(window).mean()
    return atr

def atr_position_size(capital,
                      atr,
                      risk_pct=0.01,
                      stop_mult=1.5):
    """
    Returns position size (fraction of capital)
    """
    risk_amount = capital * risk_pct
    stop_distance = atr * stop_mult

    if stop_distance == 0 or np.isnan(stop_distance):
        return 0.0

    position = risk_amount / stop_distance
    return position

def atr_trade_exit(entry_price,
                   high,
                   low,
                   atr,
                   stop_mult=1.5,
                   tp_mult=3.0):
    """
    Returns trade return based on ATR TP/SL
    """

    stop_price = entry_price - stop_mult * atr
    tp_price = entry_price + tp_mult * atr

    # Stop loss hit
    if low <= stop_price:
        return (stop_price / entry_price) - 1

    # Take profit hit
    if high >= tp_price:
        return (tp_price / entry_price) - 1

    # Neither hit → exit at close
    return (high / entry_price) - 1
