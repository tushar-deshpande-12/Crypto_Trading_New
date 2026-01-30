import pandas as pd
import numpy as np

def compute_regime_features(df):
    df = df.copy()

    df['return'] = df['close'].pct_change()
    df['vol'] = df['return'].rolling(20).std()
    df['vol_pct'] = df['vol'].rank(pct=True)

    df['ema_fast'] = df['close'].ewm(span=20).mean()
    df['ema_slow'] = df['close'].ewm(span=50).mean()
    df['trend_strength'] = (df['ema_fast'] - df['ema_slow']) / df['close']

    return df.dropna()

def regime_filter(df,
                  vol_thresh=0.6,
                  trend_thresh=0.002):
    """
    Returns boolean: trade_allowed
    """

    cond_vol = df['vol_pct'] > vol_thresh
    cond_trend = abs(df['trend_strength']) > trend_thresh

    df['trade_regime'] = cond_vol & cond_trend
    return df
