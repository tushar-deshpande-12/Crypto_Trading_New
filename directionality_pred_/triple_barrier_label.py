import numpy as np
import pandas as pd

def apply_triple_barrier(
    df,
    horizon=60,
    pt_mult=2.0,
    sl_mult=1.0,
    atr_window=14
):
    """
    Triple barrier labeling using OHLCV only
    Returns labels: +1, -1, 0
    """

    df = df.copy()
    df['atr'] = (
        (df['high'] - df['low'])
        .rolling(atr_window)
        .mean()
    )

    labels = np.zeros(len(df))

    for i in range(len(df) - horizon):
        entry = df['close'].iloc[i]
        atr = df['atr'].iloc[i]

        if np.isnan(atr):
            continue

        pt = entry + pt_mult * atr
        sl = entry - sl_mult * atr

        future = df.iloc[i+1:i+horizon+1]

        hit_pt = (future['high'] >= pt).any()
        hit_sl = (future['low'] <= sl).any()

        if hit_pt and not hit_sl:
            labels[i] = 1
        elif hit_sl and not hit_pt:
            labels[i] = -1
        else:
            labels[i] = 0

    df['label'] = labels
    return df.dropna()
