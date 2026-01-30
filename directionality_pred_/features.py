import numpy as np
import pandas as pd

def rank_normalize(X):
    return X.rank(pct=True) - 0.5

def generate_feature_matrix(df,
                            return_windows=(1, 5, 15, 30),
                            vol_windows=(5, 15, 30),
                            ema_pairs=((10, 30), (20, 50)),
                            bb_window=20):
    """
    Generate OHLCV feature matrix for ML trading
    Safe for IC / ranking models
    """

    df = df.copy()

    # -----------------------------
    # 1️⃣ RETURNS (core signal)
    # -----------------------------
    for w in return_windows:
        df[f'return_{w}'] = df['close'].pct_change(w)

    # -----------------------------
    # 2️⃣ VOLATILITY (regime)
    # -----------------------------
    log_ret = np.log(df['close']).diff()

    for w in vol_windows:
        df[f'vol_{w}'] = log_ret.rolling(w).std()

    # Volatility percentile (regime filter)
    df['vol_pct'] = df['vol_30'].rolling(200).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False
    )

    # -----------------------------
    # 3️⃣ TREND STRENGTH
    # -----------------------------
    for fast, slow in ema_pairs:
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        df[f'trend_{fast}_{slow}'] = (ema_fast - ema_slow) / df['close']

    # -----------------------------
    # 4️⃣ PRICE POSITION (compression / expansion)
    # -----------------------------
    mid = df['close'].rolling(bb_window).mean()
    std = df['close'].rolling(bb_window).std()

    df['bb_width'] = (2 * std) / mid
    df['bb_position'] = (df['close'] - mid) / (2 * std)

    # -----------------------------
    # 5️⃣ VOLUME FEATURES (confirmation)
    # -----------------------------
    df['volume_z'] = (
        (df['volume'] - df['volume'].rolling(20).mean()) /
        df['volume'].rolling(20).std()
    )

    df['volume_trend'] = (
        df['volume'].ewm(span=10).mean() /
        df['volume'].ewm(span=30).mean()
    )

    # -----------------------------
    # 6️⃣ RANGE / EFFICIENCY
    # -----------------------------
    df['range_pct'] = (df['high'] - df['low']) / df['close']

    df['efficiency'] = (
        df['close'].diff(10).abs() /
        (df['high'] - df['low']).rolling(10).sum()
    )

    # -----------------------------
    # CLEANUP
    # -----------------------------
    feature_cols = [c for c in df.columns
                    if c not in ['open', 'high', 'low', 'close', 'volume']]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()

    return X, feature_cols

def generate_ohlcv_features(df):
    df = df.copy()

    C = df['close']
    O = df['open']
    H = df['high']
    L = df['low']
    V = df['volume']

    feature_cols = []

    # ===== Returns =====
    df['ret_1']  = np.log(C / C.shift(1))
    df['ret_5']  = np.log(C / C.shift(5))
    df['ret_20'] = np.log(C / C.shift(20))
    df['ret_60'] = np.log(C / C.shift(60))
    feature_cols += ['ret_1', 'ret_5', 'ret_20', 'ret_60']

    # ===== Volatility structure =====
    df['vol_5']  = df['ret_1'].rolling(5).std()
    df['vol_20'] = df['ret_1'].rolling(20).std()
    df['vol_ratio'] = df['vol_5'] / (df['vol_20'] + 1e-8)
    feature_cols += ['vol_5', 'vol_20', 'vol_ratio']

    # ===== Trend slope =====
    def slope(series, window):
        x = np.arange(window)
        return series.rolling(window).apply(
            lambda y: np.polyfit(x, y, 1)[0], raw=True
        )

    df['slope_20'] = slope(np.log(C), 20)
    df['slope_60'] = slope(np.log(C), 60)
    feature_cols += ['slope_20', 'slope_60']

    # ===== Candle anatomy =====
    df['body'] = (C - O) / O
    df['upper_wick'] = (H - np.maximum(O, C)) / C
    df['lower_wick'] = (np.minimum(O, C) - L) / C
    df['range_pct'] = (H - L) / C
    feature_cols += ['body', 'upper_wick', 'lower_wick', 'range_pct']

    # ===== Volume pressure =====
    df['vol_z'] = (V - V.rolling(20).mean()) / (V.rolling(20).std() + 1e-8)
    df['vol_change'] = V / (V.shift(1) + 1e-8)
    feature_cols += ['vol_z', 'vol_change']

    # ===== Relative position =====
    def position(high, low, close, window):
        return (close - low.rolling(window).min()) / (
            high.rolling(window).max() - low.rolling(window).min() + 1e-8
        )

    df['pos_20'] = position(H, L, C, 20)
    df['pos_60'] = position(H, L, C, 60)
    feature_cols += ['pos_20', 'pos_60']

    return df, feature_cols