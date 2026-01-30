import pandas as pd
from risk import compute_atr, atr_position_size, atr_trade_exit
import numpy as np


def ranked_backtest_atr(df,
                         preds,
                         trade_mask,
                         top_k=5,
                         initial_capital=10_000,
                         risk_pct=0.01,
                         stop_mult=1.5,
                         tp_mult=3.0):

    df = df.copy()
    df['pred'] = preds
    df['trade'] = trade_mask.astype(bool)
    df['atr'] = compute_atr(df)

    capital = initial_capital
    equity_curve = []
    trade_returns = []

    for t, slice_ in df.groupby(df.index):
        slice_ = slice_.loc[slice_['trade'] & slice_['atr'].notna()]

        if len(slice_) == 0:
            equity_curve.append(capital)
            continue

        top = slice_.nlargest(top_k, 'pred')

        pnl_t = 0.0

        for _, row in top.iterrows():
            size = atr_position_size(
                capital,
                row['atr'],
                risk_pct,
                stop_mult
            )

            if size == 0:
                continue

            ret = atr_trade_exit(
                entry_price=row['close'],
                high=row['high'],
                low=row['low'],
                atr=row['atr'],
                stop_mult=stop_mult,
                tp_mult=tp_mult
            )

            pnl = size * ret
            pnl_t += pnl
            trade_returns.append(ret)

        capital += pnl_t
        equity_curve.append(capital)

        

    return equity_curve, trade_returns



def backtest_fixed_profit(
    df,
    preds,
    trade_mask,
    take_profit_pct=0.02,
    stop_loss_pct=0.005,
    price_col="close"
):
    """
    Fixed TP / SL backtest

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain price column
    preds : array-like
        Model predictions (aligned with df)
    trade_mask : pandas.Series (bool)
        When True → model allowed to trade
    take_profit_pct : float
        Take profit threshold (e.g. 0.02 = +2%)
    stop_loss_pct : float
        Stop loss threshold (e.g. 0.005 = -0.5%)
    price_col : str
        Column used for execution price

    Returns
    -------
    trades_df : pandas.DataFrame
        One row per trade
    equity_curve : pandas.Series
        Cumulative PnL over time
    """

    df = df.copy()
    df["pred"] = preds
    df["trade_allowed"] = trade_mask.astype(bool)

    position = 0
    entry_price = None
    entry_time = None

    trades = []
    equity = []
    cumulative_pnl = 0.0

    for i in range(len(df)):
        price = df[price_col].iloc[i]
        ts = df.index[i]

        # =========================
        # 1️⃣ MANAGE OPEN POSITION
        # =========================
        if position == 1:
            pnl_pct = (price - entry_price) / entry_price

            # STOP LOSS
            if pnl_pct <= -stop_loss_pct:
                cumulative_pnl += pnl_pct
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl_pct": pnl_pct,
                    "exit_reason": "STOP_LOSS"
                })
                position = 0
                entry_price = None
                equity.append(cumulative_pnl)
                continue

            # TAKE PROFIT
            if pnl_pct >= take_profit_pct:
                cumulative_pnl += pnl_pct
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl_pct": pnl_pct,
                    "exit_reason": "TAKE_PROFIT"
                })
                position = 0
                entry_price = None
                equity.append(cumulative_pnl)
                continue

            # Hold position
            equity.append(cumulative_pnl)
            continue

        # =========================
        # 2️⃣ OPEN NEW POSITION
        # =========================
        if position == 0:
            if df["trade_allowed"].iloc[i] and df["pred"].iloc[i] > 0:
                position = 1
                entry_price = price
                entry_time = ts

            equity.append(cumulative_pnl)

    trades_df = pd.DataFrame(trades)
    equity_curve = pd.Series(equity, index=df.index)

    return trades_df, equity_curve





def compute_confidence_zscore(preds, window=200):
    """
    Rolling z-score confidence for ranking models
    """
    preds = pd.Series(preds)

    mean = preds.rolling(window).mean()
    std = preds.rolling(window).std() + 1e-8

    confidence = (preds - mean).abs() / std
    return confidence



def backtest_fixed_profit_with_confidence(
    df,
    preds,
    initial_capital=10000.0,
    position_fraction=0.1,      # % capital per trade
    confidence_window=200,
    confidence_threshold=1.0,
    take_profit=0.02,           # +2%
    stop_loss=0.005,            # -0.5%
    trading_fee_pct=0.001,      # 0.1%
    tds_pct=0.01                # 1% (applied on abs PnL)
):
    import numpy as np
    import pandas as pd

    df = df.copy()
    preds = pd.Series(preds, index=df.index)

    # ===============================
    # 1️⃣ Confidence (rolling Z-score)
    # ===============================
    mean = preds.rolling(confidence_window).mean()
    std = preds.rolling(confidence_window).std() + 1e-8
    confidence = (preds - mean).abs() / std
    trade_allowed = confidence > confidence_threshold

    prices = df["close"].values

    # ===============================
    # 2️⃣ State
    # ===============================
    capital = initial_capital
    equity_curve = [capital]

    trades = wins = losses = 0
    trade_returns = []
    trade_profits = []
    trade_losses = []

    # Store decision points
    buy_points = []
    sell_points = []

    # ===============================
    # 3️⃣ Backtest loop
    # ===============================
    for i in range(len(df) - 1):

        if not trade_allowed.iloc[i]:
            equity_curve.append(capital)
            continue

        direction = np.sign(preds.iloc[i])
        if direction == 0:
            equity_curve.append(capital)
            continue

        entry_price = prices[i]
        position_size = capital * position_fraction
        trades += 1

        # Record BUY / SELL decision
        action = "BUY" if direction > 0 else "SELL"
        buy_points.append((df.index[i], df["close"].iloc[i], action))

        for j in range(i + 1, len(df)):
            move = (prices[j] - entry_price) / entry_price
            pnl_pct = direction * move

            # ===== Exit condition =====
            if pnl_pct >= take_profit or pnl_pct <= -stop_loss:

                raw_pct = take_profit if pnl_pct >= take_profit else -stop_loss
                gross_pnl = position_size * raw_pct

                fee = abs(gross_pnl) * trading_fee_pct
                tds = abs(gross_pnl) * tds_pct
                net_pnl = gross_pnl - fee - tds

                capital += net_pnl
                trade_returns.append(net_pnl / position_size)

                # Record EXIT
                sell_points.append(
                    (df.index[j], df["close"].iloc[j], "TP" if net_pnl > 0 else "SL")
                )

                if net_pnl > 0:
                    wins += 1
                    trade_profits.append(net_pnl)
                else:
                    losses += 1
                    trade_losses.append(net_pnl)

                break

        equity_curve.append(capital)

    equity_curve = np.array(equity_curve)

    # ===============================
    # 4️⃣ Metrics
    # ===============================
    market_coverage = trades / len(df)
    win_rate = wins / trades if trades > 0 else 0.0

    avg_profit = np.mean(trade_profits) if trade_profits else 0.0
    avg_loss = np.mean(trade_losses) if trade_losses else 0.0

    sharpe = (
        np.mean(trade_returns) / (np.std(trade_returns) + 1e-8)
        if len(trade_returns) > 1 else 0.0
    )

    return {
        "Initial Capital": initial_capital,
        "Final Capital": capital,
        "Equity Generated": capital - initial_capital,
        "Trades": trades,
        "Market Coverage": market_coverage,
        "Win Rate": win_rate,
        "Avg Profit / Trade": avg_profit,
        "Avg Loss / Trade": avg_loss,
        "Sharpe Ratio": sharpe,
        "Equity Curve": equity_curve,

        # 🔥 NEW DIAGNOSTIC OUTPUTS
        "Buy Points": buy_points,     # (timestamp, price, BUY/SELL)
        "Sell Points": sell_points    # (timestamp, price, TP/SL)
    }
