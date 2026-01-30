import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_ice_safe(preds, future_returns, min_samples=50):
    """
    Robust Information Coefficient (Spearman)

    Returns NaN-safe ICE with explicit diagnostics
    """

    preds = np.asarray(preds).astype(float)
    future_returns = np.asarray(future_returns).astype(float)

    # 1️⃣ Remove NaNs / infs
    mask = (
        np.isfinite(preds) &
        np.isfinite(future_returns)
    )

    preds = preds[mask]
    future_returns = future_returns[mask]

    # 2️⃣ Minimum sample check
    if len(preds) < min_samples:
        return {
            "ic": 0.0,
            "p_value": 1.0,
            "reason": "INSUFFICIENT_SAMPLES"
        }

    # 3️⃣ Variance check (CRITICAL)
    if np.std(preds) == 0 or np.std(future_returns) == 0:
        return {
            "ic": 0.0,
            "p_value": 1.0,
            "reason": "ZERO_VARIANCE"
        }

    # 4️⃣ Compute IC
    ic, pval = spearmanr(preds, future_returns)

    # 5️⃣ Final safety
    if np.isnan(ic):
        return {
            "ic": 0.0,
            "p_value": 1.0,
            "reason": "NUMERICAL_NAN"
        }

    return {
        "ic": float(ic),
        "p_value": float(pval),
        "reason": "OK"
    }


def compute_directional_accuracy(preds, future_returns):
    """
    Directional accuracy (sign agreement)
    """
    pred_dir = np.sign(preds)
    true_dir = np.sign(future_returns)

    valid = (pred_dir != 0) & (true_dir != 0)
    acc = (pred_dir[valid] == true_dir[valid]).mean()

    return acc


def compute_trade_metrics(trade_returns):
    """
    Per-trade performance metrics
    """
    trade_returns = np.array(trade_returns)

    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    return {
        "avg_profit": wins.mean() if len(wins) > 0 else 0.0,
        "avg_loss": losses.mean() if len(losses) > 0 else 0.0,
        "win_rate": len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0.0,
        "num_trades": len(trade_returns)
    }


def compute_sharpe(returns, freq=252):
    """
    Sharpe ratio (annualized)
    """
    returns = np.array(returns)

    if returns.std() == 0:
        return 0.0

    sharpe = np.sqrt(freq) * returns.mean() / returns.std()
    return sharpe


def compute_market_coverage(trade_mask):
    """
    Fraction of time model is active
    """
    return trade_mask.mean()
