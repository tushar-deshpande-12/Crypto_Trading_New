import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go

def plot_feature_histograms(
    X,
    bins=50,
    cols_per_row=4,
    figsize=(18, 12),
    clip_quantile=0.01
):
    """
    Plot histograms for all numeric features in X

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix (numeric only)
    bins : int
        Number of histogram bins
    cols_per_row : int
        Grid width
    figsize : tuple
        Matplotlib figure size
    clip_quantile : float
        Tail clipping for visualization (e.g. 0.01 = 1% each side)
    """

    # Keep only numeric features
    X_num = X.select_dtypes(include=[np.number])

    n_features = X_num.shape[1]
    rows = math.ceil(n_features / cols_per_row)

    fig, axes = plt.subplots(rows, cols_per_row, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(X_num.columns):
        data = X_num[col].dropna()

        # Optional tail clipping (visual only)
        if clip_quantile > 0:
            lo = data.quantile(clip_quantile)
            hi = data.quantile(1 - clip_quantile)
            data = data.clip(lo, hi)

        axes[i].hist(data, bins=bins, density=True, alpha=0.7)
        axes[i].set_title(col, fontsize=9)
        axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_confidence_tradeoff(
    df,
    preds,
    backtest_fn,
    thresholds=np.linspace(0.4, 1.6, 9),
    confidence_window=200,
    take_profit=0.02,
    stop_loss=0.005
):
    """
    Plot Coverage vs Performance trade-off for confidence threshold tuning.

    Parameters
    ----------
    df : pd.DataFrame
        Market data with 'close'
    preds : array-like
        Model predictions
    backtest_fn : function
        backtest_fixed_profit_with_confidence
    thresholds : iterable
        Confidence thresholds to sweep
    """

    coverages = []
    sharpes = []
    avg_profits = []
    trades = []

    for thr in thresholds:
        results = backtest_fn(
            df=df,
            preds=preds,
            confidence_threshold=thr,
            confidence_window=confidence_window,
            take_profit=take_profit,
            stop_loss=stop_loss
        )

        coverages.append(results["Market Coverage"])
        sharpes.append(results["Sharpe Ratio"])
        avg_profits.append(results["Avg Profit / Trade"])
        trades.append(results["Trades"])

    # ===============================
    # Plot
    # ===============================
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(coverages, sharpes, marker='o', label="Sharpe Ratio")
    ax1.set_xlabel("Market Coverage")
    ax1.set_ylabel("Sharpe Ratio")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(coverages, avg_profits, marker='s', color='orange', label="Avg Profit / Trade")
    ax2.set_ylabel("Avg Profit / Trade")

    # Annotate thresholds
    for i, thr in enumerate(thresholds):
        ax1.annotate(
            f"{thr:.2f}",
            (coverages[i], sharpes[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9
        )

    # Legend handling
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Confidence Threshold Trade-off\n(coverage vs performance)")
    plt.tight_layout()
    plt.show()




def plot_feature_correlation_heatmap(
    df,
    feature_cols,
    method="spearman",
    min_abs_corr=0.0,
    figsize=(12, 10)
):
    """
    Plot correlation heatmap of available feature columns only.
    Automatically filters missing columns.
    """

    # ===============================
    # 1️⃣ Keep only valid feature columns
    # ===============================
    available_cols = [c for c in feature_cols if c in df.columns]

    missing = set(feature_cols) - set(available_cols)
    if missing:
        print(f"[WARN] Dropping {len(missing)} missing features:")
        print(sorted(missing))

    if len(available_cols) < 2:
        raise ValueError("Not enough valid features to compute correlation.")

    # ===============================
    # 2️⃣ Compute correlation
    # ===============================
    corr = df[available_cols].corr(method=method)

    # ===============================
    # 3️⃣ Mask weak correlations
    # ===============================
    mask = None
    if min_abs_corr > 0:
        mask = corr.abs() < min_abs_corr

    # ===============================
    # 4️⃣ Plot heatmap
    # ===============================
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75}
    )

    plt.title(f"Feature Correlation Heatmap ({method.capitalize()})")
    plt.tight_layout()
    plt.show()



def plot_trade_points_interactive(
    df,
    buy_points,
    sell_points,
    price_col="close",
    title="Interactive Trade Visualization"
):
    """
    Interactive plot of price with BUY / SELL / TP / SL points.

    Parameters
    ----------
    df : pd.DataFrame
        Price data (index must be datetime or ordered)
    buy_points : list of tuples
        (timestamp, price, 'BUY' or 'SELL')
    sell_points : list of tuples
        (timestamp, price, 'TP' or 'SL')
    price_col : str
        Column name of price
    title : str
        Plot title
    """

    fig = go.Figure()

    # ===============================
    # 1️⃣ Price line
    # ===============================
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[price_col],
            mode="lines",
            name="Price",
            line=dict(color="black", width=1),
            hovertemplate="Time: %{x}<br>Price: %{y:.2f}<extra></extra>",
        )
    )

    # ===============================
    # 2️⃣ BUY / SELL entries
    # ===============================
    if buy_points:
        buy_df = pd.DataFrame(
            buy_points, columns=["timestamp", "price", "action"]
        )

        long_df = buy_df[buy_df["action"] == "BUY"]
        short_df = buy_df[buy_df["action"] == "SELL"]

        if not long_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=long_df["timestamp"],
                    y=long_df["price"],
                    mode="markers",
                    name="BUY (Long)",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color="green",
                        line=dict(width=1, color="darkgreen"),
                    ),
                    hovertemplate="BUY<br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>",
                )
            )

        if not short_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=short_df["timestamp"],
                    y=short_df["price"],
                    mode="markers",
                    name="SELL (Short)",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color="orange",
                        line=dict(width=1, color="darkorange"),
                    ),
                    hovertemplate="SELL<br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>",
                )
            )

    # ===============================
    # 3️⃣ TP / SL exits
    # ===============================
    if sell_points:
        sell_df = pd.DataFrame(
            sell_points, columns=["timestamp", "price", "exit"]
        )

        tp_df = sell_df[sell_df["exit"] == "TP"]
        sl_df = sell_df[sell_df["exit"] == "SL"]

        if not tp_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=tp_df["timestamp"],
                    y=tp_df["price"],
                    mode="markers",
                    name="Take Profit",
                    marker=dict(
                        symbol="circle",
                        size=9,
                        color="blue",
                        line=dict(width=1, color="darkblue"),
                    ),
                    hovertemplate="TP<br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>",
                )
            )

        if not sl_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=sl_df["timestamp"],
                    y=sl_df["price"],
                    mode="markers",
                    name="Stop Loss",
                    marker=dict(
                        symbol="x",
                        size=10,
                        color="red",
                    ),
                    hovertemplate="SL<br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>",
                )
            )

    # ===============================
    # 4️⃣ Layout / UX
    # ===============================
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        template="plotly_white",
        height=600,
    )

    fig.show()