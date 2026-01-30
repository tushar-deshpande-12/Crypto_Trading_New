from backtest_ranked import *
from rank_model import *
from regime_model import *
from triple_barrier_label import *
import pandas as pd
from features import *
from targets import compute_future_return
from visualization import *
from metrics import (
    compute_ice_safe,
    compute_sharpe,
    compute_trade_metrics,
    compute_market_coverage,
    compute_directional_accuracy
)
from visualization import plot_feature_correlation_heatmap
import torch


#DATA_PATH = r"C:\crypto\ver7\dataset\BTCUSDT\2026-01-11_11-07-31_50000candles\data.csv"
DATA_PATH = r"C:\crypto\ver7\dataset\ETHUSDT\2026-01-10_17-39-30_50000candles\data.csv"
SHOW_FEATURE_HIST = False
PLOT_CONFIDENCE_TRADEOFF = False
SHOW_CORR_HEATMAP = False
TAKE_PROFIT_PCT = 0.03    # +2%
STOP_LOSS_PCT   = 0.0075   # -0.5%
CONFIDENCE_THRESHOLD = 0.9
PLOT_BUY_SELL_POINTS = True

data = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])

def load_ohlcv(data):
    df = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df.set_index('timestamp', inplace=True)
    return df

df = load_ohlcv(data)

df = apply_triple_barrier(df)
df = compute_regime_features(df)
df = regime_filter(df)

X, feature_cols = generate_feature_matrix(df)
df, feature_cols_ohlcv = generate_ohlcv_features(df)

feature_cols += feature_cols_ohlcv

if SHOW_FEATURE_HIST:
    plot_feature_histograms(df)

if SHOW_CORR_HEATMAP:
    plot_feature_correlation_heatmap(df, feature_cols)

df = compute_future_return(df, horizon=30)
y = df['future_return']

X = X.select_dtypes(include=[np.number])
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# 🔥 4️⃣ ALIGN df to X (THIS IS THE KEY LINE)
df = df.loc[X.index]

# 5️⃣ Extract y ONLY AFTER alignment
y = df['future_return']

print("Len of X :", len(X), "len of y:", len(y))

X = rank_normalize(X)

# 8. Create model AFTER everything is frozen
model = RankNet(X.shape[1])

# # 9. Train
model = train_rank_model(model, X, y)

X_tensor = torch.tensor(X.values, dtype=torch.float32)
preds = model(X_tensor).detach().cpu().numpy()
# Backtest
equity, trade_returns = ranked_backtest_atr(
    df,
    preds,
    df['trade_regime'],
    top_k=3,
    initial_capital=10_000,
    risk_pct=0.01,
    stop_mult=1.5,
    tp_mult=3.0
)

trades_fixed, equity_fixed = backtest_fixed_profit(
    df,
    preds,
    df["trade_regime"],
    take_profit_pct=TAKE_PROFIT_PCT,
    stop_loss_pct=STOP_LOSS_PCT
)


# Metrics
ice = compute_ice_safe(preds, df['future_return'].values)
sharpe = compute_sharpe(trade_returns)
trade_stats = compute_trade_metrics(trade_returns)
coverage = compute_market_coverage(df['trade_regime'])
dir_acc = compute_directional_accuracy(preds, df['future_return'].values)

print("\n================= MODEL METRICS =================")
print(f"ICE (Spearman):        {ice['ic']:.4f}  (p={ice['p_value']:.2e})")
print(f"Directional Accuracy: {dir_acc*100:.2f}%")
print(f"Market Coverage:      {coverage*100:.2f}%")
print(f"Sharpe Ratio:         {sharpe:.2f}")

print("--------- ATR BASED POSITIONIN STATS ----------------")
print(f"Trades:               {trade_stats['num_trades']}")
print(f"Win Rate:             {trade_stats['win_rate']*100:.2f}%")
print(f"Avg Profit / Trade:   {trade_stats['avg_profit']:.4f}")
print(f"Avg Loss / Trade:     {trade_stats['avg_loss']:.4f}")


print("======== FIXED TRADE STATS =====================\n")

avg_profit = trades_fixed[trades_fixed.pnl_pct > 0].pnl_pct.mean()
avg_loss = trades_fixed[trades_fixed.pnl_pct < 0].pnl_pct.mean()
win_rate = (trades_fixed.pnl_pct > 0).mean()
coverage = len(trades_fixed) / len(df)

print(f"Trades: {len(trades_fixed)}")
print(f"Win Rate: {win_rate*100:.2f}%")
print(f"Avg Profit: {avg_profit*100:.2f}%")
print(f"Avg Loss: {avg_loss*100:.2f}%")
print(f"Market Coverage: {coverage*100:.2f}%")

print("======== VARIABLE CONFIDENCE THRESHOLD BACKTEST ========\n")

results = backtest_fixed_profit_with_confidence(
    df,
    preds,
    confidence_window=200,
    confidence_threshold= CONFIDENCE_THRESHOLD,
    take_profit=TAKE_PROFIT_PCT,    # +2%
    stop_loss=STOP_LOSS_PCT,      # -0.5%
            trading_fee_pct=0.001,
                position_fraction=0.1,   # 🔥 FIX
        tds_pct=0.01
)

print(f"Traders: {results['Trades']}")
print(f"Market Coverage: {results['Market Coverage']*100:.2f}%")
print(f"Win Rate: {results['Win Rate']*100:.2f}%")
print(f"Equity Generated: {results['Equity Generated']*100:.2f}%")
print(f"Sharpe Ratio: {results['Sharpe Ratio']:.2f}")
# print(f"Directional Accuracy: {results['Directional Accuracy']*100:.2f}%")
print(f"Initial Capital: ${results['Initial Capital']:.2f}")
print(f"Final Capital: ${results['Final Capital']:.2f}")

if PLOT_CONFIDENCE_TRADEOFF:
    plot_confidence_tradeoff(
        df=df,
        preds=preds,
        backtest_fn=backtest_fixed_profit_with_confidence,
        thresholds=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.1],
        trading_fee_pct=0.001,
        tds_pct=0.01
    )

if PLOT_BUY_SELL_POINTS:
    plot_trade_points_interactive(
    df,
    results["Buy Points"],
    results["Sell Points"],
    price_col="close",
    title="Trade Entry & Exit Points")