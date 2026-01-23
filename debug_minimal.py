"""
MINIMAL DEBUG - Find if ANY predictive signal exists in crypto data
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

print("="*70)
print("CRYPTO PREDICTION DEBUG - Finding if signal exists")
print("="*70)

# ============================================================
# LOAD REAL DATA
# ============================================================

# Find largest dataset
data_files = list(Path('C:/crypto/ver7/dataset').rglob('*.csv'))
data_files = [f for f in data_files if '50000' in str(f) or '47528' in str(f)]

if not data_files:
    print("No large data files found!")
    exit()

# Use BTCUSDT (most liquid)
btc_files = [f for f in data_files if 'BTC' in str(f)]
data_file = btc_files[0] if btc_files else data_files[0]

print(f"\nLoading: {data_file}")
df = pd.read_csv(data_file)
print(f"Rows: {len(df)}, Columns: {list(df.columns)}")

# ============================================================
# CREATE SIMPLE FEATURES
# ============================================================

print("\n" + "="*70)
print("STEP 1: Create features and check for predictive signal")
print("="*70)

# Basic returns at different horizons
df['ret_1h'] = df['close'].pct_change(1)
df['ret_2h'] = df['close'].pct_change(2)
df['ret_4h'] = df['close'].pct_change(4)
df['ret_8h'] = df['close'].pct_change(8)
df['ret_12h'] = df['close'].pct_change(12)
df['ret_24h'] = df['close'].pct_change(24)

# Volatility
df['vol_24h'] = df['ret_1h'].rolling(24).std()

# Simple momentum
df['mom_4h'] = df['close'] / df['close'].shift(4) - 1
df['mom_12h'] = df['close'] / df['close'].shift(12) - 1

# Volume features
if 'volume' in df.columns:
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(24).mean()
    df['vol_change'] = df['volume'].pct_change()

# Create targets at different horizons
for horizon in [1, 2, 4, 8, 12, 24]:
    df[f'future_{horizon}h'] = df['close'].shift(-horizon) / df['close'] - 1
    df[f'dir_{horizon}h'] = (df[f'future_{horizon}h'] > 0).astype(int)

df = df.dropna()
print(f"After cleaning: {len(df)} rows")

# ============================================================
# CHECK CORRELATIONS AT DIFFERENT HORIZONS
# ============================================================

print("\n" + "-"*70)
print("Feature correlations with FUTURE DIRECTION at different horizons:")
print("-"*70)

features = ['ret_1h', 'ret_4h', 'ret_12h', 'ret_24h', 'vol_24h', 'mom_4h', 'mom_12h']
if 'vol_ratio' in df.columns:
    features += ['vol_ratio']

horizons = [1, 4, 12, 24]

print(f"\n{'Feature':<15}", end='')
for h in horizons:
    print(f"  {h}h ahead", end='')
print()
print("-" * 55)

best_corr = 0
best_feature = None
best_horizon = None

for feat in features:
    print(f"{feat:<15}", end='')
    for h in horizons:
        corr = df[feat].corr(df[f'dir_{h}h'])
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_feature = feat
            best_horizon = h
        marker = "*" if abs(corr) > 0.02 else " "
        print(f"  {corr:+.4f}{marker}", end='')
    print()

print("-" * 55)
print(f"\nBest signal: {best_feature} -> {best_horizon}h direction (corr={best_corr:.4f})")

# ============================================================
# CHECK IF THERE'S ANY PERSISTENCE/MOMENTUM
# ============================================================

print("\n" + "-"*70)
print("Return autocorrelation (momentum check):")
print("-"*70)

for lag in [1, 2, 4, 8, 12, 24]:
    autocorr = df['ret_1h'].autocorr(lag)
    marker = "*" if abs(autocorr) > 0.02 else " "
    print(f"  Lag {lag:2d}h: {autocorr:+.4f} {marker}")

# ============================================================
# TEST: CAN WE PREDICT ANYTHING?
# ============================================================

print("\n" + "="*70)
print("STEP 2: Train minimal model on best signal")
print("="*70)

# Use best horizon
target_col = f'dir_{best_horizon}h'
print(f"Target: {target_col} (predicting {best_horizon}h ahead direction)")
print(f"Target distribution: {df[target_col].mean():.1%} up")

# Simple features
feature_cols = ['ret_1h', 'ret_4h', 'ret_12h', 'mom_4h', 'vol_24h']
if 'vol_ratio' in df.columns:
    feature_cols.append('vol_ratio')

print(f"Features: {feature_cols}")

# Prepare data
X = df[feature_cols].values.astype(np.float32)
y = df[target_col].values.astype(np.float32)

# Simple chronological split
split1 = int(len(X) * 0.7)
split2 = int(len(X) * 0.85)

X_train, X_val, X_test = X[:split1], X[split1:split2], X[split2:]
y_train, y_val, y_test = y[:split1], y[split1:split2], y[split2:]

# Normalize
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Create sequences
seq_len = 12  # Short sequence for simplicity

def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys).reshape(-1, 1)

X_train_seq, y_train_seq = make_sequences(X_train, y_train, seq_len)
X_val_seq, y_val_seq = make_sequences(X_val, y_val, seq_len)
X_test_seq, y_test_seq = make_sequences(X_test, y_test, seq_len)

print(f"Sequences: Train={len(X_train_seq)}, Val={len(X_val_seq)}, Test={len(X_test_seq)}")

# Convert to tensors
X_train_t = torch.FloatTensor(X_train_seq)
y_train_t = torch.FloatTensor(y_train_seq)
X_val_t = torch.FloatTensor(X_val_seq)
y_val_t = torch.FloatTensor(y_val_seq)
X_test_t = torch.FloatTensor(X_test_seq)
y_test_t = torch.FloatTensor(y_test_seq)

# ============================================================
# SIMPLE LSTM MODEL
# ============================================================

class MinimalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = MinimalLSTM(input_size=len(feature_cols), hidden_size=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

print("\nTraining LSTM...")
best_val_acc = 0.5

for epoch in range(100):
    # Train
    model.train()
    optimizer.zero_grad()
    pred = model(X_train_t)
    loss = criterion(pred, y_train_t)
    loss.backward()
    optimizer.step()

    # Validate
    model.eval()
    with torch.no_grad():
        val_pred = torch.sigmoid(model(X_val_t))
        val_acc = ((val_pred > 0.5).float() == y_val_t).float().mean().item()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pt')

    if epoch % 20 == 0:
        train_acc = ((torch.sigmoid(pred) > 0.5).float() == y_train_t).float().mean().item()
        print(f"Epoch {epoch:3d}: Train={train_acc:.1%}, Val={val_acc:.1%}")

# Load best model and test
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

with torch.no_grad():
    test_pred = torch.sigmoid(model(X_test_t))
    test_acc = ((test_pred > 0.5).float() == y_test_t).float().mean().item()

    # IC
    pred_np = test_pred.numpy().flatten()
    actual_np = y_test_t.numpy().flatten()
    test_ic = np.corrcoef(pred_np, actual_np)[0, 1] if len(pred_np) > 2 else 0

# Baseline
baseline = max(y_test.mean(), 1 - y_test.mean())

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Test Accuracy:     {test_acc:.1%}")
print(f"Test IC:           {test_ic:.4f}")
print(f"Baseline (random): {baseline:.1%}")
print(f"Edge over random:  {(test_acc - 0.5)*100:+.2f}%")
print(f"Edge over baseline:{(test_acc - baseline)*100:+.2f}%")

# ============================================================
# ALSO TEST: Simple logistic regression (sanity check)
# ============================================================

print("\n" + "-"*70)
print("COMPARISON: Logistic Regression (no sequence)")
print("-"*70)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Just use last timestep features
X_train_flat = X_train_seq[:, -1, :]
X_val_flat = X_val_seq[:, -1, :]
X_test_flat = X_test_seq[:, -1, :]

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_flat, y_train_seq.flatten())

lr_train_acc = accuracy_score(y_train_seq.flatten(), lr.predict(X_train_flat))
lr_val_acc = accuracy_score(y_val_seq.flatten(), lr.predict(X_val_flat))
lr_test_acc = accuracy_score(y_test_seq.flatten(), lr.predict(X_test_flat))

print(f"LogReg Train: {lr_train_acc:.1%}, Val: {lr_val_acc:.1%}, Test: {lr_test_acc:.1%}")

# Feature importance from LogReg
print("\nLogReg feature importance:")
for feat, coef in sorted(zip(feature_cols, lr.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {feat:<12}: {coef:+.4f}")

# ============================================================
# CONCLUSION
# ============================================================

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if test_acc > 0.52 and test_ic > 0.02:
    print("[OK] There IS some predictive signal in this data!")
    print(f"     Best model: {'LSTM' if test_acc > lr_test_acc else 'LogReg'}")
    print(f"     Best horizon: {best_horizon}h")
elif test_acc > 0.51:
    print("[WEAK] Very weak signal exists but barely tradeable")
    print("       Consider: longer horizons, more features, or accept market efficiency")
else:
    print("[NONE] No significant predictive signal found")
    print("       The market appears efficient at this timeframe")
    print("       Options:")
    print("       1. Try different timeframe (daily instead of hourly)")
    print("       2. Add external features (sentiment, on-chain, macro)")
    print("       3. Focus on risk management instead of prediction")
