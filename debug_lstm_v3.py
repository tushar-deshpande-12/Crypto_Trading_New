"""
DEBUG V3: Find features with ACTUAL predictive signal
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# ============================================================
# TEST: Multi-feature LSTM with ACTUAL predictive signal
# ============================================================

print("="*60)
print("TEST: Features that ACTUALLY predict target")
print("="*60)

# Create pattern where CURRENT features predict FUTURE target
n_samples = 3000
seq_len = 24

# Momentum with autocorrelation (like real markets have some persistence)
np.random.seed(42)
momentum = np.zeros(n_samples + seq_len + 10)
for i in range(1, len(momentum)):
    # AR(1) process with 0.3 autocorrelation
    momentum[i] = 0.3 * momentum[i-1] + np.random.randn() * 0.02

# Other features derived from momentum
vol = pd.Series(momentum).rolling(12).std().fillna(0.01).values
trend = pd.Series(momentum).rolling(6).mean().fillna(0).values

# Target: future direction (4 steps ahead)
future_idx = 4
targets = (momentum[seq_len + future_idx:seq_len + future_idx + n_samples] > 0).astype(np.float32)

# Stack features
all_features = np.column_stack([
    momentum[:n_samples + seq_len],
    vol[:n_samples + seq_len],
    trend[:n_samples + seq_len],
]).astype(np.float32)

# Create sequences
X = np.array([all_features[i:i+seq_len] for i in range(n_samples)])
y = targets.reshape(-1, 1)

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Target: {y.mean():.1%} up")

# Check autocorrelation of momentum
autocorr = np.corrcoef(momentum[:-future_idx], momentum[future_idx:])[0,1]
print(f"Momentum autocorrelation (lag {future_idx}): {autocorr:.4f}")

# Split
split = int(n_samples * 0.7)
val_split = int(n_samples * 0.85)

X_train, X_val, X_test = X[:split], X[split:val_split], X[val_split:]
y_train, y_val, y_test = y[:split], y[split:val_split], y[val_split:]

# Normalize (fit on train only)
mean = X_train.mean(axis=(0,1), keepdims=True)
std = X_train.std(axis=(0,1), keepdims=True) + 1e-8
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# To tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.FloatTensor(y_val)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

# Model
class SimpleBiLSTM(nn.Module):
    def __init__(self, n_features, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = SimpleBiLSTM(n_features=3, hidden=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.BCEWithLogitsLoss()

print("\nTraining...")
best_val_acc = 0
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train_t)
    loss = criterion(pred, y_train_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_acc = ((torch.sigmoid(val_pred) > 0.5).float() == y_val_t).float().mean().item()

    if val_acc > best_val_acc:
        best_val_acc = val_acc

    if epoch % 20 == 0:
        train_acc = ((torch.sigmoid(pred) > 0.5).float() == y_train_t).float().mean().item()
        print(f"Epoch {epoch}: Train={train_acc:.1%}, Val={val_acc:.1%}")

# Test
model.eval()
with torch.no_grad():
    test_pred = torch.sigmoid(model(X_test_t))
    test_acc = ((test_pred > 0.5).float() == y_test_t).float().mean().item()
    test_ic = np.corrcoef(test_pred.numpy().flatten(), y_test_t.numpy().flatten())[0,1]

print(f"\nTEST: Acc={test_acc:.1%}, IC={test_ic:.4f}")

if test_acc > 0.55 and test_ic > 0.1:
    print("[OK] Model learns when features have signal!")
else:
    print("[FAIL] Still not learning")


# ============================================================
# NOW TEST ON REAL DATA - Find what features have signal
# ============================================================

print("\n" + "="*60)
print("REAL DATA: Testing feature predictiveness")
print("="*60)

from src.core.config import AppConfig
dataset_dir = Path(AppConfig.DATASET_DIR)
csv_files = list(dataset_dir.glob("*USDT*.csv"))

if not csv_files:
    print("No data files found")
    exit()

df = pd.read_csv(csv_files[0])
print(f"Loaded {csv_files[0].name}: {len(df)} rows")

# Create features and check their correlation with future returns
df['return_1h'] = df['close'].pct_change()
df['return_4h'] = df['close'].pct_change(4)
df['return_12h'] = df['close'].pct_change(12)
df['return_24h'] = df['close'].pct_change(24)

# Future return (what we want to predict)
df['future_4h'] = df['close'].shift(-4) / df['close'] - 1
df['future_direction'] = (df['future_4h'] > 0).astype(int)

# Technical indicators
df['rsi'] = 50  # Placeholder - calculate properly
df['volatility'] = df['return_1h'].rolling(24).std()
df['momentum_12h'] = df['close'] / df['close'].shift(12) - 1
df['momentum_24h'] = df['close'] / df['close'].shift(24) - 1

# Volume features (if available)
if 'volume' in df.columns:
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(24).mean()

df = df.dropna()

# Check correlations with future direction
print("\nFeature correlations with future 4h direction:")
print("-" * 50)

features_to_check = ['return_1h', 'return_4h', 'return_12h', 'return_24h',
                     'volatility', 'momentum_12h', 'momentum_24h']

if 'volume_change' in df.columns:
    features_to_check += ['volume_change', 'volume_ma_ratio']

correlations = {}
for feat in features_to_check:
    if feat in df.columns:
        corr = df[feat].corr(df['future_direction'])
        correlations[feat] = corr
        signal = "***" if abs(corr) > 0.02 else ""
        print(f"  {feat:20s}: {corr:+.4f} {signal}")

# Find best features
best_features = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
print(f"\nBest features by |correlation|:")
for feat, corr in best_features[:5]:
    print(f"  {feat}: {corr:+.4f}")

# Check if ANY feature has meaningful correlation
max_corr = max(abs(c) for c in correlations.values())
print(f"\nMax absolute correlation: {max_corr:.4f}")

if max_corr < 0.01:
    print("\n[WARNING] No feature has meaningful correlation with target!")
    print("This means the market is essentially unpredictable at 4h horizon.")
    print("Options:")
    print("  1. Try longer prediction horizon (24h, 48h)")
    print("  2. Try different features (order book, sentiment)")
    print("  3. Accept that crypto is highly efficient/random")
elif max_corr < 0.03:
    print("\n[WEAK] Some weak signal exists but very noisy")
    print("Model will struggle to beat baseline significantly")
else:
    print("\n[OK] Some predictive signal found")
    print("Model should be able to learn something")

# ============================================================
# TRAIN ON REAL DATA WITH BEST FEATURES
# ============================================================

print("\n" + "="*60)
print("TRAINING ON REAL DATA")
print("="*60)

# Use top features
top_features = [f for f, _ in best_features[:4]]
print(f"Using features: {top_features}")

# Prepare data
feature_data = df[top_features].values.astype(np.float32)
target_data = df['future_direction'].values.astype(np.float32)

# Create sequences
seq_len = 24
X_real = np.array([feature_data[i:i+seq_len] for i in range(len(feature_data) - seq_len)])
y_real = target_data[seq_len:].reshape(-1, 1)

print(f"X shape: {X_real.shape}, y shape: {y_real.shape}")
print(f"Target distribution: {y_real.mean():.1%} up")

# Split chronologically
split = int(len(X_real) * 0.7)
val_split = int(len(X_real) * 0.85)

X_train = X_real[:split]
X_val = X_real[split:val_split]
X_test = X_real[val_split:]
y_train = y_real[:split]
y_val = y_real[split:val_split]
y_test = y_real[val_split:]

# Normalize
mean = X_train.mean(axis=(0,1), keepdims=True)
std = X_train.std(axis=(0,1), keepdims=True) + 1e-8
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# Check for NaN/Inf
print(f"NaN in train: {np.isnan(X_train).sum()}")
print(f"Inf in train: {np.isinf(X_train).sum()}")

# To tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.FloatTensor(y_val)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

# Model
model = SimpleBiLSTM(n_features=len(top_features), hidden=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss()

print("\nTraining on real data...")
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train_t)
    loss = criterion(pred, y_train_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            train_acc = ((torch.sigmoid(pred) > 0.5).float() == y_train_t).float().mean().item()
            val_pred = torch.sigmoid(model(X_val_t))
            val_acc = ((val_pred > 0.5).float() == y_val_t).float().mean().item()
            val_ic = np.corrcoef(val_pred.numpy().flatten(), y_val_t.numpy().flatten())[0,1]
        print(f"Epoch {epoch}: Train={train_acc:.1%}, Val={val_acc:.1%}, IC={val_ic:.4f}")

# Final test
model.eval()
with torch.no_grad():
    test_pred = torch.sigmoid(model(X_test_t))
    test_acc = ((test_pred > 0.5).float() == y_test_t).float().mean().item()
    test_ic = np.corrcoef(test_pred.numpy().flatten(), y_test_t.numpy().flatten())[0,1]

baseline = max(y_test.mean(), 1 - y_test.mean())

print(f"\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Test Accuracy:  {test_acc:.1%}")
print(f"Test IC:        {test_ic:.4f}")
print(f"Baseline:       {baseline:.1%}")
print(f"Edge over base: {(test_acc - baseline)*100:.2f}%")

if test_acc > baseline + 0.01 and test_ic > 0.01:
    print("\n[OK] Model shows some edge!")
else:
    print("\n[RESULT] Model cannot beat baseline on this data")
    print("The market may be too efficient at this timeframe")
