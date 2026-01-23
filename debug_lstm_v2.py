"""
ULTRA MINIMAL LSTM DEBUG - Find why model isn't learning
"""

import numpy as np
import torch
import torch.nn as nn

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# ============================================================
# TEST 1: Can a simple linear layer learn XOR-like pattern?
# ============================================================

print("\n" + "="*60)
print("TEST 1: Simple MLP on easy pattern")
print("="*60)

# Create simple pattern: if sum of features > 0, target = 1
np.random.seed(42)
X = np.random.randn(1000, 4).astype(np.float32)
y = (X.sum(axis=1) > 0).astype(np.float32).reshape(-1, 1)

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"y distribution: {y.mean():.1%} ones")

X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Simple MLP
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(20):
    optimizer.zero_grad()
    pred = model(X_tensor)
    loss = criterion(pred, y_tensor)
    loss.backward()
    optimizer.step()

    acc = ((torch.sigmoid(pred) > 0.5).float() == y_tensor).float().mean()
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc.item():.1%}")

final_acc = ((torch.sigmoid(model(X_tensor)) > 0.5).float() == y_tensor).float().mean()
print(f"Final accuracy: {final_acc.item():.1%}")

if final_acc > 0.9:
    print("[OK] MLP works!")
else:
    print("[FAIL] Even MLP doesn't work - check PyTorch")

# ============================================================
# TEST 2: Simple LSTM on sequence pattern
# ============================================================

print("\n" + "="*60)
print("TEST 2: Simple LSTM on sequence")
print("="*60)

# Create sequences where target = 1 if last value is positive
seq_len = 10
n_samples = 1000

X_seq = np.random.randn(n_samples, seq_len, 1).astype(np.float32)
y_seq = (X_seq[:, -1, 0] > 0).astype(np.float32).reshape(-1, 1)

print(f"X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape}")

X_seq_tensor = torch.FloatTensor(X_seq)
y_seq_tensor = torch.FloatTensor(y_seq)

# Simple LSTM
class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 16, batch_first=True)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

lstm_model = SimpleLSTM()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.01)

for epoch in range(30):
    optimizer.zero_grad()
    pred = lstm_model(X_seq_tensor)
    loss = criterion(pred, y_seq_tensor)
    loss.backward()
    optimizer.step()

    acc = ((torch.sigmoid(pred) > 0.5).float() == y_seq_tensor).float().mean()
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc.item():.1%}")

final_acc = ((torch.sigmoid(lstm_model(X_seq_tensor)) > 0.5).float() == y_seq_tensor).float().mean()
print(f"Final LSTM accuracy: {final_acc.item():.1%}")

if final_acc > 0.9:
    print("[OK] LSTM works!")
else:
    print("[FAIL] LSTM not learning")

# ============================================================
# TEST 3: Bidirectional LSTM (what we're using)
# ============================================================

print("\n" + "="*60)
print("TEST 3: Bidirectional LSTM")
print("="*60)

class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 16, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(32, 1)  # 16*2 for bidirectional

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

bilstm_model = BiLSTM()
optimizer = torch.optim.Adam(bilstm_model.parameters(), lr=0.01)

for epoch in range(30):
    optimizer.zero_grad()
    pred = bilstm_model(X_seq_tensor)
    loss = criterion(pred, y_seq_tensor)
    loss.backward()
    optimizer.step()

    acc = ((torch.sigmoid(pred) > 0.5).float() == y_seq_tensor).float().mean()
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc.item():.1%}")

final_acc = ((torch.sigmoid(bilstm_model(X_seq_tensor)) > 0.5).float() == y_seq_tensor).float().mean()
print(f"Final BiLSTM accuracy: {final_acc.item():.1%}")

if final_acc > 0.9:
    print("[OK] BiLSTM works!")
else:
    print("[FAIL] BiLSTM not learning")

# ============================================================
# TEST 4: Multi-feature LSTM (like our real use case)
# ============================================================

print("\n" + "="*60)
print("TEST 4: Multi-feature LSTM with momentum pattern")
print("="*60)

# Create pattern: momentum predicts direction
n_samples = 2000
seq_len = 24
n_features = 4

# Generate "momentum" feature that predicts target
momentum = np.random.randn(n_samples + seq_len) * 0.02
noise1 = np.random.randn(n_samples + seq_len) * 0.01
noise2 = np.random.randn(n_samples + seq_len) * 0.01
noise3 = np.random.randn(n_samples + seq_len) * 0.01

# Stack features
all_features = np.column_stack([momentum, noise1, noise2, noise3]).astype(np.float32)

# Target: based on future momentum (which correlates with current)
targets = (momentum[seq_len:] > 0).astype(np.float32)

# Create sequences
X_multi = np.array([all_features[i:i+seq_len] for i in range(n_samples)])
y_multi = targets.reshape(-1, 1)

print(f"X_multi shape: {X_multi.shape}, y_multi shape: {y_multi.shape}")
print(f"Target distribution: {y_multi.mean():.1%} ones")

# Split
split = int(n_samples * 0.8)
X_train, X_val = X_multi[:split], X_multi[split:]
y_train, y_val = y_multi[:split], y_multi[split:]

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.FloatTensor(y_val)

# Model
class MultiFeatureLSTM(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 32, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

multi_model = MultiFeatureLSTM(n_features)
optimizer = torch.optim.Adam(multi_model.parameters(), lr=0.005)

print("\nTraining multi-feature LSTM...")
for epoch in range(50):
    # Train
    multi_model.train()
    optimizer.zero_grad()
    pred = multi_model(X_train_t)
    loss = criterion(pred, y_train_t)
    loss.backward()
    optimizer.step()

    # Validate
    multi_model.eval()
    with torch.no_grad():
        val_pred = multi_model(X_val_t)
        val_loss = criterion(val_pred, y_val_t)
        val_acc = ((torch.sigmoid(val_pred) > 0.5).float() == y_val_t).float().mean()

    if epoch % 10 == 0:
        train_acc = ((torch.sigmoid(pred) > 0.5).float() == y_train_t).float().mean()
        print(f"Epoch {epoch}: Train Acc={train_acc.item():.1%}, Val Acc={val_acc.item():.1%}")

final_val_acc = val_acc.item()
print(f"\nFinal validation accuracy: {final_val_acc:.1%}")

if final_val_acc > 0.55:
    print("[OK] Multi-feature LSTM shows learning!")
else:
    print("[WARNING] Multi-feature LSTM struggling")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("If all tests pass, the model architecture is fine.")
print("If multi-feature test fails but others pass, the issue is")
print("that features don't have predictive signal for the target.")
