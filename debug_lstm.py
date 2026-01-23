"""
MINIMAL LSTM DEBUG SCRIPT
Run this to verify the model can actually learn something.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ============================================================
# STEP 1: MINIMAL LSTM MODEL
# ============================================================

class MinimalLSTM(nn.Module):
    """Absolute minimum LSTM for binary direction prediction."""

    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # Binary output

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]  # Take last timestep
        return self.fc(last)


class SimpleDataset(Dataset):
    """Minimal dataset - just sequences and binary targets."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int = 24):
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.FloatTensor(x), torch.FloatTensor([y])


# ============================================================
# STEP 2: TEST WITH SYNTHETIC DATA FIRST
# ============================================================

def test_synthetic():
    """Test if model can learn a simple pattern."""
    print("\n" + "="*60)
    print("TEST 1: SYNTHETIC DATA (Model should get ~90%+ accuracy)")
    print("="*60)

    # Create synthetic data with clear pattern
    np.random.seed(42)
    n_samples = 5000

    # Feature: momentum (if positive, price goes up)
    momentum = np.random.randn(n_samples) * 0.02
    noise = np.random.randn(n_samples) * 0.005

    # Target: 1 if momentum > 0, else 0 (with some noise)
    target = (momentum + noise > 0).astype(np.float32)

    # Add some lag features
    features = np.column_stack([
        momentum,
        np.roll(momentum, 1),
        np.roll(momentum, 2),
        np.roll(momentum, 3),
    ])
    features[0:4] = 0  # Fix rolled values

    print(f"Features shape: {features.shape}")
    print(f"Target distribution: {target.mean():.1%} up, {1-target.mean():.1%} down")

    # Split
    split = int(len(features) * 0.8)
    train_ds = SimpleDataset(features[:split], target[:split], seq_len=12)
    val_ds = SimpleDataset(features[split:], target[split:], seq_len=12)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    # Model
    model = MinimalLSTM(input_size=4, hidden_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # Train
    print("\nTraining...")
    for epoch in range(10):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                predicted = (torch.sigmoid(pred) > 0.5).float()
                correct += (predicted == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val Acc={acc:.1%}")

    if acc > 0.7:
        print("\n[OK] Model CAN learn patterns!")
        return True
    else:
        print("\n[FAIL] Model cannot learn even simple patterns")
        return False


# ============================================================
# STEP 3: TEST WITH REAL DATA
# ============================================================

def test_real_data():
    """Test with actual crypto data."""
    print("\n" + "="*60)
    print("TEST 2: REAL CRYPTO DATA")
    print("="*60)

    # Load data
    from src.core.config import AppConfig
    dataset_dir = Path(AppConfig.DATASET_DIR)

    # Find a dataset file
    csv_files = list(dataset_dir.glob("*USDT*.csv"))
    if not csv_files:
        print("[FAIL] No dataset files found in", dataset_dir)
        return False

    csv_file = csv_files[0]
    print(f"Loading: {csv_file.name}")

    df = pd.read_csv(csv_file)
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Check for required columns
    if 'close' not in df.columns:
        print("[FAIL] No 'close' column")
        return False

    # ============================================================
    # MINIMAL FEATURES - Just returns and simple momentum
    # ============================================================

    df['return_1h'] = df['close'].pct_change()
    df['return_4h'] = df['close'].pct_change(4)
    df['return_12h'] = df['close'].pct_change(12)
    df['return_24h'] = df['close'].pct_change(24)

    # Volatility
    df['volatility'] = df['return_1h'].rolling(24).std()

    # Simple momentum
    df['momentum'] = df['close'] / df['close'].shift(12) - 1

    # Target: Will price go UP in next 4 hours?
    df['future_return'] = df['close'].shift(-4) / df['close'] - 1
    df['target'] = (df['future_return'] > 0).astype(float)

    # Drop NaN
    df = df.dropna()
    print(f"After cleaning: {len(df)} rows")

    # Features
    feature_cols = ['return_1h', 'return_4h', 'return_12h', 'return_24h', 'volatility', 'momentum']

    # Check for NaN/Inf
    features = df[feature_cols].values
    targets = df['target'].values

    print(f"\nFeature stats:")
    for i, col in enumerate(feature_cols):
        vals = features[:, i]
        print(f"  {col}: mean={np.nanmean(vals):.6f}, std={np.nanstd(vals):.6f}, "
              f"nan={np.isnan(vals).sum()}, inf={np.isinf(vals).sum()}")

    print(f"\nTarget distribution: {targets.mean():.1%} up, {1-targets.mean():.1%} down")

    # Normalize features (critical!)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # Split BEFORE scaling to prevent leakage
    split = int(len(features) * 0.7)
    val_split = int(len(features) * 0.85)

    train_features = features[:split]
    val_features = features[split:val_split]
    test_features = features[val_split:]

    train_targets = targets[:split]
    val_targets = targets[split:val_split]
    test_targets = targets[val_split:]

    # Fit scaler on train only
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    print(f"\nSplit: Train={len(train_features)}, Val={len(val_features)}, Test={len(test_features)}")

    # Check scaled values
    print(f"Scaled train mean: {train_features.mean():.4f}, std: {train_features.std():.4f}")

    # Create datasets
    seq_len = 24  # 24 hours of context
    train_ds = SimpleDataset(train_features, train_targets, seq_len=seq_len)
    val_ds = SimpleDataset(val_features, val_targets, seq_len=seq_len)
    test_ds = SimpleDataset(test_features, test_targets, seq_len=seq_len)

    print(f"Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    # Model
    model = MinimalLSTM(input_size=len(feature_cols), hidden_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    print("\nTraining on real data...")
    best_val_acc = 0

    for epoch in range(30):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            predicted = (torch.sigmoid(pred) > 0.5).float()
            train_correct += (predicted == y).sum().item()
            train_total += y.size(0)

        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_preds = []
        val_actuals = []

        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                prob = torch.sigmoid(pred)
                predicted = (prob > 0.5).float()
                val_correct += (predicted == y).sum().item()
                val_total += y.size(0)
                val_preds.extend(prob.numpy().flatten())
                val_actuals.extend(y.numpy().flatten())

        val_acc = val_correct / val_total

        # Calculate IC (correlation between predictions and actuals)
        val_preds = np.array(val_preds)
        val_actuals = np.array(val_actuals)
        ic = np.corrcoef(val_preds, val_actuals)[0, 1] if len(val_preds) > 2 else 0

        if epoch % 5 == 0 or val_acc > best_val_acc:
            print(f"Epoch {epoch+1:2d}: Train Acc={train_acc:.1%}, Val Acc={val_acc:.1%}, IC={ic:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    # Final test
    print("\n" + "-"*40)
    print("FINAL TEST SET EVALUATION")
    print("-"*40)

    model.eval()
    test_preds = []
    test_actuals = []

    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x)
            prob = torch.sigmoid(pred)
            test_preds.extend(prob.numpy().flatten())
            test_actuals.extend(y.numpy().flatten())

    test_preds = np.array(test_preds)
    test_actuals = np.array(test_actuals)

    test_acc = ((test_preds > 0.5) == test_actuals).mean()
    test_ic = np.corrcoef(test_preds, test_actuals)[0, 1]

    # Directional accuracy for confident predictions
    confident_mask = (test_preds > 0.6) | (test_preds < 0.4)
    if confident_mask.sum() > 0:
        confident_acc = ((test_preds[confident_mask] > 0.5) == test_actuals[confident_mask]).mean()
        confident_pct = confident_mask.mean()
    else:
        confident_acc = 0
        confident_pct = 0

    print(f"Test Accuracy:     {test_acc:.1%}")
    print(f"Test IC:           {test_ic:.4f}")
    print(f"Confident Acc:     {confident_acc:.1%} (on {confident_pct:.1%} of predictions)")

    # Baseline: always predict majority class
    baseline = max(test_actuals.mean(), 1 - test_actuals.mean())
    print(f"Baseline (random): {baseline:.1%}")

    if test_acc > baseline + 0.02 and test_ic > 0.02:
        print("\n[OK] Model shows some predictive power!")
        return True
    else:
        print("\n[WARNING] Model barely beats baseline")
        return False


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("LSTM Debug Script")
    print("="*60)

    # Test 1: Can the model learn at all?
    synthetic_ok = test_synthetic()

    if synthetic_ok:
        # Test 2: Does it work on real data?
        real_ok = test_real_data()

        if real_ok:
            print("\n" + "="*60)
            print("SUCCESS: Model architecture is working")
            print("Next: Tune hyperparameters and add more features")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("ISSUE: Model works on synthetic but not real data")
            print("Possible causes:")
            print("  1. Features don't have predictive signal")
            print("  2. Market is too noisy/random")
            print("  3. Need different features or longer horizon")
            print("="*60)
    else:
        print("\n" + "="*60)
        print("CRITICAL: Model cannot learn basic patterns")
        print("Check: PyTorch installation, CUDA, basic setup")
        print("="*60)
