"""
Simple baseline test - can we predict ANYTHING from this data?
Tests: random baseline, always-up, always-down, simple logistic regression.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.ml.preprocessing.preprocessor import CryptoPreprocessor
from src.ml.models.lstm_model import get_lstm_feature_columns

print("="*70)
print("BASELINE PREDICTION TEST")
print("="*70)

# Load data
prep = CryptoPreprocessor("dataset")
train_df, val_df, test_df = prep.process_all(["BTCUSDT", "ETHUSDT"])

# Get features (same as LSTM uses)
feature_cols = get_lstm_feature_columns(train_df)
print(f"\nUsing {len(feature_cols)} features")

X_train = train_df[feature_cols].values
X_val = val_df[feature_cols].values
y_train_raw = train_df['target_return'].values
y_val_raw = val_df['target_return'].values

# Create class labels (same as LSTM)
THRESHOLD = 0.3
y_train = np.ones(len(y_train_raw), dtype=int)  # NEUTRAL
y_train[y_train_raw < -THRESHOLD] = 0  # DOWN
y_train[y_train_raw > THRESHOLD] = 2   # UP

y_val = np.ones(len(y_val_raw), dtype=int)
y_val[y_val_raw < -THRESHOLD] = 0
y_val[y_val_raw > THRESHOLD] = 2

print(f"\nClass distribution:")
print(f"  Train: DOWN={np.mean(y_train==0)*100:.1f}%, NEUTRAL={np.mean(y_train==1)*100:.1f}%, UP={np.mean(y_train==2)*100:.1f}%")
print(f"  Val:   DOWN={np.mean(y_val==0)*100:.1f}%, NEUTRAL={np.mean(y_val==1)*100:.1f}%, UP={np.mean(y_val==2)*100:.1f}%")

# Test 1: Random baseline
print("\n" + "-"*50)
print("TEST 1: Random Baseline")
random_pred = np.random.randint(0, 3, len(y_val))
random_acc = accuracy_score(y_val, random_pred)
print(f"  Random accuracy: {random_acc*100:.1f}%")

# Test 2: Always predict most common class
print("\n" + "-"*50)
print("TEST 2: Always Predict Most Common Class")
most_common = np.bincount(y_train).argmax()
always_same = np.full(len(y_val), most_common)
always_acc = accuracy_score(y_val, always_same)
print(f"  Most common class: {['DOWN', 'NEUTRAL', 'UP'][most_common]}")
print(f"  Accuracy: {always_acc*100:.1f}%")

# Test 3: Simple Logistic Regression
print("\n" + "-"*50)
print("TEST 3: Logistic Regression (simple linear model)")
try:
    lr = LogisticRegression(max_iter=1000, C=0.01)  # Strong regularization
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_val)
    lr_acc = accuracy_score(y_val, lr_pred)
    print(f"  Logistic Regression accuracy: {lr_acc*100:.1f}%")

    # Per-class
    for i, name in enumerate(['DOWN', 'NEUTRAL', 'UP']):
        mask = y_val == i
        if mask.sum() > 0:
            class_acc = (lr_pred[mask] == i).mean()
            print(f"    {name}: {class_acc*100:.1f}%")
except Exception as e:
    print(f"  Error: {e}")

# Test 4: Predict based on recent return
print("\n" + "-"*50)
print("TEST 4: Momentum (predict same as recent direction)")
# Use return_1h as predictor
if 'return_1h' in val_df.columns:
    recent_return = val_df['return_1h'].values
    momentum_pred = np.ones(len(y_val), dtype=int)
    momentum_pred[recent_return < -THRESHOLD] = 0
    momentum_pred[recent_return > THRESHOLD] = 2
    momentum_acc = accuracy_score(y_val, momentum_pred)
    print(f"  Momentum accuracy: {momentum_acc*100:.1f}%")

# Test 5: Mean Reversion (predict opposite of recent)
print("\n" + "-"*50)
print("TEST 5: Mean Reversion (predict opposite of recent)")
if 'return_1h' in val_df.columns:
    reversion_pred = np.ones(len(y_val), dtype=int)
    reversion_pred[recent_return < -THRESHOLD] = 2  # If down, predict up
    reversion_pred[recent_return > THRESHOLD] = 0   # If up, predict down
    reversion_acc = accuracy_score(y_val, reversion_pred)
    print(f"  Mean Reversion accuracy: {reversion_acc*100:.1f}%")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print(f"""
If Logistic Regression accuracy is close to random (~33%):
  -> Features have NO predictive power for this target
  -> LSTM will also fail (it can't do magic)
  -> Need different features or different target

If Logistic Regression is better than random:
  -> There IS some signal in the features
  -> LSTM might be overfitting
  -> Need stronger regularization or simpler model
""")
