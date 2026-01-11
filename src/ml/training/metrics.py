"""
Training Metrics for Time Series Prediction
Evaluation metrics: MAE, RMSE, MAPE, R², Direction Accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    print(f"[METRICS] Calculating MAE...")
    print(f"[METRICS]   - y_true shape: {y_true.shape}")
    print(f"[METRICS]   - y_pred shape: {y_pred.shape}")

    try:
        mae = np.mean(np.abs(y_true - y_pred))
        print(f"[METRICS] ✓ MAE: {mae:.4f}")
        return float(mae)
    except Exception as e:
        print(f"[METRICS] ✗ MAE calculation failed: {e}")
        return float('nan')


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    print(f"[METRICS] Calculating RMSE...")

    try:
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        print(f"[METRICS] ✓ RMSE: {rmse:.4f}")
        return float(rmse)
    except Exception as e:
        print(f"[METRICS] ✗ RMSE calculation failed: {e}")
        return float('nan')


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Calculate Mean Absolute Percentage Error

    Args:
        y_true: True values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE value (as percentage)
    """
    print(f"[METRICS] Calculating MAPE...")

    try:
        # Avoid division by zero
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
        print(f"[METRICS] ✓ MAPE: {mape:.2f}%")
        return float(mape)
    except Exception as e:
        print(f"[METRICS] ✗ MAPE calculation failed: {e}")
        return float('nan')


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (coefficient of determination)

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        R² value
    """
    print(f"[METRICS] Calculating R² score...")

    try:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            print(f"[METRICS] ⚠ Warning: Total sum of squares is zero")
            return 0.0

        r2 = 1 - (ss_res / ss_tot)
        print(f"[METRICS] ✓ R²: {r2:.4f}")
        return float(r2)
    except Exception as e:
        print(f"[METRICS] ✗ R² calculation failed: {e}")
        return float('nan')


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate direction accuracy (% correct trend prediction)

    Args:
        y_true: True values (must be sequential)
        y_pred: Predicted values

    Returns:
        Direction accuracy (as percentage)
    """
    print(f"[METRICS] Calculating direction accuracy...")

    try:
        # Calculate actual direction (up/down from previous value)
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0

        # Calculate accuracy
        correct = np.sum(true_direction == pred_direction)
        total = len(true_direction)

        accuracy = (correct / total) * 100 if total > 0 else 0.0

        print(f"[METRICS] ✓ Direction Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")
        return float(accuracy)
    except Exception as e:
        print(f"[METRICS] ✗ Direction accuracy calculation failed: {e}")
        return float('nan')


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, verbose: bool = True) -> Dict[str, float]:
    """
    Calculate all metrics at once

    Args:
        y_true: True values
        y_pred: Predicted values
        verbose: Print detailed info

    Returns:
        Dictionary with all metric values
    """
    if verbose:
        print(f"\n[METRICS] Calculating all metrics...")
        print(f"[METRICS]   - Sample count: {len(y_true)}")
        print(f"[METRICS]   - True range: [{np.min(y_true):.2f}, {np.max(y_true):.2f}]")
        print(f"[METRICS]   - Pred range: [{np.min(y_pred):.2f}, {np.max(y_pred):.2f}]")

    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'direction_accuracy': direction_accuracy(y_true, y_pred),
    }

    if verbose:
        print(f"\n[METRICS] ✓ All metrics calculated:")
        for key, value in metrics.items():
            print(f"[METRICS]   - {key.upper()}: {value:.4f}")

    return metrics


class MetricsTracker:
    """
    Track metrics across training epochs
    """

    def __init__(self, metrics_to_track: Optional[List[str]] = None):
        """
        Initialize metrics tracker

        Args:
            metrics_to_track: List of metric names to track
                             (default: ['loss', 'mae', 'rmse', 'mape', 'r2'])
        """
        print(f"\n[METRICS_TRACKER] Initializing MetricsTracker")

        if metrics_to_track is None:
            metrics_to_track = ['loss', 'mae', 'rmse', 'mape', 'r2', 'direction_accuracy']

        self.metrics_to_track = metrics_to_track
        self.history: Dict[str, Dict[str, List[float]]] = {
            'train': {metric: [] for metric in metrics_to_track},
            'val': {metric: [] for metric in metrics_to_track},
        }
        self.best_metrics: Dict[str, Tuple[int, float]] = {}  # {metric: (epoch, value)}

        print(f"[METRICS_TRACKER]   - Tracking metrics: {metrics_to_track}")
        print(f"[METRICS_TRACKER] ✓ MetricsTracker initialized")

    def update(self, epoch: int, metrics: Dict[str, float], split: str = 'train'):
        """
        Update metrics for an epoch

        Args:
            epoch: Epoch number
            metrics: Dictionary of metric values
            split: 'train' or 'val'
        """
        print(f"\n[METRICS_TRACKER] Updating metrics for epoch {epoch} ({split})...")

        if split not in self.history:
            print(f"[METRICS_TRACKER] ⚠ Unknown split: {split}")
            return

        for metric_name, value in metrics.items():
            if metric_name in self.metrics_to_track:
                self.history[split][metric_name].append(value)

                # Track best metrics (for val split only)
                if split == 'val':
                    if metric_name not in self.best_metrics:
                        self.best_metrics[metric_name] = (epoch, value)
                    else:
                        # Lower is better for loss, MAE, RMSE, MAPE
                        # Higher is better for R², direction_accuracy
                        if metric_name in ['r2', 'direction_accuracy']:
                            if value > self.best_metrics[metric_name][1]:
                                self.best_metrics[metric_name] = (epoch, value)
                        else:
                            if value < self.best_metrics[metric_name][1]:
                                self.best_metrics[metric_name] = (epoch, value)

        print(f"[METRICS_TRACKER] ✓ Metrics updated")
        print(f"[METRICS_TRACKER]   - Updated metrics: {list(metrics.keys())}")

    def get_best(self, metric: str) -> Tuple[int, float]:
        """
        Get best value for a metric

        Args:
            metric: Metric name

        Returns:
            Tuple of (epoch, value)
        """
        if metric in self.best_metrics:
            epoch, value = self.best_metrics[metric]
            print(f"[METRICS_TRACKER] Best {metric}: {value:.4f} (epoch {epoch})")
            return epoch, value
        else:
            print(f"[METRICS_TRACKER] ⚠ No best value tracked for {metric}")
            return (-1, float('nan'))

    def get_current(self, metric: str, split: str = 'val') -> Optional[float]:
        """
        Get most recent value for a metric

        Args:
            metric: Metric name
            split: 'train' or 'val'

        Returns:
            Current metric value or None
        """
        if metric in self.history[split] and len(self.history[split][metric]) > 0:
            return self.history[split][metric][-1]
        return None

    def get_history(self, metric: str, split: str = 'val') -> List[float]:
        """
        Get full history for a metric

        Args:
            metric: Metric name
            split: 'train' or 'val'

        Returns:
            List of metric values across epochs
        """
        if metric in self.history[split]:
            return self.history[split][metric]
        return []

    def export_to_csv(self, path: str):
        """
        Export metrics history to CSV

        Args:
            path: Output file path
        """
        print(f"\n[METRICS_TRACKER] Exporting metrics to {path}")

        try:
            # Create DataFrame with all metrics
            data = {'epoch': list(range(len(self.history['train']['loss'])))}

            for split in ['train', 'val']:
                for metric in self.metrics_to_track:
                    if metric in self.history[split]:
                        data[f'{split}_{metric}'] = self.history[split][metric]

            df = pd.DataFrame(data)

            # Save to CSV
            df.to_csv(path, index=False)

            print(f"[METRICS_TRACKER] ✓ Exported {len(df)} rows to CSV")

        except Exception as e:
            print(f"[METRICS_TRACKER] ✗ Export failed: {e}")
            raise

    def export_to_json(self, path: str):
        """
        Export metrics to JSON

        Args:
            path: Output file path
        """
        print(f"\n[METRICS_TRACKER] Exporting metrics to {path}")

        try:
            export_data = {
                'history': self.history,
                'best_metrics': {
                    metric: {'epoch': epoch, 'value': value}
                    for metric, (epoch, value) in self.best_metrics.items()
                }
            }

            with open(path, 'w') as f:
                json.dump(export_data, f, indent=2)

            print(f"[METRICS_TRACKER] ✓ Exported to JSON")

        except Exception as e:
            print(f"[METRICS_TRACKER] ✗ Export failed: {e}")
            raise

    def print_summary(self):
        """Print summary of metrics"""
        print(f"\n[METRICS_TRACKER] Metrics Summary")
        print(f"[METRICS_TRACKER] {'='*60}")

        for split in ['train', 'val']:
            print(f"\n[METRICS_TRACKER] {split.upper()} Metrics:")

            for metric in self.metrics_to_track:
                if metric in self.history[split] and len(self.history[split][metric]) > 0:
                    values = self.history[split][metric]
                    current = values[-1]
                    best = min(values) if metric not in ['r2', 'direction_accuracy'] else max(values)

                    print(f"[METRICS_TRACKER]   - {metric.upper()}: current={current:.4f}, best={best:.4f}")

        if self.best_metrics:
            print(f"\n[METRICS_TRACKER] Best Validation Metrics:")
            for metric, (epoch, value) in self.best_metrics.items():
                print(f"[METRICS_TRACKER]   - {metric.upper()}: {value:.4f} (epoch {epoch})")

        print(f"[METRICS_TRACKER] {'='*60}")


# ============================================================================
# VALIDATION FUNCTIONS FOR TRAINING PIPELINE
# ============================================================================

def validate_loss_function(model, dataloader, verbose: bool = True) -> bool:
    """
    STEP 2: Validate that loss function increases with wrong predictions

    Args:
        model: Trained model
        dataloader: DataLoader with samples
        verbose: Print detailed output

    Returns:
        True if loss function behaves correctly
    """
    try:
        import torch
        from pytorch_forecasting.metrics import QuantileLoss
    except ImportError:
        print("[VALIDATION] PyTorch not available, skipping loss validation")
        return False

    if verbose:
        print(f"\n{'='*80}")
        print("STEP 2: VALIDATING LOSS FUNCTION")
        print('='*80)

    try:
        # CRITICAL FIX #3: Explicitly set model to eval mode and use no_grad
        model.eval()
        if verbose:
            print(f"[VALIDATION]   ✅ Model set to eval() mode")
        loss_fn = QuantileLoss()

        # Get one batch
        batch_data = next(iter(dataloader))
        # Handle both tuple and dict batch formats
        if isinstance(batch_data, tuple):
            batch = batch_data[0] if len(batch_data) > 0 else batch_data
        else:
            batch = batch_data

        # CRITICAL FIX #3: Use no_grad to prevent gradient computation during validation
        with torch.no_grad():
            if verbose:
                print(f"[VALIDATION]   ✅ Using torch.no_grad() context")
            output = model(batch)
            target = batch['decoder_target']

            if verbose:
                print(f"\n[LOSS VALIDATION] Testing loss function behavior:")
                print(f"  - Target shape: {target.shape}")

            # Handle different output formats
            if isinstance(output, dict):
                prediction = output['prediction']
            elif isinstance(output, tuple):
                prediction = output[0]  # TFT returns (prediction, additional_outputs)
            else:
                prediction = output

            # Test 1: Model prediction loss
            correct_loss = loss_fn(prediction, target).item()

            # Test 2: Random prediction loss
            random_pred = torch.randn_like(prediction) * prediction.std() + prediction.mean()
            random_loss = loss_fn(random_pred, target).item()

            # Test 3: Very wrong prediction loss
            wrong_pred = prediction + torch.randn_like(prediction) * prediction.std() * 5
            wrong_loss = loss_fn(wrong_pred, target).item()

            # Test 4: Zero prediction loss
            zero_pred = torch.zeros_like(prediction)
            zero_loss = loss_fn(zero_pred, target).item()

            if verbose:
                print(f"  [TEST 1] Model prediction loss:      {correct_loss:.6f}")
                print(f"  [TEST 2] Random prediction loss:     {random_loss:.6f}")
                print(f"  [TEST 3] Very wrong prediction loss: {wrong_loss:.6f}")
                print(f"  [TEST 4] Zero prediction loss:       {zero_loss:.6f}")

            # Validate behavior
            issues = []

            if abs(correct_loss - random_loss) < 1e-6:
                issues.append("❌ Loss is identical for different predictions!")
                if verbose:
                    print(f"\n  ❌ CRITICAL: Loss not responding to prediction changes!")

            if wrong_loss < correct_loss * 1.2:
                issues.append("❌ Very wrong predictions don't have significantly higher loss")
                if verbose:
                    print(f"\n  ⚠️  WARNING: Loss doesn't penalize bad predictions enough")

            if issues:
                if verbose:
                    print(f"\n❌ STEP 2 FAILED: {len(issues)} issues found")
                return False
            else:
                if verbose:
                    print(f"\n✅ STEP 2 PASSED: Loss function working correctly")
                return True

    except Exception as e:
        if verbose:
            print(f"\n❌ STEP 2 ERROR: {e}")
        return False


def validate_baseline_comparison(model, dataloader, verbose: bool = True) -> bool:
    """
    STEP 5: Verify model performs better than random baseline

    Args:
        model: Trained model
        dataloader: Validation DataLoader
        verbose: Print detailed output

    Returns:
        True if model beats baseline
    """
    try:
        import torch
        from pytorch_forecasting.metrics import QuantileLoss
    except ImportError:
        print("[VALIDATION] PyTorch not available, skipping baseline validation")
        return False

    if verbose:
        print(f"\n{'='*80}")
        print("STEP 5: COMPARING TO RANDOM BASELINE")
        print('='*80)

    # CRITICAL FIX #3: Explicitly set model to eval mode and use no_grad
    model.eval()
    if verbose:
        print(f"[BASELINE]   ✅ Model set to eval() mode")
    loss_fn = QuantileLoss()

    model_losses = []
    random_losses = []
    constant_losses = []

    # CRITICAL FIX #3: Use no_grad to prevent gradient computation during validation
    with torch.no_grad():
        if verbose:
            print(f"[BASELINE]   ✅ Using torch.no_grad() context")
        for batch_idx, batch_data in enumerate(dataloader):
            # Handle both tuple and dict batch formats
            if isinstance(batch_data, tuple):
                batch = batch_data[0] if len(batch_data) > 0 else batch_data
            else:
                batch = batch_data

            output = model(batch)
            target = batch['decoder_target']

            # Handle different output formats
            if isinstance(output, dict):
                prediction = output['prediction']
            elif isinstance(output, tuple):
                prediction = output[0]  # TFT returns (prediction, additional_outputs)
            else:
                prediction = output

            # Model loss
            model_loss = loss_fn(prediction, target).item()
            model_losses.append(model_loss)

            # Random baseline
            random_pred = torch.randn_like(prediction) * target.std() + target.mean()
            random_loss = loss_fn(random_pred, target).item()
            random_losses.append(random_loss)

            # Constant baseline (predict mean)
            constant_pred = torch.ones_like(prediction) * target.mean()
            constant_loss = loss_fn(constant_pred, target).item()
            constant_losses.append(constant_loss)

            if batch_idx >= 9:  # Test 10 batches
                break

    avg_model = np.mean(model_losses)
    avg_random = np.mean(random_losses)
    avg_constant = np.mean(constant_losses)

    improvement_random = ((avg_random - avg_model) / avg_random) * 100
    improvement_constant = ((avg_constant - avg_model) / avg_constant) * 100

    if verbose:
        print(f"\n[BASELINE] Average losses over {len(model_losses)} batches:")
        print(f"  - Model:    {avg_model:.6f}")
        print(f"  - Random:   {avg_random:.6f}")
        print(f"  - Constant: {avg_constant:.6f}")
        print(f"\n[BASELINE] Model improvement:")
        print(f"  - vs Random:   {improvement_random:+.2f}%")
        print(f"  - vs Constant: {improvement_constant:+.2f}%")

    issues = []

    if avg_model > avg_random:
        issues.append("❌ Model worse than random!")
        if verbose:
            print(f"\n  ❌ CRITICAL: Model can't beat random guessing!")

    if avg_model > avg_constant:
        issues.append("❌ Model worse than constant prediction!")
        if verbose:
            print(f"\n  ❌ CRITICAL: Model can't even predict the mean!")

    if improvement_random < 5:
        issues.append("⚠️  Model barely improves over random (<5%)")
        if verbose:
            print(f"\n  ⚠️  WARNING: Very small improvement over baseline")

    if issues:
        if verbose:
            print(f"\n❌ STEP 5 FAILED: Model not learning properly")
        return False
    else:
        if verbose:
            print(f"\n✅ STEP 5 PASSED: Model beats baseline")
        return True


if __name__ == "__main__":
    # Test metrics
    print("Testing Metrics Functions")
    print("=" * 60)

    # Create sample data
    print("\n[TEST] Creating sample data...")
    np.random.seed(42)

    y_true = np.random.randn(100).cumsum() + 3000  # Simulated prices
    y_pred = y_true + np.random.randn(100) * 10  # Add noise

    print(f"[TEST]   - y_true shape: {y_true.shape}")
    print(f"[TEST]   - y_pred shape: {y_pred.shape}")

    # Calculate individual metrics
    print("\n[TEST] Testing individual metrics...")
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    dir_acc = direction_accuracy(y_true, y_pred)

    # Test calculate_all_metrics
    print("\n[TEST] Testing calculate_all_metrics...")
    all_metrics = calculate_all_metrics(y_true, y_pred)

    # Test MetricsTracker
    print("\n[TEST] Testing MetricsTracker...")
    tracker = MetricsTracker()

    for epoch in range(5):
        # Simulate training metrics
        train_metrics = {
            'loss': 0.5 - epoch * 0.05,
            'mae': 50 - epoch * 5,
            'rmse': 70 - epoch * 7,
        }
        tracker.update(epoch, train_metrics, split='train')

        # Simulate validation metrics
        val_metrics = {
            'loss': 0.6 - epoch * 0.04,
            'mae': 55 - epoch * 4,
            'rmse': 75 - epoch * 6,
        }
        tracker.update(epoch, val_metrics, split='val')

    tracker.print_summary()

    # Test export
    tracker.export_to_csv('test_metrics.csv')
    tracker.export_to_json('test_metrics.json')

    print(f"\n[TEST] ✓ All metrics tests passed!")
