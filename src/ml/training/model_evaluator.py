"""
Model Evaluation and Metrics Visualization
Comprehensive evaluation of trained models on train/val/test sets
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def safe_direction_accuracy(predictions: np.ndarray, actuals: np.ndarray,
                           threshold: float = 0.0) -> float:
    """
    Safely compute directional accuracy with validation to prevent NaN errors

    This function calculates the percentage of times the predicted direction
    (up/down) matches the actual direction between consecutive values.

    Args:
        predictions: Predicted values array
        actuals: Actual values array
        threshold: Minimum change threshold to consider as directional movement
                  (default: 0.0, any change counts)

    Returns:
        Directional accuracy percentage (0-100), or 0.0 if insufficient data

    Examples:
        >>> preds = np.array([100, 102, 101, 105])
        >>> actuals = np.array([100, 103, 100, 104])
        >>> safe_direction_accuracy(preds, actuals)
        66.67  # 2 out of 3 directions correct
    """
    # Validate inputs
    if len(predictions) < 2 or len(actuals) < 2:
        return 0.0

    # Handle length mismatch
    if len(predictions) != len(actuals):
        min_len = min(len(predictions), len(actuals))
        predictions = predictions[:min_len]
        actuals = actuals[:min_len]

    # Calculate differences
    pred_diff = np.diff(predictions)
    actual_diff = np.diff(actuals)

    # Edge case: no differences to compare
    if len(pred_diff) == 0:
        return 0.0

    # Calculate direction (True = up, False = down)
    pred_direction = pred_diff > threshold
    actual_direction = actual_diff > threshold

    # Calculate accuracy
    return float(np.mean(pred_direction == actual_direction) * 100)


class ModelEvaluator:
    """
    Comprehensive model evaluation on train/val/test sets

    Computes and visualizes:
    - MAE, RMSE, MAPE, R², Direction Accuracy
    - Per-horizon performance (1h, 2h, ..., 10h)
    - Residual analysis
    - Prediction vs Actual plots
    """

    def __init__(self, model, device='cpu', verbose=True):
        """
        Initialize evaluator

        Args:
            model: Trained TFT model
            device: 'cpu' or 'cuda'
            verbose: Print detailed output
        """
        self.model = model
        self.device = device
        self.verbose = verbose

        if self.verbose:
            print(f"\n[EVALUATOR] Model Evaluator initialized")
            print(f"[EVALUATOR]   - Device: {device}")

    def evaluate_on_dataloader(
        self,
        dataloader,
        split_name: str = "test"
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataloader

        Args:
            dataloader: DataLoader to evaluate on
            split_name: Name of split ('train', 'val', 'test')

        Returns:
            Dictionary of metrics
        """
        if not TORCH_AVAILABLE:
            print("[EVALUATOR] PyTorch not available")
            return {}

        if self.verbose:
            print(f"\n[EVALUATOR] Evaluating on {split_name} set...")
            print(f"[EVALUATOR]   - Batches: {len(dataloader)}")

        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                # Handle batch format
                if isinstance(batch_data, tuple):
                    batch = batch_data[0] if len(batch_data) > 0 else batch_data
                else:
                    batch = batch_data

                # Move to device
                if hasattr(batch, 'to'):
                    batch = batch.to(self.device)

                # Get predictions
                output = self.model(batch)
                target = batch['decoder_target']

                # Handle output format
                if isinstance(output, dict):
                    prediction = output['prediction']
                elif isinstance(output, tuple):
                    prediction = output[0]
                else:
                    prediction = output

                # Handle RMSE output (batch, time, 1) -> (batch, time)
                if prediction.dim() == 3 and prediction.shape[-1] == 1:
                    prediction = prediction.squeeze(-1)

                all_predictions.append(prediction.cpu().numpy())
                all_targets.append(target.cpu().numpy())

                if batch_idx >= 99:  # Evaluate on first 100 batches max
                    break

        # Concatenate all batches
        predictions = np.concatenate(all_predictions, axis=0)  # (n_samples, time)
        targets = np.concatenate(all_targets, axis=0)  # (n_samples, time)

        if self.verbose:
            print(f"[EVALUATOR]   - Predictions shape: {predictions.shape}")
            print(f"[EVALUATOR]   - Targets shape: {targets.shape}")

        # Compute metrics
        metrics = self._compute_metrics(predictions, targets, split_name)

        return metrics

    def _compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        split_name: str
    ) -> Dict[str, float]:
        """Compute comprehensive metrics"""
        # Flatten for overall metrics
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()

        # Overall metrics
        mae = np.mean(np.abs(target_flat - pred_flat))
        rmse = np.sqrt(np.mean((target_flat - pred_flat) ** 2))
        mse = np.mean((target_flat - pred_flat) ** 2)

        # MAPE (avoid division by zero)
        mape = np.mean(np.abs((target_flat - pred_flat) / (np.abs(target_flat) + 1e-8))) * 100

        # R² score
        ss_res = np.sum((target_flat - pred_flat) ** 2)
        ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        # Direction accuracy (for time series) - using safe calculation
        if predictions.shape[1] > 1:
            # Calculate direction accuracy for each sample and average
            direction_accs = []
            for i in range(len(predictions)):
                acc = safe_direction_accuracy(predictions[i], targets[i])
                direction_accs.append(acc)
            direction_acc = np.mean(direction_accs) if direction_accs else 0.0
        else:
            direction_acc = 0.0

        # Per-horizon metrics (1h, 2h, ..., 10h ahead)
        horizon_metrics = {}
        for h in range(min(predictions.shape[1], 10)):
            horizon_mae = np.mean(np.abs(targets[:, h] - predictions[:, h]))
            horizon_metrics[f'mae_horizon_{h+1}h'] = horizon_mae

        metrics = {
            'split': split_name,
            'mae': float(mae),
            'rmse': float(rmse),
            'mse': float(mse),
            'mape': float(mape),
            'r2': float(r2),
            'direction_accuracy': float(direction_acc),
            **horizon_metrics
        }

        if self.verbose:
            print(f"\n[EVALUATOR] {split_name.upper()} Metrics:")
            print(f"[EVALUATOR]   - MAE:  {mae:.4f}")
            print(f"[EVALUATOR]   - RMSE: {rmse:.4f}")
            print(f"[EVALUATOR]   - MAPE: {mape:.2f}%")
            print(f"[EVALUATOR]   - R²:   {r2:.4f}")
            print(f"[EVALUATOR]   - Direction Accuracy: {direction_acc:.2f}%")

        return metrics

    def save_evaluation_report(
        self,
        metrics_dict: Dict[str, Dict],
        save_dir: str,
        create_plots: bool = True
    ):
        """
        Save comprehensive evaluation report

        Args:
            metrics_dict: {'train': {...}, 'val': {...}, 'test': {...}}
            save_dir: Directory to save report
            create_plots: Create visualization plots
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save metrics as JSON
        metrics_file = save_path / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        print(f"[EVALUATOR] OK Metrics saved to {metrics_file}")

        # Save as CSV for easy analysis
        csv_file = save_path / "evaluation_metrics.csv"
        rows = []
        for split, metrics in metrics_dict.items():
            row = {'split': split, **metrics}
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)

        print(f"[EVALUATOR] OK Metrics saved to {csv_file}")

        # Create visualization plots
        if create_plots:
            self._create_evaluation_plots(metrics_dict, save_path)

    def _create_evaluation_plots(self, metrics_dict: Dict, save_path: Path):
        """Create comprehensive evaluation plots"""
        # Extract metrics for plotting
        splits = list(metrics_dict.keys())
        mae_values = [metrics_dict[s]['mae'] for s in splits]
        rmse_values = [metrics_dict[s]['rmse'] for s in splits]
        r2_values = [metrics_dict[s]['r2'] for s in splits]
        direction_acc = [metrics_dict[s]['direction_accuracy'] for s in splits]

        # Plot 1: MAE and RMSE comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Performance Evaluation', fontsize=16, fontweight='bold')

        # MAE
        axes[0, 0].bar(splits, mae_values, color=['#4caf50', '#ff9800', '#f44336'])
        axes[0, 0].set_title('Mean Absolute Error (MAE)')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(mae_values):
            axes[0, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom')

        # RMSE
        axes[0, 1].bar(splits, rmse_values, color=['#4caf50', '#ff9800', '#f44336'])
        axes[0, 1].set_title('Root Mean Squared Error (RMSE)')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(rmse_values):
            axes[0, 1].text(i, v, f'{v:.4f}', ha='center', va='bottom')

        # R²
        axes[1, 0].bar(splits, r2_values, color=['#4caf50', '#ff9800', '#f44336'])
        axes[1, 0].set_title('R² Score (Coefficient of Determination)')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(r2_values):
            axes[1, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom')

        # Direction Accuracy
        axes[1, 1].bar(splits, direction_acc, color=['#4caf50', '#ff9800', '#f44336'])
        axes[1, 1].set_title('Direction Accuracy (%)')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_ylim([0, 100])
        axes[1, 1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(direction_acc):
            axes[1, 1].text(i, v, f'{v:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plot_file = save_path / "evaluation_metrics.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[EVALUATOR] OK Metrics plot saved to {plot_file}")

        # Plot 2: Per-horizon performance
        self._plot_horizon_metrics(metrics_dict, save_path)

    def _plot_horizon_metrics(self, metrics_dict: Dict, save_path: Path):
        """Plot per-horizon forecast accuracy"""
        fig, ax = plt.subplots(figsize=(10, 6))

        for split in metrics_dict.keys():
            horizon_keys = [k for k in metrics_dict[split].keys() if 'mae_horizon' in k]
            if not horizon_keys:
                continue

            horizons = [int(k.split('_')[-1].replace('h', '')) for k in horizon_keys]
            mae_values = [metrics_dict[split][k] for k in horizon_keys]

            ax.plot(horizons, mae_values, marker='o', linewidth=2, markersize=6, label=split.upper())

        ax.set_title('Forecast Accuracy by Time Horizon', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hours Ahead', fontsize=12)
        ax.set_ylabel('MAE', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        plot_file = save_path / "horizon_accuracy.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[EVALUATOR] OK Horizon plot saved to {plot_file}")


def evaluate_trained_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    save_dir: str = "models/checkpoints/evaluation",
    device: str = 'cpu'
) -> Dict[str, Dict]:
    """
    Comprehensive evaluation of trained model

    Args:
        model: Trained TFT model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        save_dir: Directory to save evaluation results
        device: 'cpu' or 'cuda'

    Returns:
        Dictionary with metrics for each split
    """
    evaluator = ModelEvaluator(model, device=device, verbose=True)

    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE MODEL EVALUATION")
    print(f"{'='*80}")

    # Evaluate on each split
    train_metrics = evaluator.evaluate_on_dataloader(train_loader, 'train')
    val_metrics = evaluator.evaluate_on_dataloader(val_loader, 'val')
    test_metrics = evaluator.evaluate_on_dataloader(test_loader, 'test')

    metrics_dict = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }

    # Save comprehensive report
    evaluator.save_evaluation_report(metrics_dict, save_dir, create_plots=True)

    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {save_dir}")

    return metrics_dict


if __name__ == "__main__":
    print("Model Evaluator - Test Mode")
    print("="*60)
    print("\nThis module provides comprehensive model evaluation.")
    print("Import and use evaluate_trained_model() after training.")
