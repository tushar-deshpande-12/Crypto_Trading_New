"""
Debug Logger for AI Pipeline Diagnostics
Logs debug information to JSON files for troubleshooting
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class DebugLogger:
    """
    Debug logger that saves diagnostic information to JSON files.

    Helps identify issues in:
    - Data loading and preprocessing
    - Feature engineering
    - Model training
    - Prediction/inference
    - Backtesting
    """

    def __init__(self, log_dir: str = "debug_logs", enabled: bool = True):
        """
        Initialize debug logger.

        Args:
            log_dir: Directory for debug log files
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log: List[Dict] = []

        if self.enabled:
            print(f"[DEBUG] Debug logger initialized - session: {self.session_id}")
            print(f"[DEBUG] Log directory: {self.log_dir}")

    def _to_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return {
                "type": "ndarray",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "min": float(np.nanmin(obj)) if obj.size > 0 else None,
                "max": float(np.nanmax(obj)) if obj.size > 0 else None,
                "mean": float(np.nanmean(obj)) if obj.size > 0 else None,
                "std": float(np.nanstd(obj)) if obj.size > 0 else None,
                "nan_count": int(np.isnan(obj).sum()) if obj.size > 0 else 0,
                "sample": obj.flatten()[:10].tolist() if obj.size > 0 else []
            }
        elif isinstance(obj, pd.DataFrame):
            return {
                "type": "DataFrame",
                "shape": list(obj.shape),
                "columns": list(obj.columns),
                "dtypes": {str(k): str(v) for k, v in obj.dtypes.items()},
                "nan_counts": obj.isna().sum().to_dict(),
                "sample": obj.head(3).to_dict() if len(obj) > 0 else {}
            }
        elif isinstance(obj, pd.Series):
            return {
                "type": "Series",
                "length": len(obj),
                "dtype": str(obj.dtype),
                "nan_count": int(obj.isna().sum()),
                "sample": obj.head(5).tolist()
            }
        elif isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            if len(obj) > 20:
                return {
                    "type": "list",
                    "length": len(obj),
                    "sample": [self._to_serializable(x) for x in obj[:10]]
                }
            return [self._to_serializable(x) for x in obj]
        elif hasattr(obj, '__dict__'):
            return {
                "type": type(obj).__name__,
                "attrs": {k: str(v)[:100] for k, v in obj.__dict__.items() if not k.startswith('_')}
            }
        else:
            return str(obj)[:200]

    def log(self, stage: str, step: str, data: Dict[str, Any], status: str = "info"):
        """
        Log a debug entry.

        Args:
            stage: Pipeline stage (e.g., 'preprocessing', 'training', 'prediction')
            step: Specific step within stage
            data: Data to log
            status: Status level ('info', 'warning', 'error', 'success')
        """
        if not self.enabled:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "step": step,
            "status": status,
            "data": self._to_serializable(data)
        }

        self.session_log.append(entry)

        # Print summary
        status_icon = {"info": "[i]", "warning": "[!]", "error": "[X]", "success": "[OK]"}.get(status, "[?]")
        print(f"[DEBUG] {status_icon} [{stage}] {step}")

        # Save incrementally
        self._save_session_log()

    def log_data_stats(self, stage: str, df: pd.DataFrame, label: str = "data"):
        """Log comprehensive statistics for a DataFrame."""
        if not self.enabled:
            return

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        stats = {
            "shape": df.shape,
            "columns": list(df.columns),
            "numeric_columns": numeric_cols,
            "nan_summary": df.isna().sum().to_dict(),
            "total_nan": int(df.isna().sum().sum()),
        }

        # Add numeric column statistics
        if numeric_cols:
            stats["numeric_stats"] = {}
            for col in numeric_cols[:20]:  # Limit to 20 columns
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    stats["numeric_stats"][col] = {
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "mean": float(col_data.mean()),
                        "std": float(col_data.std()),
                        "median": float(col_data.median())
                    }

        self.log(stage, f"{label}_stats", stats)

    def log_training_step(self, epoch: int, train_loss: float, val_loss: float,
                          metrics: Optional[Dict] = None):
        """Log a training step."""
        data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "loss_ratio": val_loss / train_loss if train_loss > 0 else None,
        }
        if metrics:
            data["metrics"] = metrics

        # Determine status based on loss ratio
        if val_loss / train_loss > 1.5:
            status = "warning"  # Possible overfitting
        elif val_loss < train_loss:
            status = "success"
        else:
            status = "info"

        self.log("training", f"epoch_{epoch}", data, status)

    def log_prediction(self, predictions: np.ndarray, actuals: Optional[np.ndarray] = None,
                       metrics: Optional[Dict] = None):
        """Log prediction results."""
        data = {
            "predictions": predictions,
        }
        if actuals is not None:
            data["actuals"] = actuals
            data["errors"] = predictions - actuals
        if metrics:
            data["metrics"] = metrics

        self.log("prediction", "results", data)

    def log_error(self, stage: str, step: str, error: Exception, context: Optional[Dict] = None):
        """Log an error with context."""
        data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        self.log(stage, step, data, status="error")

    def _save_session_log(self):
        """Save session log to JSON file."""
        if not self.session_log:
            return

        log_path = self.log_dir / f"debug_{self.session_id}.json"
        try:
            with open(log_path, 'w') as f:
                json.dump(self.session_log, f, indent=2, default=str)
        except Exception as e:
            print(f"[DEBUG] Failed to save log: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of logged events."""
        summary = {
            "session_id": self.session_id,
            "total_entries": len(self.session_log),
            "stages": {},
            "status_counts": {"info": 0, "warning": 0, "error": 0, "success": 0}
        }

        for entry in self.session_log:
            stage = entry.get("stage", "unknown")
            status = entry.get("status", "info")

            if stage not in summary["stages"]:
                summary["stages"][stage] = 0
            summary["stages"][stage] += 1
            summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1

        return summary

    def export_summary(self, output_path: Optional[str] = None) -> str:
        """Export summary report."""
        summary = self.get_summary()

        if output_path is None:
            output_path = self.log_dir / f"summary_{self.session_id}.json"

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"[DEBUG] Summary exported to: {output_path}")
        return str(output_path)


# Global debug logger instance
_debug_logger: Optional[DebugLogger] = None


def get_debug_logger(enabled: bool = True) -> DebugLogger:
    """Get or create the global debug logger instance."""
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = DebugLogger(enabled=enabled)
    return _debug_logger


def debug_log(stage: str, step: str, data: Dict[str, Any], status: str = "info"):
    """Convenience function for logging."""
    logger = get_debug_logger()
    logger.log(stage, step, data, status)


if __name__ == "__main__":
    # Test debug logger
    print("Testing DebugLogger")
    print("=" * 60)

    logger = DebugLogger()

    # Test basic logging
    logger.log("test", "basic_log", {"key": "value"})

    # Test with numpy array
    import numpy as np
    arr = np.random.randn(100, 10)
    logger.log("test", "numpy_array", {"array": arr})

    # Test with DataFrame
    import pandas as pd
    df = pd.DataFrame({
        'a': np.random.randn(100),
        'b': np.random.randn(100),
        'c': ['x', 'y'] * 50
    })
    logger.log_data_stats("test", df, "test_df")

    # Test training step
    logger.log_training_step(1, 0.5, 0.6)
    logger.log_training_step(2, 0.4, 0.8)  # Overfitting warning

    # Print summary
    summary = logger.get_summary()
    print(f"\nSummary: {summary}")

    print("\n[TEST] Debug logger test complete!")
