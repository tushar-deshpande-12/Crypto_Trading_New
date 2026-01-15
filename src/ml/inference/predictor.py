"""
Cryptocurrency Price Predictor
Inference pipeline for generating predictions with trained TFT model

IMPORTANT: This predictor handles RETURN predictions (target_return)
and converts them to price predictions using proper denormalization.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from datetime import datetime, timedelta

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("[PREDICTOR] [!] Warning: PyTorch not installed")
    TORCH_AVAILABLE = False

from ..models.tft_model import CryptoTFT
from ..preprocessing.preprocessor import CryptoPreprocessor

# Default prediction horizon (must match model training)
PREDICTION_HORIZON = 24


class CryptoPredictor:
    """
    Cryptocurrency price predictor using trained TFT model

    Features:
    - Load trained model and preprocessor
    - Preprocess recent data
    - Generate 10-hour predictions
    - Extract confidence intervals
    - Denormalize predictions to actual prices
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        dataset_dir: str = "dataset",
        verbose: bool = True
    ):
        """
        Initialize predictor

        Args:
            model_path: Path to trained model checkpoint
            scaler_path: Path to fitted scalers
            dataset_dir: Directory with cryptocurrency data
            verbose: Print progress information
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        self.verbose = verbose

        if self.verbose:
            print(f"\n[PREDICTOR] Initializing CryptoPredictor")
            print(f"[PREDICTOR]   - Model: {model_path}")
            print(f"[PREDICTOR]   - Scaler: {scaler_path}")
            print(f"[PREDICTOR]   - Dataset dir: {dataset_dir}")

        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.dataset_dir = Path(dataset_dir)

        # Load model
        self._load_model()

        # Load preprocessor with scalers
        self._load_preprocessor()

        if self.verbose:
            print(f"[PREDICTOR] [OK] CryptoPredictor initialized")

    def _load_model(self):
        """Load trained model"""
        if self.verbose:
            print(f"\n[PREDICTOR] Loading trained model...")

        try:
            self.model = CryptoTFT.load_model(
                str(self.model_path),
                verbose=self.verbose
            )

            if self.verbose:
                print(f"[PREDICTOR] [OK] Model loaded successfully")

        except Exception as e:
            print(f"[PREDICTOR] [X] Failed to load model: {e}")
            raise

    def _load_preprocessor(self):
        """Load preprocessor with fitted scalers"""
        if self.verbose:
            print(f"\n[PREDICTOR] Loading preprocessor...")

        try:
            self.preprocessor = CryptoPreprocessor(
                dataset_dir=str(self.dataset_dir)
            )

            # Load fitted scalers
            self.preprocessor.load_scaler(str(self.scaler_path))

            if self.verbose:
                print(f"[PREDICTOR] [OK] Preprocessor loaded")
                print(f"[PREDICTOR]   - Loaded scalers for {len(self.preprocessor.scalers)} symbols")

        except Exception as e:
            print(f"[PREDICTOR] [X] Failed to load preprocessor: {e}")
            raise

    def load_recent_data(
        self,
        symbol: str,
        n_candles: int = 40000
    ) -> pd.DataFrame:
        """
        Load recent data for prediction

        Args:
            symbol: Cryptocurrency symbol (e.g., 'ETHUSDT')
            n_candles: Number of recent candles to load (context length)

        Returns:
            DataFrame with recent data
        """
        if self.verbose:
            print(f"\n[PREDICTOR] Loading recent data for {symbol}...")
            print(f"[PREDICTOR]   - Candles needed: {n_candles}")

        try:
            # Load data using preprocessor
            df = self.preprocessor.load_multi_symbol_data([symbol])

            if self.verbose:
                print(f"[PREDICTOR]   - Loaded {len(df)} total candles")

            # Take last n_candles
            if len(df) > n_candles:
                df = df.tail(n_candles).reset_index(drop=True)
                if self.verbose:
                    print(f"[PREDICTOR]   - Using last {n_candles} candles")
            else:
                if self.verbose:
                    print(f"[PREDICTOR] [!] Warning: Only {len(df)} candles available (need {n_candles})")

            if self.verbose:
                print(f"[PREDICTOR]   - Date range: {df['datetime'].min()} to {df['datetime'].max()}")
                print(f"[PREDICTOR] [OK] Recent data loaded")

            return df

        except Exception as e:
            print(f"[PREDICTOR] [X] Failed to load recent data: {e}")
            raise

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for prediction

        Args:
            data: Raw DataFrame

        Returns:
            Preprocessed DataFrame ready for model
        """
        if self.verbose:
            print(f"\n[PREDICTOR] Preprocessing data...")
            print(f"[PREDICTOR]   - Input shape: {data.shape}")

        try:
            df = data.copy()

            # Generate features (same as training)
            df = self.preprocessor.generate_temporal_features(df)
            df = self.preprocessor.generate_technical_indicators(df)
            df = self.preprocessor.generate_lagged_features(df)
            df = self.preprocessor.generate_rolling_features(df)

            # Remove NaN rows
            if self.verbose:
                nan_before = len(df)
                df = df.dropna().reset_index(drop=True)
                nan_after = len(df)
                if nan_before != nan_after:
                    print(f"[PREDICTOR]   - Removed {nan_before - nan_after} rows with NaN")

            # Normalize using fitted scalers (fit=False to use existing scalers)
            df = self.preprocessor.normalize(df, fit=False)

            if self.verbose:
                print(f"[PREDICTOR]   - Output shape: {df.shape}")
                print(f"[PREDICTOR] [OK] Preprocessing complete")

            return df

        except Exception as e:
            print(f"[PREDICTOR] [X] Preprocessing failed: {e}")
            raise

    def predict(
        self,
        symbol: str,
        n_hours: int = 10,
        return_confidence_intervals: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate prediction for next N hours

        Args:
            symbol: Cryptocurrency symbol
            n_hours: Number of hours to predict
            return_confidence_intervals: Include confidence intervals

        Returns:
            Dictionary with predictions:
            - 'median': Median prediction
            - 'lower_95': Lower 95% confidence bound
            - 'upper_95': Upper 95% confidence bound
            - 'lower_80': Lower 80% confidence bound
            - 'upper_80': Upper 80% confidence bound
            - 'timestamps': Predicted timestamps
        """
        if self.verbose:
            print(f"\n[PREDICTOR] Generating {n_hours}-hour prediction for {symbol}...")

        try:
            # Load recent data
            context_length = self.model.config.max_encoder_length
            recent_data = self.load_recent_data(symbol, n_candles=context_length + 1000)

            # Preprocess
            preprocessed_data = self.preprocess(recent_data)

            if len(preprocessed_data) < context_length:
                raise ValueError(
                    f"Insufficient data after preprocessing: "
                    f"need {context_length}, have {len(preprocessed_data)}"
                )

            # Create dataset for prediction using model's training parameters
            from pytorch_forecasting import TimeSeriesDataSet

            # Check if model has dataset parameters
            if not hasattr(self.model.model, 'dataset_parameters'):
                raise ValueError("Model doesn't have training dataset parameters. Please retrain the model.")

            if self.verbose:
                print(f"[PREDICTOR] Creating dataset from model training parameters...")

            # Modify parameters to allow unknown categories
            params = self.model.model.dataset_parameters.copy()
            if 'categorical_encoders' in params:
                for key, encoder in params['categorical_encoders'].items():
                    if hasattr(encoder, 'add_nan'):
                        encoder.add_nan = True

            pred_dataset = TimeSeriesDataSet.from_parameters(
                params,
                preprocessed_data,
                predict=True,  # Inference mode
                stop_randomization=True  # No augmentation
            )

            # Get dataloader
            pred_dataloader = pred_dataset.to_dataloader(
                train=False,  # Inference mode: no shuffling
                batch_size=1,
                num_workers=0
            )

            # Generate predictions
            if self.verbose:
                print(f"[PREDICTOR] Generating predictions...")

            predictions = self.model.predict_next_n_hours(
                pred_dataloader,
                n_hours=n_hours,
                return_quantiles=return_confidence_intervals
            )

            # Get current price for return-to-price conversion
            # Use the last known close price from the data
            current_price = None
            if symbol in self.preprocessor.scalers:
                scaler = self.preprocessor.scalers[symbol]
                try:
                    close_idx = self.preprocessor.feature_columns.index('close')
                    # Get the last normalized close value and denormalize it
                    last_close_norm = preprocessed_data['close'].iloc[-1]
                    current_price = last_close_norm * scaler.scale_[close_idx] + scaler.mean_[close_idx]
                    if self.verbose:
                        print(f"[PREDICTOR]   - Current price for {symbol}: ${current_price:.2f}")
                except (ValueError, IndexError) as e:
                    if self.verbose:
                        print(f"[PREDICTOR]   [!] Could not get current price: {e}")

            # Denormalize predictions (returns -> prices)
            predictions_denorm = self._denormalize_predictions(
                predictions,
                symbol,
                current_price=current_price
            )

            # Generate future timestamps
            last_timestamp = preprocessed_data['datetime'].iloc[-1]
            future_timestamps = [
                last_timestamp + timedelta(hours=i+1)
                for i in range(n_hours)
            ]

            predictions_denorm['timestamps'] = future_timestamps

            if self.verbose:
                print(f"[PREDICTOR] [OK] Prediction complete")
                print(f"[PREDICTOR]   - Predicted {n_hours} hours")
                print(f"[PREDICTOR]   - Price range: ${predictions_denorm['median'].min():.2f} - "
                      f"${predictions_denorm['median'].max():.2f}")

            return predictions_denorm

        except Exception as e:
            print(f"[PREDICTOR] [X] Prediction failed: {e}")
            raise

    def _denormalize_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        symbol: str,
        current_price: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Denormalize return predictions to actual price values.

        The model predicts normalized returns (target_return).
        This method:
        1. Denormalizes returns using target_scaler
        2. Converts returns to prices using current price

        Args:
            predictions: Normalized return predictions
            symbol: Cryptocurrency symbol
            current_price: Current price for conversion (optional)

        Returns:
            Denormalized price predictions
        """
        if self.verbose:
            print(f"\n[PREDICTOR] Denormalizing predictions...")

        try:
            # Get target scaler (for target_return)
            target_scaler = self.preprocessor.target_scaler

            if target_scaler is None:
                if self.verbose:
                    print(f"[PREDICTOR]   [!] No target scaler found - assuming raw returns")
                # Assume predictions are already actual returns
                actual_returns = predictions
            else:
                if self.verbose:
                    print(f"[PREDICTOR]   - Using target scaler")
                    print(f"[PREDICTOR]   - Target mean: {target_scaler.mean_[0]:.6f}")
                    print(f"[PREDICTOR]   - Target scale: {target_scaler.scale_[0]:.6f}")

                # Denormalize returns
                actual_returns = {}
                for key, values in predictions.items():
                    if key != 'timestamps':
                        # Denormalize: actual = normalized * scale + mean
                        denorm_values = values * target_scaler.scale_[0] + target_scaler.mean_[0]
                        actual_returns[key] = denorm_values

                        if self.verbose:
                            print(f"[PREDICTOR]   - {key} returns: {denorm_values.min():.4f} to {denorm_values.max():.4f}")

            # Convert returns to prices if current price provided
            if current_price is not None:
                if self.verbose:
                    print(f"[PREDICTOR]   - Converting returns to prices using current: ${current_price:.2f}")

                denorm_predictions = {}
                for key, returns in actual_returns.items():
                    if key != 'timestamps':
                        # future_price = current_price * (1 + return)
                        prices = current_price * (1 + returns)
                        denorm_predictions[key] = prices.flatten()

                        if self.verbose:
                            print(f"[PREDICTOR]   - {key} prices: ${prices.min():.2f} to ${prices.max():.2f}")
            else:
                # Return actual returns without price conversion
                denorm_predictions = {k: v.flatten() if hasattr(v, 'flatten') else v
                                      for k, v in actual_returns.items()}
                if self.verbose:
                    print(f"[PREDICTOR]   [!] No current price - returning actual returns")

            if self.verbose:
                print(f"[PREDICTOR] [OK] Denormalization complete")

            return denorm_predictions

        except Exception as e:
            print(f"[PREDICTOR] [X] Denormalization failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def predict_multiple(
        self,
        symbols: List[str],
        n_hours: int = 10
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate predictions for multiple symbols

        Args:
            symbols: List of cryptocurrency symbols
            n_hours: Number of hours to predict

        Returns:
            Dictionary mapping symbol to predictions
        """
        if self.verbose:
            print(f"\n[PREDICTOR] Generating predictions for {len(symbols)} symbols...")

        results = {}

        for i, symbol in enumerate(symbols, 1):
            if self.verbose:
                print(f"\n[PREDICTOR] [{i}/{len(symbols)}] Predicting {symbol}...")

            try:
                predictions = self.predict(symbol, n_hours=n_hours)
                results[symbol] = predictions

                if self.verbose:
                    print(f"[PREDICTOR] [OK] {symbol} prediction complete")

            except Exception as e:
                print(f"[PREDICTOR] [X] Failed to predict {symbol}: {e}")
                results[symbol] = None

        if self.verbose:
            successful = sum(1 for v in results.values() if v is not None)
            print(f"\n[PREDICTOR] [OK] Batch prediction complete: {successful}/{len(symbols)} successful")

        return results

    def export_prediction(
        self,
        predictions: Dict[str, np.ndarray],
        output_path: str
    ):
        """
        Export predictions to CSV

        Args:
            predictions: Prediction dictionary
            output_path: Output CSV path
        """
        if self.verbose:
            print(f"\n[PREDICTOR] Exporting predictions to {output_path}")

        try:
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': predictions['timestamps'],
                'median_prediction': predictions['median'],
                'lower_95': predictions.get('lower_95', []),
                'upper_95': predictions.get('upper_95', []),
                'lower_80': predictions.get('lower_80', []),
                'upper_80': predictions.get('upper_80', []),
            })

            # Save to CSV
            df.to_csv(output_path, index=False)

            if self.verbose:
                print(f"[PREDICTOR] [OK] Predictions exported")
                print(f"[PREDICTOR]   - Rows: {len(df)}")
                print(f"[PREDICTOR]   - File: {output_path}")

        except Exception as e:
            print(f"[PREDICTOR] [X] Export failed: {e}")
            raise


if __name__ == "__main__":
    # Test predictor (requires trained model)
    print("Testing CryptoPredictor")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("PyTorch not installed. Skipping test.")
    else:
        print("\n[TEST] To test predictor, you need:")
        print("[TEST]   1. Trained model checkpoint")
        print("[TEST]   2. Fitted scaler file")
        print("[TEST]   3. Dataset with recent data")
        print("\n[TEST] Example usage:")
        print("""
predictor = CryptoPredictor(
    model_path='models/production/best_model.ckpt',
    scaler_path='models/production/scalers.pkl',
    dataset_dir='dataset'
)

# Generate 10-hour prediction
predictions = predictor.predict('ETHUSDT', n_hours=10)

# Export to CSV
predictor.export_prediction(predictions, 'eth_prediction.csv')
        """)
