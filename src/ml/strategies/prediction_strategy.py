"""
AI Prediction-based Trading Strategy

Uses machine learning model predictions to generate trading signals.
This integrates with the existing TFT/LSTM model predictions.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeSignal, StrategyConfig
from .registry import register_strategy


@register_strategy
class PredictionStrategy(BaseStrategy):
    """
    AI Model Prediction Strategy

    Signal Logic:
    - BUY when predicted return > confidence_threshold
    - SELL when predicted return < -confidence_threshold
    - HOLD otherwise

    This strategy expects a 'prediction' column in the data
    containing model predictions (normalized returns).
    """

    @property
    def name(self) -> str:
        return "prediction"

    @property
    def description(self) -> str:
        return "AI Prediction Strategy - Uses ML model predictions for trading signals"

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config)
        params = self.config.params
        self.confidence_threshold = params.get('confidence_threshold', 0.002)  # 0.2% predicted move
        self.use_direction_prob = params.get('use_direction_prob', False)
        self.prob_threshold = params.get('prob_threshold', 0.6)

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'confidence_threshold': 0.002,  # Minimum predicted return to trigger signal
            'use_direction_prob': False,    # Use direction probability if available
            'prob_threshold': 0.6           # Minimum probability for direction signal
        }

    def get_required_features(self) -> List[str]:
        return ['close', 'prediction']

    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate signals from model predictions"""
        signals = []

        # Check if prediction column exists
        has_prediction = 'prediction' in data.columns
        has_direction_prob = 'direction_prob' in data.columns

        for idx in range(len(data)):
            row = data.iloc[idx]
            price = self.get_indicator_value(row, 'close', 0)
            timestamp = row.get('datetime', None)

            action = 'HOLD'
            confidence = 0.0
            pred_return = 0.0

            if has_prediction:
                pred_return = self.get_indicator_value(row, 'prediction', 0)

                # Generate signal based on predicted return
                if pred_return > self.confidence_threshold:
                    action = 'BUY'
                    confidence = min(abs(pred_return) / self.confidence_threshold / 10, 1.0)
                elif pred_return < -self.confidence_threshold:
                    action = 'SELL'
                    confidence = min(abs(pred_return) / self.confidence_threshold / 10, 1.0)

            # Alternative: use direction probability if available
            if self.use_direction_prob and has_direction_prob:
                dir_prob = self.get_indicator_value(row, 'direction_prob', 0.5)

                if dir_prob > self.prob_threshold:
                    action = 'BUY'
                    confidence = (dir_prob - 0.5) * 2  # Scale 0.5-1.0 to 0-1
                elif dir_prob < (1 - self.prob_threshold):
                    action = 'SELL'
                    confidence = (0.5 - dir_prob) * 2

            signals.append(TradeSignal(
                action=action,
                confidence=confidence,
                price=price,
                timestamp=timestamp,
                indicator_values={
                    'prediction': pred_return,
                    'direction_prob': row.get('direction_prob', 0.5) if has_direction_prob else 0.5
                },
                metadata={
                    'threshold': self.confidence_threshold,
                    'model_based': True
                }
            ))

        return signals

    def generate_signals_from_predictions(self,
                                         predictions: np.ndarray,
                                         prices: np.ndarray,
                                         timestamps: Optional[pd.DatetimeIndex] = None) -> List[TradeSignal]:
        """
        Generate signals directly from model prediction arrays.

        This is a convenience method for integration with the backtester.

        Args:
            predictions: Array of predicted returns
            prices: Array of current prices
            timestamps: Optional array of timestamps

        Returns:
            List of TradeSignal objects
        """
        signals = []

        for idx in range(len(predictions)):
            pred = predictions[idx] if np.isscalar(predictions[idx]) else predictions[idx].mean()
            price = prices[idx]
            timestamp = timestamps[idx] if timestamps is not None else None

            action = 'HOLD'
            confidence = 0.0

            if pred > self.confidence_threshold:
                action = 'BUY'
                confidence = min(abs(pred) / self.confidence_threshold / 10, 1.0)
            elif pred < -self.confidence_threshold:
                action = 'SELL'
                confidence = min(abs(pred) / self.confidence_threshold / 10, 1.0)

            signals.append(TradeSignal(
                action=action,
                confidence=confidence,
                price=float(price),
                timestamp=timestamp,
                indicator_values={'prediction': float(pred)},
                metadata={'model_based': True}
            ))

        return signals
