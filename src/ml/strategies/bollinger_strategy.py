"""
Bollinger Bands Trading Strategy

Generates signals based on price position relative to Bollinger Bands.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from .base import BaseStrategy, TradeSignal, StrategyConfig
from .registry import register_strategy


@register_strategy
class BollingerStrategy(BaseStrategy):
    """
    Bollinger Bands Mean Reversion Strategy

    Signal Logic:
    - BUY when price touches or crosses below lower band (oversold)
    - SELL when price touches or crosses above upper band (overbought)
    - Uses bb_position indicator: -1 (lower) to +1 (upper)

    This is a mean reversion strategy, expecting price to return to the middle band.
    """

    @property
    def name(self) -> str:
        return "bollinger"

    @property
    def description(self) -> str:
        return "Bollinger Bands Strategy - Buy at lower band, Sell at upper band"

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config)
        params = self.config.params
        self.lower_threshold = params.get('lower_threshold', -0.8)  # bb_position threshold for buy
        self.upper_threshold = params.get('upper_threshold', 0.8)   # bb_position threshold for sell
        self.use_width = params.get('use_width', True)  # Consider band width for confidence

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'lower_threshold': -0.8,  # Buy when bb_position < -0.8
            'upper_threshold': 0.8,   # Sell when bb_position > 0.8
            'use_width': True         # Higher confidence when bands are wide
        }

    def get_required_features(self) -> List[str]:
        return ['close', 'bb_position', 'bb_width']

    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate Bollinger Bands mean reversion signals"""
        signals = []

        for idx in range(len(data)):
            row = data.iloc[idx]

            # bb_position ranges from -1 (at lower band) to +1 (at upper band)
            # After normalization, it's around mean=0, std=1
            bb_pos = self.get_indicator_value(row, 'bb_position', 0)
            bb_width = self.get_indicator_value(row, 'bb_width', 0.02)
            price = self.get_indicator_value(row, 'close', 0)
            timestamp = row.get('datetime', None)

            action = 'HOLD'
            confidence = 0.0

            # Price near or below lower band - potential buy
            if bb_pos < self.lower_threshold:
                action = 'BUY'
                # Confidence based on how far below threshold
                confidence = min(abs(bb_pos - self.lower_threshold) / 0.5, 1.0)
                # Adjust for band width (wider bands = stronger signal)
                if self.use_width and bb_width > 0:
                    width_factor = min(bb_width * 20, 1.5)  # Normalized width factor
                    confidence = min(confidence * width_factor, 1.0)

            # Price near or above upper band - potential sell
            elif bb_pos > self.upper_threshold:
                action = 'SELL'
                confidence = min(abs(bb_pos - self.upper_threshold) / 0.5, 1.0)
                if self.use_width and bb_width > 0:
                    width_factor = min(bb_width * 20, 1.5)
                    confidence = min(confidence * width_factor, 1.0)

            signals.append(TradeSignal(
                action=action,
                confidence=confidence,
                price=price,
                timestamp=timestamp,
                indicator_values={
                    'bb_position': bb_pos,
                    'bb_width': bb_width
                },
                metadata={
                    'band_zone': 'lower' if bb_pos < -0.5 else ('upper' if bb_pos > 0.5 else 'middle')
                }
            ))

        return signals
