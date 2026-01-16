"""
RSI (Relative Strength Index) Trading Strategy

Generates BUY signals when RSI indicates oversold conditions,
and SELL signals when RSI indicates overbought conditions.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeSignal, StrategyConfig
from .registry import register_strategy


@register_strategy
class RSIStrategy(BaseStrategy):
    """
    RSI Oversold/Overbought Strategy

    Signal Logic:
    - BUY when RSI < oversold_threshold (default: 30)
    - SELL when RSI > overbought_threshold (default: 70)
    - HOLD otherwise

    Confidence is based on how far RSI is from the threshold.
    """

    @property
    def name(self) -> str:
        return "rsi"

    @property
    def description(self) -> str:
        return "RSI Oversold/Overbought Strategy - Buy when RSI < 30, Sell when RSI > 70"

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config)
        params = self.config.params
        self.oversold = params.get('oversold_threshold', 30)
        self.overbought = params.get('overbought_threshold', 70)

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'oversold_threshold': 30,
            'overbought_threshold': 70,
        }

    def get_required_features(self) -> List[str]:
        return ['close', 'rsi']

    def _detect_and_convert_rsi(self, data: pd.DataFrame) -> np.ndarray:
        """
        Detect if RSI is normalized or raw and convert to 0-100 scale.

        - Raw RSI: values in 0-100 range
        - Normalized RSI: values typically in -3 to +3 range (mean=0, std=1)
        """
        rsi_values = data['rsi'].dropna()
        if len(rsi_values) == 0:
            return np.full(len(data), 50.0)

        rsi_min = rsi_values.min()
        rsi_max = rsi_values.max()
        rsi_mean = rsi_values.mean()

        # If values are in 0-100 range (or close), it's raw RSI
        if rsi_min >= -5 and rsi_max <= 105 and rsi_max > 50:
            # Raw RSI - use as is, just clip
            return data['rsi'].clip(0, 100).fillna(50).values
        else:
            # Normalized RSI (mean ~0, std ~1) - denormalize
            # RSI typically has std of ~15-20 around mean of ~50
            rsi_denorm = rsi_values * 17 + 50
            result = data['rsi'] * 17 + 50
            return result.clip(0, 100).fillna(50).values

    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate RSI-based trading signals"""
        signals = []

        # Detect and convert RSI to 0-100 scale
        rsi_values = self._detect_and_convert_rsi(data)

        for idx in range(len(data)):
            row = data.iloc[idx]
            rsi = rsi_values[idx]

            price = self.get_indicator_value(row, 'close', 0)
            timestamp = row.get('datetime', data.index[idx] if hasattr(data.index, '__getitem__') else None)

            # Generate signal
            if rsi < self.oversold:
                action = 'BUY'
                # Confidence increases as RSI gets further below oversold level
                confidence = min((self.oversold - rsi) / self.oversold, 1.0)
            elif rsi > self.overbought:
                action = 'SELL'
                # Confidence increases as RSI gets further above overbought level
                confidence = min((rsi - self.overbought) / (100 - self.overbought), 1.0)
            else:
                action = 'HOLD'
                confidence = 0.0

            signals.append(TradeSignal(
                action=action,
                confidence=confidence,
                price=price,
                timestamp=timestamp,
                indicator_values={'rsi': rsi},
                metadata={'oversold': self.oversold, 'overbought': self.overbought}
            ))

        return signals
