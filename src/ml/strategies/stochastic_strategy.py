"""
Stochastic Oscillator Trading Strategy

Generates signals based on %K and %D crossovers in oversold/overbought zones.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeSignal, StrategyConfig
from .registry import register_strategy


@register_strategy
class StochasticStrategy(BaseStrategy):
    """
    Stochastic Oscillator Strategy

    Signal Logic:
    - BUY when %K crosses above %D in oversold zone (below 20)
    - SELL when %K crosses below %D in overbought zone (above 80)

    This combines momentum and oversold/overbought signals.
    """

    @property
    def name(self) -> str:
        return "stochastic"

    @property
    def description(self) -> str:
        return "Stochastic Strategy - Buy on %K/%D crossover in oversold, Sell in overbought"

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config)
        params = self.config.params
        self.oversold = params.get('oversold_threshold', 20)
        self.overbought = params.get('overbought_threshold', 80)
        self.require_crossover = params.get('require_crossover', True)

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'oversold_threshold': 20,
            'overbought_threshold': 80,
            'require_crossover': True  # Require %K/%D crossover, not just zone
        }

    def get_required_features(self) -> List[str]:
        return ['close', 'stoch_k', 'stoch_d']

    def _detect_and_convert_stochastic(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect if Stochastic values are normalized or raw and convert to 0-100 scale.

        - Raw Stochastic: values in 0-100 range
        - Normalized Stochastic: values typically in -3 to +3 range (mean=0, std=1)

        Returns:
            Tuple of (stoch_k, stoch_d) arrays in 0-100 scale
        """
        k_values = data['stoch_k'].dropna()
        d_values = data['stoch_d'].dropna()

        if len(k_values) == 0:
            return np.full(len(data), 50.0), np.full(len(data), 50.0)

        k_min, k_max = k_values.min(), k_values.max()

        # If values are in 0-100 range (or close), it's raw Stochastic
        if k_min >= -5 and k_max <= 105 and k_max > 50:
            # Raw Stochastic - use as is, just clip
            stoch_k = data['stoch_k'].clip(0, 100).fillna(50).values
            stoch_d = data['stoch_d'].clip(0, 100).fillna(50).values
        else:
            # Normalized Stochastic (mean ~0, std ~1) - denormalize
            # Stochastic typically has std of ~25-30 around mean of ~50
            stoch_k = (data['stoch_k'] * 28 + 50).clip(0, 100).fillna(50).values
            stoch_d = (data['stoch_d'] * 28 + 50).clip(0, 100).fillna(50).values

        return stoch_k, stoch_d

    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate stochastic crossover signals"""
        signals = []

        # Detect and convert to 0-100 scale
        stoch_k_arr, stoch_d_arr = self._detect_and_convert_stochastic(data)

        prev_k = None
        prev_d = None

        for idx in range(len(data)):
            row = data.iloc[idx]

            stoch_k = stoch_k_arr[idx]
            stoch_d = stoch_d_arr[idx]

            price = self.get_indicator_value(row, 'close', 0)
            timestamp = row.get('datetime', data.index[idx] if hasattr(data.index, '__getitem__') else None)

            action = 'HOLD'
            confidence = 0.0

            if prev_k is not None and prev_d is not None:
                # Check for crossover
                k_crossed_above_d = prev_k <= prev_d and stoch_k > stoch_d
                k_crossed_below_d = prev_k >= prev_d and stoch_k < stoch_d

                # Bullish: %K crosses above %D in oversold zone
                if stoch_k < self.oversold or (stoch_k < 30 and k_crossed_above_d):
                    if k_crossed_above_d or not self.require_crossover:
                        action = 'BUY'
                        # Higher confidence when deeper in oversold
                        zone_factor = (self.oversold - min(stoch_k, self.oversold)) / self.oversold
                        crossover_factor = 0.5 if k_crossed_above_d else 0.2
                        confidence = min(zone_factor + crossover_factor, 1.0)

                # Bearish: %K crosses below %D in overbought zone
                elif stoch_k > self.overbought or (stoch_k > 70 and k_crossed_below_d):
                    if k_crossed_below_d or not self.require_crossover:
                        action = 'SELL'
                        zone_factor = (min(stoch_k, 100) - self.overbought) / (100 - self.overbought)
                        crossover_factor = 0.5 if k_crossed_below_d else 0.2
                        confidence = min(zone_factor + crossover_factor, 1.0)

            signals.append(TradeSignal(
                action=action,
                confidence=confidence,
                price=price,
                timestamp=timestamp,
                indicator_values={
                    'stoch_k': stoch_k,
                    'stoch_d': stoch_d,
                },
                metadata={
                    'zone': 'oversold' if stoch_k < 30 else ('overbought' if stoch_k > 70 else 'neutral'),
                    'k_above_d': stoch_k > stoch_d
                }
            ))

            prev_k = stoch_k
            prev_d = stoch_d

        return signals
