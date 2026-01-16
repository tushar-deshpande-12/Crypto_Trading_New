"""
MACD (Moving Average Convergence Divergence) Trading Strategy

Generates signals based on MACD line crossing the signal line.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from .base import BaseStrategy, TradeSignal, StrategyConfig
from .registry import register_strategy


@register_strategy
class MACDStrategy(BaseStrategy):
    """
    MACD Crossover Strategy

    Signal Logic:
    - BUY when MACD crosses above Signal line (bullish crossover)
    - SELL when MACD crosses below Signal line (bearish crossover)
    - HOLD otherwise

    Confidence is based on the magnitude of the MACD-Signal difference.
    """

    @property
    def name(self) -> str:
        return "macd"

    @property
    def description(self) -> str:
        return "MACD Crossover Strategy - Buy on bullish crossover, Sell on bearish crossover"

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config)
        params = self.config.params
        self.min_crossover_strength = params.get('min_crossover_strength', 0.001)
        self.use_histogram = params.get('use_histogram', True)

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'min_crossover_strength': 0.001,  # Minimum MACD-Signal diff to trigger
            'use_histogram': True  # Use MACD histogram for signal strength
        }

    def get_required_features(self) -> List[str]:
        return ['close', 'macd', 'macd_signal']

    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate MACD crossover signals"""
        signals = []

        prev_macd = None
        prev_signal = None

        for idx in range(len(data)):
            row = data.iloc[idx]

            macd = self.get_indicator_value(row, 'macd', 0)
            signal = self.get_indicator_value(row, 'macd_signal', 0)
            price = self.get_indicator_value(row, 'close', 0)
            timestamp = row.get('datetime', None)

            # Calculate histogram (MACD - Signal)
            histogram = macd - signal

            # Detect crossover
            action = 'HOLD'
            confidence = 0.0

            if prev_macd is not None and prev_signal is not None:
                prev_histogram = prev_macd - prev_signal

                # Bullish crossover: MACD crosses above signal
                if prev_histogram <= 0 and histogram > 0:
                    action = 'BUY'
                    confidence = min(abs(histogram) * 10, 1.0)

                # Bearish crossover: MACD crosses below signal
                elif prev_histogram >= 0 and histogram < 0:
                    action = 'SELL'
                    confidence = min(abs(histogram) * 10, 1.0)

                # Continuing trend signals (weaker)
                elif histogram > self.min_crossover_strength:
                    if histogram > prev_histogram:  # Strengthening bullish
                        action = 'BUY'
                        confidence = min(abs(histogram) * 5, 0.7)  # Lower confidence for continuation

                elif histogram < -self.min_crossover_strength:
                    if histogram < prev_histogram:  # Strengthening bearish
                        action = 'SELL'
                        confidence = min(abs(histogram) * 5, 0.7)

            signals.append(TradeSignal(
                action=action,
                confidence=confidence,
                price=price,
                timestamp=timestamp,
                indicator_values={
                    'macd': macd,
                    'macd_signal': signal,
                    'histogram': histogram
                },
                metadata={'crossover_type': 'bullish' if action == 'BUY' else ('bearish' if action == 'SELL' else 'none')}
            ))

            prev_macd = macd
            prev_signal = signal

        return signals
