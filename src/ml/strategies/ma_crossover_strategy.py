"""
Moving Average Crossover Trading Strategy

Generates signals based on short-term MA crossing long-term MA.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from .base import BaseStrategy, TradeSignal, StrategyConfig
from .registry import register_strategy


@register_strategy
class MACrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy

    Signal Logic:
    - BUY when short MA crosses above long MA (Golden Cross)
    - SELL when short MA crosses below long MA (Death Cross)
    - Uses sma10_sma20_ratio indicator

    This is a trend-following strategy.
    """

    @property
    def name(self) -> str:
        return "ma_crossover"

    @property
    def description(self) -> str:
        return "MA Crossover Strategy - Buy on Golden Cross, Sell on Death Cross"

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config)
        params = self.config.params
        self.crossover_threshold = params.get('crossover_threshold', 1.0)
        self.trend_strength_threshold = params.get('trend_strength_threshold', 0.01)

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'crossover_threshold': 1.0,  # Ratio of 1.0 means equal MAs
            'trend_strength_threshold': 0.01  # Minimum ratio difference to signal
        }

    def get_required_features(self) -> List[str]:
        return ['close', 'sma10_sma20_ratio', 'price_sma10_ratio', 'price_sma20_ratio']

    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate MA crossover signals"""
        signals = []

        prev_ratio = None

        for idx in range(len(data)):
            row = data.iloc[idx]

            # sma10_sma20_ratio > 1 means short MA above long MA (bullish)
            # After normalization, values center around 0
            ma_ratio = self.get_indicator_value(row, 'sma10_sma20_ratio', 1.0)
            price_sma10 = self.get_indicator_value(row, 'price_sma10_ratio', 1.0)
            price_sma20 = self.get_indicator_value(row, 'price_sma20_ratio', 1.0)
            price = self.get_indicator_value(row, 'close', 0)
            timestamp = row.get('datetime', None)

            action = 'HOLD'
            confidence = 0.0

            # Normalized ratio: positive means short > long (bullish)
            # Detect crossover
            if prev_ratio is not None:
                # Golden Cross: ratio crosses above threshold
                if prev_ratio < 0 and ma_ratio > 0:
                    action = 'BUY'
                    confidence = min(abs(ma_ratio) * 5, 1.0)

                # Death Cross: ratio crosses below threshold
                elif prev_ratio > 0 and ma_ratio < 0:
                    action = 'SELL'
                    confidence = min(abs(ma_ratio) * 5, 1.0)

                # Trend continuation (weaker signals)
                elif ma_ratio > self.trend_strength_threshold:
                    # In bullish trend, price above both MAs strengthens signal
                    if price_sma10 > 0 and price_sma20 > 0:
                        action = 'BUY'
                        confidence = min(ma_ratio * 3, 0.6)

                elif ma_ratio < -self.trend_strength_threshold:
                    # In bearish trend, price below both MAs strengthens signal
                    if price_sma10 < 0 and price_sma20 < 0:
                        action = 'SELL'
                        confidence = min(abs(ma_ratio) * 3, 0.6)

            signals.append(TradeSignal(
                action=action,
                confidence=confidence,
                price=price,
                timestamp=timestamp,
                indicator_values={
                    'sma10_sma20_ratio': ma_ratio,
                    'price_sma10_ratio': price_sma10,
                    'price_sma20_ratio': price_sma20
                },
                metadata={
                    'trend': 'bullish' if ma_ratio > 0 else 'bearish',
                    'crossover': prev_ratio is not None and (
                        (prev_ratio < 0 and ma_ratio > 0) or
                        (prev_ratio > 0 and ma_ratio < 0)
                    )
                }
            ))

            prev_ratio = ma_ratio

        return signals
