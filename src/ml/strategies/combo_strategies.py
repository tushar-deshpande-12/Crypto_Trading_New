"""
Combo Strategies - Famous indicator combinations for trading signals

These strategies combine multiple indicators for more robust signals:
- RSI + MACD: Momentum confirmation
- Bollinger + RSI: Volatility + momentum
- MACD + MA: Trend following with momentum
- Stochastic + RSI: Double momentum filter
- Triple EMA: Classic trend following
- ADX + MACD: Trend strength with momentum
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeSignal, StrategyConfig
from .registry import register_strategy


@register_strategy
class RSIMACDComboStrategy(BaseStrategy):
    """
    RSI + MACD Combo Strategy

    Combines RSI oversold/overbought with MACD crossovers for confirmation.
    - BUY: RSI < 35 AND MACD crosses above signal
    - SELL: RSI > 65 AND MACD crosses below signal

    This reduces false signals by requiring both indicators to agree.
    """

    @property
    def name(self) -> str:
        return "rsi_macd"

    @property
    def description(self) -> str:
        return "RSI + MACD Combo - Momentum confirmation strategy"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'rsi_oversold': 35,
            'rsi_overbought': 65,
            'rsi_period': 14,
        }

    def get_required_features(self) -> List[str]:
        return ['rsi', 'macd', 'macd_signal']

    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        signals = []
        params = self.config.params

        rsi_oversold = params.get('rsi_oversold', 35)
        rsi_overbought = params.get('rsi_overbought', 65)

        for i in range(1, len(data)):
            row = data.iloc[i]
            prev_row = data.iloc[i - 1]

            rsi = row.get('rsi', 50)
            macd = row.get('macd', 0)
            macd_signal = row.get('macd_signal', 0)
            prev_macd = prev_row.get('macd', 0)
            prev_macd_signal = prev_row.get('macd_signal', 0)

            price = row.get('close', 0)
            timestamp = row.get('datetime', data.index[i] if hasattr(data.index, '__getitem__') else None)

            action = 'HOLD'
            confidence = 0.0

            # MACD crossover detection
            macd_cross_up = prev_macd <= prev_macd_signal and macd > macd_signal
            macd_cross_down = prev_macd >= prev_macd_signal and macd < macd_signal

            # BUY: RSI oversold + MACD bullish crossover
            if rsi < rsi_oversold and macd_cross_up:
                action = 'BUY'
                # Confidence based on how oversold RSI is
                confidence = min((rsi_oversold - rsi) / rsi_oversold + 0.3, 1.0)

            # SELL: RSI overbought + MACD bearish crossover
            elif rsi > rsi_overbought and macd_cross_down:
                action = 'SELL'
                confidence = min((rsi - rsi_overbought) / (100 - rsi_overbought) + 0.3, 1.0)

            signals.append(TradeSignal(
                action=action,
                confidence=confidence,
                price=price,
                timestamp=timestamp,
                indicator_values={'rsi': rsi, 'macd': macd, 'macd_signal': macd_signal}
            ))

        return signals


@register_strategy
class BollingerRSIComboStrategy(BaseStrategy):
    """
    Bollinger Bands + RSI Combo Strategy

    Combines Bollinger Band position with RSI for volatility-adjusted signals.
    - BUY: Price at lower band AND RSI < 35 (oversold at support)
    - SELL: Price at upper band AND RSI > 65 (overbought at resistance)

    Best for range-bound markets with mean reversion tendencies.
    """

    @property
    def name(self) -> str:
        return "bb_rsi"

    @property
    def description(self) -> str:
        return "Bollinger + RSI Combo - Volatility with momentum"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'bb_lower_threshold': -0.8,  # Position in band (-1 to 1)
            'bb_upper_threshold': 0.8,
            'rsi_oversold': 35,
            'rsi_overbought': 65,
        }

    def get_required_features(self) -> List[str]:
        return ['bb_position', 'rsi', 'close', 'bb_upper', 'bb_lower']

    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        signals = []
        params = self.config.params

        bb_lower = params.get('bb_lower_threshold', -0.8)
        bb_upper = params.get('bb_upper_threshold', 0.8)
        rsi_oversold = params.get('rsi_oversold', 35)
        rsi_overbought = params.get('rsi_overbought', 65)

        for i in range(len(data)):
            row = data.iloc[i]

            bb_pos = row.get('bb_position', 0)
            rsi = row.get('rsi', 50)
            price = row.get('close', 0)
            timestamp = row.get('datetime', data.index[i] if hasattr(data.index, '__getitem__') else None)

            action = 'HOLD'
            confidence = 0.0

            # BUY: Near lower band AND RSI oversold
            if bb_pos <= bb_lower and rsi < rsi_oversold:
                action = 'BUY'
                # Stronger signal when both are extreme
                bb_strength = abs(bb_pos - bb_lower) / 0.2
                rsi_strength = (rsi_oversold - rsi) / rsi_oversold
                confidence = min((bb_strength + rsi_strength) / 2 + 0.4, 1.0)

            # SELL: Near upper band AND RSI overbought
            elif bb_pos >= bb_upper and rsi > rsi_overbought:
                action = 'SELL'
                bb_strength = (bb_pos - bb_upper) / 0.2
                rsi_strength = (rsi - rsi_overbought) / (100 - rsi_overbought)
                confidence = min((bb_strength + rsi_strength) / 2 + 0.4, 1.0)

            signals.append(TradeSignal(
                action=action,
                confidence=confidence,
                price=price,
                timestamp=timestamp,
                indicator_values={'bb_position': bb_pos, 'rsi': rsi}
            ))

        return signals


@register_strategy
class MACDMAComboStrategy(BaseStrategy):
    """
    MACD + Moving Average Combo Strategy

    Combines MACD with MA crossover for trend-following with momentum.
    - BUY: MACD bullish crossover AND price above SMA20
    - SELL: MACD bearish crossover AND price below SMA20

    Strong trend-following strategy for trending markets.
    """

    @property
    def name(self) -> str:
        return "macd_ma"

    @property
    def description(self) -> str:
        return "MACD + MA Combo - Trend following with momentum"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'ma_period': 20,
        }

    def get_required_features(self) -> List[str]:
        return ['macd', 'macd_signal', 'sma_20', 'close']

    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        signals = []

        for i in range(1, len(data)):
            row = data.iloc[i]
            prev_row = data.iloc[i - 1]

            macd = row.get('macd', 0)
            macd_signal = row.get('macd_signal', 0)
            prev_macd = prev_row.get('macd', 0)
            prev_macd_signal = prev_row.get('macd_signal', 0)
            sma = row.get('sma_20', row.get('close', 0))
            price = row.get('close', 0)
            timestamp = row.get('datetime', data.index[i] if hasattr(data.index, '__getitem__') else None)

            action = 'HOLD'
            confidence = 0.0

            # MACD crossover detection
            macd_cross_up = prev_macd <= prev_macd_signal and macd > macd_signal
            macd_cross_down = prev_macd >= prev_macd_signal and macd < macd_signal

            # BUY: MACD bullish cross AND price above MA
            if macd_cross_up and price > sma:
                action = 'BUY'
                # Confidence based on MACD histogram strength and price distance from MA
                histogram = macd - macd_signal
                ma_distance = (price - sma) / sma * 100
                confidence = min(0.5 + abs(histogram) * 10 + ma_distance / 5, 1.0)

            # SELL: MACD bearish cross AND price below MA
            elif macd_cross_down and price < sma:
                action = 'SELL'
                histogram = macd_signal - macd
                ma_distance = (sma - price) / sma * 100
                confidence = min(0.5 + abs(histogram) * 10 + ma_distance / 5, 1.0)

            signals.append(TradeSignal(
                action=action,
                confidence=confidence,
                price=price,
                timestamp=timestamp,
                indicator_values={'macd': macd, 'macd_signal': macd_signal, 'sma_20': sma}
            ))

        return signals


@register_strategy
class StochRSIComboStrategy(BaseStrategy):
    """
    Stochastic + RSI Combo Strategy

    Uses both oscillators for double momentum confirmation.
    - BUY: Stochastic %K crosses %D in oversold AND RSI < 40
    - SELL: Stochastic %K crosses %D in overbought AND RSI > 60

    Reduces whipsaws by requiring agreement between oscillators.
    """

    @property
    def name(self) -> str:
        return "stoch_rsi"

    @property
    def description(self) -> str:
        return "Stochastic + RSI - Double momentum filter"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'stoch_oversold': 20,
            'stoch_overbought': 80,
            'rsi_oversold': 40,
            'rsi_overbought': 60,
        }

    def get_required_features(self) -> List[str]:
        return ['stoch_k', 'stoch_d', 'rsi']

    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        signals = []
        params = self.config.params

        stoch_oversold = params.get('stoch_oversold', 20)
        stoch_overbought = params.get('stoch_overbought', 80)
        rsi_oversold = params.get('rsi_oversold', 40)
        rsi_overbought = params.get('rsi_overbought', 60)

        for i in range(1, len(data)):
            row = data.iloc[i]
            prev_row = data.iloc[i - 1]

            stoch_k = row.get('stoch_k', 50)
            stoch_d = row.get('stoch_d', 50)
            prev_k = prev_row.get('stoch_k', 50)
            prev_d = prev_row.get('stoch_d', 50)
            rsi = row.get('rsi', 50)
            price = row.get('close', 0)
            timestamp = row.get('datetime', data.index[i] if hasattr(data.index, '__getitem__') else None)

            action = 'HOLD'
            confidence = 0.0

            # Stochastic crossover in zones
            stoch_cross_up = prev_k <= prev_d and stoch_k > stoch_d
            stoch_cross_down = prev_k >= prev_d and stoch_k < stoch_d
            in_oversold = stoch_k < stoch_oversold
            in_overbought = stoch_k > stoch_overbought

            # BUY: Stochastic bullish cross in oversold AND RSI agrees
            if stoch_cross_up and in_oversold and rsi < rsi_oversold:
                action = 'BUY'
                stoch_strength = (stoch_oversold - stoch_k) / stoch_oversold
                rsi_strength = (rsi_oversold - rsi) / rsi_oversold
                confidence = min((stoch_strength + rsi_strength) / 2 + 0.5, 1.0)

            # SELL: Stochastic bearish cross in overbought AND RSI agrees
            elif stoch_cross_down and in_overbought and rsi > rsi_overbought:
                action = 'SELL'
                stoch_strength = (stoch_k - stoch_overbought) / (100 - stoch_overbought)
                rsi_strength = (rsi - rsi_overbought) / (100 - rsi_overbought)
                confidence = min((stoch_strength + rsi_strength) / 2 + 0.5, 1.0)

            signals.append(TradeSignal(
                action=action,
                confidence=confidence,
                price=price,
                timestamp=timestamp,
                indicator_values={'stoch_k': stoch_k, 'stoch_d': stoch_d, 'rsi': rsi}
            ))

        return signals


@register_strategy
class TripleEMAStrategy(BaseStrategy):
    """
    Triple EMA Crossover Strategy

    Classic trend-following using 3 EMAs (short, medium, long).
    - BUY: EMA5 > EMA13 > EMA50 (all aligned bullish)
    - SELL: EMA5 < EMA13 < EMA50 (all aligned bearish)

    Best for capturing strong trends and avoiding choppy markets.
    """

    @property
    def name(self) -> str:
        return "triple_ema"

    @property
    def description(self) -> str:
        return "Triple EMA Crossover - Classic trend following"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'ema_short': 5,
            'ema_medium': 13,
            'ema_long': 50,
        }

    def get_required_features(self) -> List[str]:
        return ['close']  # We'll calculate EMAs if not present

    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        signals = []
        df = data.copy()

        # Calculate EMAs if not present
        close = df['close']
        if 'ema_5' not in df.columns:
            df['ema_5'] = close.ewm(span=5, adjust=False).mean()
        if 'ema_13' not in df.columns:
            df['ema_13'] = close.ewm(span=13, adjust=False).mean()
        if 'ema_50' not in df.columns:
            df['ema_50'] = close.ewm(span=50, adjust=False).mean()

        prev_aligned_bull = False
        prev_aligned_bear = False

        for i in range(1, len(df)):
            row = df.iloc[i]

            ema5 = row.get('ema_5', row['close'])
            ema13 = row.get('ema_13', row['close'])
            ema50 = row.get('ema_50', row['close'])
            price = row.get('close', 0)
            timestamp = row.get('datetime', df.index[i] if hasattr(df.index, '__getitem__') else None)

            action = 'HOLD'
            confidence = 0.0

            # Check alignment
            aligned_bullish = ema5 > ema13 > ema50
            aligned_bearish = ema5 < ema13 < ema50

            # BUY: EMAs just aligned bullish
            if aligned_bullish and not prev_aligned_bull:
                action = 'BUY'
                # Confidence based on spread between EMAs
                spread = (ema5 - ema50) / ema50 * 100
                confidence = min(0.5 + spread / 5, 1.0)

            # SELL: EMAs just aligned bearish
            elif aligned_bearish and not prev_aligned_bear:
                action = 'SELL'
                spread = (ema50 - ema5) / ema50 * 100
                confidence = min(0.5 + spread / 5, 1.0)

            prev_aligned_bull = aligned_bullish
            prev_aligned_bear = aligned_bearish

            signals.append(TradeSignal(
                action=action,
                confidence=confidence,
                price=price,
                timestamp=timestamp,
                indicator_values={'ema_5': ema5, 'ema_13': ema13, 'ema_50': ema50}
            ))

        return signals


@register_strategy
class ADXMACDStrategy(BaseStrategy):
    """
    ADX + MACD Strategy

    Combines trend strength (ADX) with momentum (MACD).
    - BUY: ADX > 25 (trending) AND MACD bullish crossover
    - SELL: ADX > 25 (trending) AND MACD bearish crossover

    Only trades when there's a strong trend, avoiding choppy markets.
    """

    @property
    def name(self) -> str:
        return "adx_macd"

    @property
    def description(self) -> str:
        return "ADX + MACD - Trend strength with momentum"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'adx_threshold': 25,  # Minimum ADX for strong trend
        }

    def get_required_features(self) -> List[str]:
        return ['adx', 'macd', 'macd_signal']

    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        signals = []
        params = self.config.params

        adx_threshold = params.get('adx_threshold', 25)

        for i in range(1, len(data)):
            row = data.iloc[i]
            prev_row = data.iloc[i - 1]

            adx = row.get('adx', 0)
            macd = row.get('macd', 0)
            macd_signal = row.get('macd_signal', 0)
            prev_macd = prev_row.get('macd', 0)
            prev_macd_signal = prev_row.get('macd_signal', 0)
            price = row.get('close', 0)
            timestamp = row.get('datetime', data.index[i] if hasattr(data.index, '__getitem__') else None)

            action = 'HOLD'
            confidence = 0.0

            # MACD crossover
            macd_cross_up = prev_macd <= prev_macd_signal and macd > macd_signal
            macd_cross_down = prev_macd >= prev_macd_signal and macd < macd_signal

            # Only trade in strong trends
            strong_trend = adx > adx_threshold

            # BUY: Strong trend + MACD bullish cross
            if strong_trend and macd_cross_up:
                action = 'BUY'
                # Confidence based on ADX strength and MACD histogram
                adx_strength = (adx - adx_threshold) / (100 - adx_threshold)
                histogram = abs(macd - macd_signal)
                confidence = min(0.4 + adx_strength * 0.3 + histogram * 10, 1.0)

            # SELL: Strong trend + MACD bearish cross
            elif strong_trend and macd_cross_down:
                action = 'SELL'
                adx_strength = (adx - adx_threshold) / (100 - adx_threshold)
                histogram = abs(macd - macd_signal)
                confidence = min(0.4 + adx_strength * 0.3 + histogram * 10, 1.0)

            signals.append(TradeSignal(
                action=action,
                confidence=confidence,
                price=price,
                timestamp=timestamp,
                indicator_values={'adx': adx, 'macd': macd, 'macd_signal': macd_signal}
            ))

        return signals
