"""
Chart Calculations Module

Provides calculations for:
- Pivot Points (Classic, Fibonacci, Woodie, Camarilla, DeMark)
- Support and Resistance levels (swing highs/lows, volume profile)
- Price levels and zones
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class PivotType(Enum):
    """Types of pivot point calculations."""
    CLASSIC = 'classic'
    FIBONACCI = 'fibonacci'
    WOODIE = 'woodie'
    CAMARILLA = 'camarilla'
    DEMARK = 'demark'


@dataclass
class PivotLevels:
    """Container for pivot point levels."""
    pivot: float
    r1: float
    r2: float
    r3: float
    r4: Optional[float] = None
    s1: float = 0
    s2: float = 0
    s3: float = 0
    s4: Optional[float] = None
    pivot_type: str = 'classic'

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            'P': self.pivot,
            'R1': self.r1,
            'R2': self.r2,
            'R3': self.r3,
            'S1': self.s1,
            'S2': self.s2,
            'S3': self.s3,
        }
        if self.r4 is not None:
            result['R4'] = self.r4
        if self.s4 is not None:
            result['S4'] = self.s4
        return result


@dataclass
class SupportResistanceLevel:
    """A support or resistance level."""
    price: float
    strength: float  # 0-1, based on number of touches
    level_type: str  # 'support' or 'resistance'
    touches: int
    start_idx: int
    end_idx: int


def calculate_pivot_points(high: float, low: float, close: float,
                          open_price: Optional[float] = None,
                          pivot_type: PivotType = PivotType.CLASSIC) -> PivotLevels:
    """
    Calculate pivot points for a single period (typically daily).

    Args:
        high: Period high price
        low: Period low price
        close: Period close price
        open_price: Period open price (required for DeMark)
        pivot_type: Type of pivot calculation

    Returns:
        PivotLevels object with all levels
    """
    if pivot_type == PivotType.CLASSIC:
        # Classic/Standard Pivot Points
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)

        return PivotLevels(pivot=pivot, r1=r1, r2=r2, r3=r3,
                         s1=s1, s2=s2, s3=s3, pivot_type='classic')

    elif pivot_type == PivotType.FIBONACCI:
        # Fibonacci Pivot Points
        pivot = (high + low + close) / 3
        diff = high - low
        r1 = pivot + 0.382 * diff
        r2 = pivot + 0.618 * diff
        r3 = pivot + 1.000 * diff
        s1 = pivot - 0.382 * diff
        s2 = pivot - 0.618 * diff
        s3 = pivot - 1.000 * diff

        return PivotLevels(pivot=pivot, r1=r1, r2=r2, r3=r3,
                         s1=s1, s2=s2, s3=s3, pivot_type='fibonacci')

    elif pivot_type == PivotType.WOODIE:
        # Woodie's Pivot Points (more weight on close)
        pivot = (high + low + 2 * close) / 4
        r1 = 2 * pivot - low
        r2 = pivot + high - low
        r3 = high + 2 * (pivot - low)
        s1 = 2 * pivot - high
        s2 = pivot - high + low
        s3 = low - 2 * (high - pivot)

        return PivotLevels(pivot=pivot, r1=r1, r2=r2, r3=r3,
                         s1=s1, s2=s2, s3=s3, pivot_type='woodie')

    elif pivot_type == PivotType.CAMARILLA:
        # Camarilla Pivot Points (tighter levels)
        pivot = (high + low + close) / 3
        diff = high - low
        r1 = close + diff * 1.1 / 12
        r2 = close + diff * 1.1 / 6
        r3 = close + diff * 1.1 / 4
        r4 = close + diff * 1.1 / 2
        s1 = close - diff * 1.1 / 12
        s2 = close - diff * 1.1 / 6
        s3 = close - diff * 1.1 / 4
        s4 = close - diff * 1.1 / 2

        return PivotLevels(pivot=pivot, r1=r1, r2=r2, r3=r3, r4=r4,
                         s1=s1, s2=s2, s3=s3, s4=s4, pivot_type='camarilla')

    elif pivot_type == PivotType.DEMARK:
        # DeMark Pivot Points
        if open_price is None:
            open_price = close  # Fallback

        if close < open_price:
            x = high + 2 * low + close
        elif close > open_price:
            x = 2 * high + low + close
        else:
            x = high + low + 2 * close

        pivot = x / 4
        r1 = x / 2 - low
        s1 = x / 2 - high

        # DeMark only has 1 support and 1 resistance
        return PivotLevels(pivot=pivot, r1=r1, r2=r1, r3=r1,
                         s1=s1, s2=s1, s3=s1, pivot_type='demark')

    else:
        raise ValueError(f"Unknown pivot type: {pivot_type}")


def calculate_pivot_points_from_df(df: pd.DataFrame,
                                   pivot_type: PivotType = PivotType.CLASSIC,
                                   period: str = 'daily') -> PivotLevels:
    """
    Calculate pivot points from a DataFrame.

    Uses the previous period's OHLC for pivot calculation.

    Args:
        df: DataFrame with 'high', 'low', 'close', 'open' columns
        pivot_type: Type of pivot calculation
        period: 'daily', 'weekly', or 'monthly' for aggregation

    Returns:
        PivotLevels for the next period
    """
    if len(df) == 0:
        return PivotLevels(pivot=0, r1=0, r2=0, r3=0, s1=0, s2=0, s3=0)

    # Resample to get period OHLC
    if period == 'weekly':
        resampled = df.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
    elif period == 'monthly':
        resampled = df.resample('M').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
    else:  # daily
        resampled = df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()

    if len(resampled) == 0:
        # Use raw data if resampling fails
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        open_price = df['open'].iloc[-1] if 'open' in df.columns else close
    else:
        # Use the most recent complete period
        last_period = resampled.iloc[-1]
        high = last_period['high']
        low = last_period['low']
        close = last_period['close']
        open_price = last_period['open']

    return calculate_pivot_points(high, low, close, open_price, pivot_type)


def find_swing_highs_lows(df: pd.DataFrame, lookback: int = 5) -> Tuple[List[int], List[int]]:
    """
    Find swing high and swing low indices.

    A swing high is a high that is higher than the N bars before and after.
    A swing low is a low that is lower than the N bars before and after.

    Args:
        df: DataFrame with 'high' and 'low' columns
        lookback: Number of bars to look before and after

    Returns:
        Tuple of (swing_high_indices, swing_low_indices)
    """
    highs = df['high'].values
    lows = df['low'].values

    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(df) - lookback):
        # Check for swing high
        is_swing_high = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing_high = False
                break
        if is_swing_high:
            swing_highs.append(i)

        # Check for swing low
        is_swing_low = True
        for j in range(1, lookback + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing_low = False
                break
        if is_swing_low:
            swing_lows.append(i)

    return swing_highs, swing_lows


def find_support_resistance_levels(df: pd.DataFrame,
                                   lookback: int = 5,
                                   tolerance_pct: float = 0.5,
                                   min_touches: int = 2) -> List[SupportResistanceLevel]:
    """
    Find support and resistance levels based on swing points.

    Groups nearby swing points to identify strong S/R zones.

    Args:
        df: DataFrame with OHLC data
        lookback: Bars to look for swing points
        tolerance_pct: Percentage tolerance for grouping levels
        min_touches: Minimum touches to be considered a level

    Returns:
        List of SupportResistanceLevel objects
    """
    swing_highs, swing_lows = find_swing_highs_lows(df, lookback)

    # Get prices at swing points
    high_prices = [(idx, df['high'].iloc[idx]) for idx in swing_highs]
    low_prices = [(idx, df['low'].iloc[idx]) for idx in swing_lows]

    levels = []

    # Group resistance levels (swing highs)
    resistance_groups = _group_price_levels(high_prices, tolerance_pct)
    for group in resistance_groups:
        if len(group) >= min_touches:
            avg_price = np.mean([p for _, p in group])
            strength = min(len(group) / 5, 1.0)  # Normalize to 0-1
            indices = [idx for idx, _ in group]
            levels.append(SupportResistanceLevel(
                price=avg_price,
                strength=strength,
                level_type='resistance',
                touches=len(group),
                start_idx=min(indices),
                end_idx=max(indices)
            ))

    # Group support levels (swing lows)
    support_groups = _group_price_levels(low_prices, tolerance_pct)
    for group in support_groups:
        if len(group) >= min_touches:
            avg_price = np.mean([p for _, p in group])
            strength = min(len(group) / 5, 1.0)
            indices = [idx for idx, _ in group]
            levels.append(SupportResistanceLevel(
                price=avg_price,
                strength=strength,
                level_type='support',
                touches=len(group),
                start_idx=min(indices),
                end_idx=max(indices)
            ))

    # Sort by strength
    levels.sort(key=lambda x: x.strength, reverse=True)

    return levels


def _group_price_levels(price_points: List[Tuple[int, float]],
                       tolerance_pct: float) -> List[List[Tuple[int, float]]]:
    """Group price points within tolerance into levels."""
    if not price_points:
        return []

    # Sort by price
    sorted_points = sorted(price_points, key=lambda x: x[1])

    groups = []
    current_group = [sorted_points[0]]

    for i in range(1, len(sorted_points)):
        idx, price = sorted_points[i]
        _, prev_price = current_group[-1]

        # Check if within tolerance of group average
        group_avg = np.mean([p for _, p in current_group])
        tolerance = group_avg * tolerance_pct / 100

        if abs(price - group_avg) <= tolerance:
            current_group.append((idx, price))
        else:
            groups.append(current_group)
            current_group = [(idx, price)]

    groups.append(current_group)
    return groups


def find_volume_profile_levels(df: pd.DataFrame,
                               num_bins: int = 50,
                               significance_threshold: float = 1.5) -> List[Dict]:
    """
    Find significant price levels based on volume profile.

    High-volume price levels often act as support/resistance.

    Args:
        df: DataFrame with 'close', 'high', 'low', 'volume' columns
        num_bins: Number of price bins for volume profile
        significance_threshold: Multiplier of average volume to be significant

    Returns:
        List of dicts with 'price', 'volume', 'type' keys
    """
    if 'volume' not in df.columns or len(df) < 10:
        return []

    # Create price range
    price_min = df['low'].min()
    price_max = df['high'].max()
    bin_size = (price_max - price_min) / num_bins

    if bin_size == 0:
        return []

    # Calculate volume at each price level
    volume_profile = np.zeros(num_bins)

    for _, row in df.iterrows():
        # Distribute volume across the candle's range
        low_bin = int((row['low'] - price_min) / bin_size)
        high_bin = int((row['high'] - price_min) / bin_size)

        low_bin = max(0, min(low_bin, num_bins - 1))
        high_bin = max(0, min(high_bin, num_bins - 1))

        # Distribute volume evenly across bins
        num_candle_bins = high_bin - low_bin + 1
        vol_per_bin = row['volume'] / num_candle_bins

        for b in range(low_bin, high_bin + 1):
            volume_profile[b] += vol_per_bin

    # Find significant levels (high volume nodes)
    avg_volume = volume_profile.mean()
    threshold = avg_volume * significance_threshold

    significant_levels = []
    for i, vol in enumerate(volume_profile):
        if vol >= threshold:
            price = price_min + (i + 0.5) * bin_size
            current_price = df['close'].iloc[-1]

            level_type = 'resistance' if price > current_price else 'support'

            significant_levels.append({
                'price': price,
                'volume': vol,
                'type': level_type,
                'strength': vol / volume_profile.max()
            })

    # Sort by volume
    significant_levels.sort(key=lambda x: x['volume'], reverse=True)

    return significant_levels[:10]  # Return top 10 levels


def calculate_fibonacci_retracement(high: float, low: float,
                                   uptrend: bool = True) -> Dict[str, float]:
    """
    Calculate Fibonacci retracement levels.

    Args:
        high: Swing high price
        low: Swing low price
        uptrend: If True, retracement from low to high

    Returns:
        Dict of Fibonacci levels
    """
    diff = high - low

    if uptrend:
        # Retracement levels from high going down
        levels = {
            '0%': high,
            '23.6%': high - 0.236 * diff,
            '38.2%': high - 0.382 * diff,
            '50%': high - 0.5 * diff,
            '61.8%': high - 0.618 * diff,
            '78.6%': high - 0.786 * diff,
            '100%': low,
        }
    else:
        # Retracement levels from low going up
        levels = {
            '0%': low,
            '23.6%': low + 0.236 * diff,
            '38.2%': low + 0.382 * diff,
            '50%': low + 0.5 * diff,
            '61.8%': low + 0.618 * diff,
            '78.6%': low + 0.786 * diff,
            '100%': high,
        }

    return levels


def calculate_fibonacci_extension(high: float, low: float, retracement: float,
                                 uptrend: bool = True) -> Dict[str, float]:
    """
    Calculate Fibonacci extension levels.

    Args:
        high: Swing high price
        low: Swing low price
        retracement: Price where retracement ended
        uptrend: If True, extension going up

    Returns:
        Dict of Fibonacci extension levels
    """
    diff = high - low

    if uptrend:
        # Extension levels going up from retracement
        levels = {
            '100%': retracement + diff,
            '127.2%': retracement + 1.272 * diff,
            '161.8%': retracement + 1.618 * diff,
            '200%': retracement + 2.0 * diff,
            '261.8%': retracement + 2.618 * diff,
        }
    else:
        # Extension levels going down from retracement
        levels = {
            '100%': retracement - diff,
            '127.2%': retracement - 1.272 * diff,
            '161.8%': retracement - 1.618 * diff,
            '200%': retracement - 2.0 * diff,
            '261.8%': retracement - 2.618 * diff,
        }

    return levels


def auto_detect_fibonacci_levels(df: pd.DataFrame,
                                lookback: int = 100) -> Dict[str, float]:
    """
    Automatically detect and calculate Fibonacci levels from recent price action.

    Args:
        df: DataFrame with OHLC data
        lookback: Number of bars to analyze

    Returns:
        Dict of Fibonacci retracement levels
    """
    if len(df) < lookback:
        lookback = len(df)

    recent_df = df.iloc[-lookback:]

    # Find swing high and low
    swing_high_idx = recent_df['high'].idxmax()
    swing_low_idx = recent_df['low'].idxmin()

    swing_high = recent_df.loc[swing_high_idx, 'high']
    swing_low = recent_df.loc[swing_low_idx, 'low']

    # Determine trend direction
    # If high came before low, it's a downtrend (retracement up)
    # If low came before high, it's an uptrend (retracement down)
    if isinstance(swing_high_idx, pd.Timestamp) and isinstance(swing_low_idx, pd.Timestamp):
        uptrend = swing_low_idx < swing_high_idx
    else:
        # Fallback: use position
        high_pos = recent_df.index.get_loc(swing_high_idx)
        low_pos = recent_df.index.get_loc(swing_low_idx)
        uptrend = low_pos < high_pos

    return calculate_fibonacci_retracement(swing_high, swing_low, uptrend)
