import pandas as pd
import numpy as np
import plotly.graph_objects as go

def identify_candlestick_patterns(data):
    """
    Identify various candlestick patterns.
    """
    patterns = pd.DataFrame(index=data.index, columns=['Pattern'])
    
    # Calculate candle body and shadows
    data['Body'] = abs(data['Close'] - data['Open'])
    data['Upper_Shadow'] = data['High'] - data[['Open', 'Close']].max(axis=1)
    data['Lower_Shadow'] = data[['Open', 'Close']].min(axis=1) - data['Low']
    
    # Doji
    doji = (abs(data['Open'] - data['Close']) / (data['High'] - data['Low'])) < 0.1
    patterns.loc[doji, 'Pattern'] = 'Doji'
    
    # Hammer
    hammer = (
        (data['Low'] < data['Open']) &
        (data['Low'] < data['Close']) &
        (data['Body'] < data['Lower_Shadow']) &
        (data['Upper_Shadow'] < data['Body'] * 0.5)
    )
    patterns.loc[hammer, 'Pattern'] = 'Hammer'
    
    # Inverted Hammer
    inverted_hammer = (
        (data['High'] > data['Open']) &
        (data['High'] > data['Close']) &
        (data['Body'] < data['Upper_Shadow']) &
        (data['Lower_Shadow'] < data['Body'] * 0.5)
    )
    patterns.loc[inverted_hammer, 'Pattern'] = 'Inverted Hammer'
    
    # Bullish Engulfing
    bullish_engulfing = (
        (data['Open'].shift(1) > data['Close'].shift(1)) &  # Previous candle is bearish
        (data['Close'] > data['Open']) &  # Current candle is bullish
        (data['Open'] < data['Close'].shift(1)) &  # Current open is lower than previous close
        (data['Close'] > data['Open'].shift(1))  # Current close is higher than previous open
    )
    patterns.loc[bullish_engulfing, 'Pattern'] = 'Bullish Engulfing'
    
    # Bearish Engulfing
    bearish_engulfing = (
        (data['Close'].shift(1) > data['Open'].shift(1)) &  # Previous candle is bullish
        (data['Open'] > data['Close']) &  # Current candle is bearish
        (data['Close'] < data['Open'].shift(1)) &  # Current close is lower than previous open
        (data['Open'] > data['Close'].shift(1))  # Current open is higher than previous close
    )
    patterns.loc[bearish_engulfing, 'Pattern'] = 'Bearish Engulfing'
    
    # Morning Star (simplified)
    morning_star = (
        (data['Close'].shift(2) > data['Open'].shift(2)) &  # First candle is bearish
        (abs(data['Open'].shift(1) - data['Close'].shift(1)) < data['Body'].shift(2) * 0.3) &  # Second candle has a small body
        (data['Close'] > data['Open']) &  # Third candle is bullish
        (data['Close'] > (data['Open'].shift(2) + data['Close'].shift(2)) / 2)  # Third candle closes above midpoint of first candle
    )
    patterns.loc[morning_star, 'Pattern'] = 'Morning Star'
    
    # Evening Star (simplified)
    evening_star = (
        (data['Close'].shift(2) < data['Open'].shift(2)) &  # First candle is bullish
        (abs(data['Open'].shift(1) - data['Close'].shift(1)) < data['Body'].shift(2) * 0.3) &  # Second candle has a small body
        (data['Close'] < data['Open']) &  # Third candle is bearish
        (data['Close'] < (data['Open'].shift(2) + data['Close'].shift(2)) / 2)  # Third candle closes below midpoint of first candle
    )
    patterns.loc[evening_star, 'Pattern'] = 'Evening Star'
    
    return patterns

def identify_strong_buy_sell_points(data, patterns):
    """
    Identify strong buying and selling points based on candlestick patterns and pivot points.
    """
    buy_points = pd.Series(index=data.index, dtype=bool).fillna(False)
    sell_points = pd.Series(index=data.index, dtype=bool).fillna(False)
    
    # Strong buy signals
    buy_conditions = (
        ((patterns['Pattern'] == 'Hammer') & (data['Close'] < data['S1'])) |
        ((patterns['Pattern'] == 'Bullish Engulfing') & (data['Close'] < data['S2'])) |
        ((patterns['Pattern'] == 'Morning Star') & (data['Close'] < data['S1']))
    )
    
    # Strong sell signals
    sell_conditions = (
        ((patterns['Pattern'] == 'Inverted Hammer') & (data['Close'] > data['R1'])) |
        ((patterns['Pattern'] == 'Bearish Engulfing') & (data['Close'] > data['R2'])) |
        ((patterns['Pattern'] == 'Evening Star') & (data['Close'] > data['R1']))
    )
    
    # Remove additional conditions to allow more signals
    buy_points = buy_conditions
    sell_points = sell_conditions
    
    return buy_points, sell_points

def plot_candlestick_patterns_with_signals(data, patterns):
    """
    Plot candlestick chart with identified patterns, pivot lines, and strong buy/sell signals.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=data['Open time'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlesticks'
    ))
    
    pattern_colors = {
        'Hammer': 'green',
        'Inverted Hammer': 'lime',
        'Bullish Engulfing': 'darkgreen',
        'Bearish Engulfing': 'red',
        'Morning Star': 'blue',
        'Evening Star': 'orange'
    }
    
    for pattern in pattern_colors:
        pattern_data = patterns[patterns['Pattern'] == pattern]
        fig.add_trace(go.Scatter(
            x=data.loc[pattern_data.index, 'Open time'],
            y=data.loc[pattern_data.index, 'Low'],
            mode='markers',
            marker=dict(symbol='triangle-up', size=10, color=pattern_colors[pattern]),
            name=pattern
        ))
    
    # Add pivot lines
    pivot_levels = ['Pivot', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3']
    pivot_colors = ['yellow', 'green', 'red', 'blue', 'purple', 'cyan', 'magenta']
    
    for level, color in zip(pivot_levels, pivot_colors):
        fig.add_trace(go.Scatter(
            x=data['Open time'],
            y=data[level],
            mode='lines',
            line=dict(color=color, width=1, dash='dash'),
            name=level
        ))
    
    # Add strong buy and sell signals
    buy_points, sell_points = identify_strong_buy_sell_points(data, patterns)
    
    fig.add_trace(go.Scatter(
        x=data.loc[buy_points, 'Open time'],
        y=data.loc[buy_points, 'Low'],
        mode='markers',
        marker=dict(symbol='triangle-up', size=15, color='green'),
        name='Strong Buy Signal'
    ))
    
    fig.add_trace(go.Scatter(
        x=data.loc[sell_points, 'Open time'],
        y=data.loc[sell_points, 'High'],
        mode='markers',
        marker=dict(symbol='triangle-down', size=15, color='red'),
        name='Strong Sell Signal'
    ))
    
    fig.update_layout(
        title='Candlestick Patterns with Pivot Lines and Strong Buy/Sell Signals',
        yaxis_title='Price',
        xaxis_title='Time',
        xaxis_rangeslider_visible=False
    )
    
    fig.show()

# Example usage:
# data = pd.read_csv('BNBUSDT.csv')
# patterns = identify_candlestick_patterns(data)
# plot_candlestick_patterns_with_signals(data, patterns)