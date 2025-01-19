import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import pandas_ta as ta
import numpy as np

def stochastic_ma_buy(data):
    """
    Identify buy signals based on Stochastic Oscillator and Moving Averages.

    Parameters:
    - data (pd.DataFrame): DataFrame containing crypto data with technical indicators.

    Returns:
    - pd.DataFrame: DataFrame with an additional column indicating buy signals.
    """
    # Ensure that the necessary columns are available
    required_columns = ['STOCH_K', 'STOCH_D', 'SMA_20', 'SMA_50']
    if not all(col in data.columns for col in required_columns):
        print("Data is missing required columns for signal identification.")
        return data

    # Initialize the new column with default values
    data['Stoch_MA_Signal'] = 'None'

    # Identify buy signals
    for i in range(1, len(data)):
        # Stochastic Oscillator conditions
        stoch_oversold = data['STOCH_K'].iloc[i-1] < 20 and data['STOCH_K'].iloc[i] > 20
        stoch_crossover = (data['STOCH_K'].iloc[i] > data['STOCH_D'].iloc[i]) and \
                          (data['STOCH_K'].iloc[i-1] <= data['STOCH_D'].iloc[i-1])
        
        # Moving Average conditions
        price_above_ma20 = data['Close'].iloc[i] > data['SMA_20'].iloc[i]
        ma20_above_ma50 = data['SMA_20'].iloc[i] > data['SMA_50'].iloc[i]

        # Strong Buy Signal Criteria
        if stoch_oversold and stoch_crossover and price_above_ma20 and ma20_above_ma50:
            data.at[i, 'Stoch_MA_Signal'] = 'Strong Buy'
        elif stoch_crossover and price_above_ma20:
            data.at[i, 'Stoch_MA_Signal'] = 'Buy'

    return data

def stochastic_ma_sell(data):
    """
    Identify sell signals based on Stochastic Oscillator and Moving Averages.

    Parameters:
    - data (pd.DataFrame): DataFrame containing crypto data with technical indicators.

    Returns:
    - pd.DataFrame: DataFrame with an additional column indicating sell signals.
    """
    # Ensure that the necessary columns are available
    required_columns = ['STOCH_K', 'STOCH_D', 'SMA_20', 'SMA_50']
    if not all(col in data.columns for col in required_columns):
        print("Data is missing required columns for signal identification.")
        return data

    # Initialize the new column with default values
    data['Stoch_MA_Sell'] = 'None'

    # Identify sell signals
    for i in range(1, len(data)):
        # Stochastic Oscillator conditions
        stoch_overbought = data['STOCH_K'].iloc[i-1] > 80 and data['STOCH_K'].iloc[i] < 80
        stoch_crossunder = (data['STOCH_K'].iloc[i] < data['STOCH_D'].iloc[i]) and \
                           (data['STOCH_K'].iloc[i-1] >= data['STOCH_D'].iloc[i-1])
        
        # Moving Average conditions
        price_below_ma20 = data['Close'].iloc[i] < data['SMA_20'].iloc[i]
        ma20_below_ma50 = data['SMA_20'].iloc[i] < data['SMA_50'].iloc[i]

        # Strong Sell Signal Criteria
        if stoch_overbought and stoch_crossunder and price_below_ma20 and ma20_below_ma50:
            data.at[i, 'Stoch_MA_Sell'] = 'Strong Sell'
        elif stoch_crossunder and price_below_ma20:
            data.at[i, 'Stoch_MA_Sell'] = 'Sell'

    return data

def plot_stochastic_ma(data, buy_price_usdt):
    """
    Plot an interactive chart with Stochastic Oscillator, Moving Averages, and buy/sell signals.

    Parameters:
    - data (pd.DataFrame): DataFrame containing crypto data with technical indicators.
    - buy_price_usdt (float): The buy price in USDT to plot a horizontal line.

    Returns:
    - None: Displays the interactive plot.
    """
    # Create subplots with two rows
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=('Candlestick Chart with Moving Averages', 'Stochastic Oscillator'),
                        row_heights=[0.7, 0.3])

    # Add Candlestick chart to the first subplot
    fig.add_trace(go.Candlestick(x=data['Open time'],
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Candlestick'),
                  row=1, col=1)

    # Add Moving Averages to the first subplot
    fig.add_trace(go.Scatter(x=data['Open time'], y=data['SMA_20'], mode='lines', name='SMA 20'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Open time'], y=data['SMA_50'], mode='lines', name='SMA 50'),
                  row=1, col=1)

    # Add Stochastic Oscillator to the second subplot
    fig.add_trace(go.Scatter(x=data['Open time'], y=data['STOCH_K'], mode='lines', name='Stoch %K'),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=data['Open time'], y=data['STOCH_D'], mode='lines', name='Stoch %D'),
                  row=2, col=1)

    # Highlight buy signals on the candlestick chart
    buy_signals = data[data['Stoch_MA_Signal'] == 'Strong Buy']
    fig.add_trace(go.Scatter(x=buy_signals['Open time'], 
                             y=buy_signals['Low'] * 0.99,
                             mode='markers',
                             marker=dict(symbol='triangle-up', color='green', size=10),
                             name='Strong Buy Signal'),
                  row=1, col=1)

    buy_signals = data[data['Stoch_MA_Signal'] == 'Buy']
    fig.add_trace(go.Scatter(x=buy_signals['Open time'], 
                             y=buy_signals['Low'] * 0.99,
                             mode='markers',
                             marker=dict(symbol='triangle-up', color='lime', size=8),
                             name='Buy Signal'),
                  row=1, col=1)

    # Highlight sell signals on the candlestick chart
    sell_signals = data[data['Stoch_MA_Sell'] == 'Strong Sell']
    fig.add_trace(go.Scatter(x=sell_signals['Open time'], 
                             y=sell_signals['High'] * 1.01,
                             mode='markers',
                             marker=dict(symbol='triangle-down', color='red', size=10),
                             name='Strong Sell Signal'),
                  row=1, col=1)

    sell_signals = data[data['Stoch_MA_Sell'] == 'Sell']
    fig.add_trace(go.Scatter(x=sell_signals['Open time'], 
                             y=sell_signals['High'] * 1.01,
                             mode='markers',
                             marker=dict(symbol='triangle-down', color='orange', size=8),
                             name='Sell Signal'),
                  row=1, col=1)

    # Add horizontal line for buy price
    fig.add_hline(y=buy_price_usdt, line=dict(color='white', width=2), row=1, col=1)

    # Add horizontal lines for overbought and oversold levels in Stochastic Oscillator
    fig.add_hline(y=80, line=dict(color='red', width=1, dash='dash'), row=2, col=1)
    fig.add_hline(y=20, line=dict(color='green', width=1, dash='dash'), row=2, col=1)

    # Update the layout for better visualization
    fig.update_layout(
        title='Stochastic Oscillator and Moving Averages Approach',
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=800
    )

    # Update y-axis range for better scaling
    fig.update_yaxes(range=[data['Low'].min() * 0.95, data['High'].max() * 1.05], row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    # Show the plot
    fig.show()

# You can call these functions in your main script like this:
# data = ... # Your DataFrame with OHLC data and indicators
# data = stochastic_ma_buy(data)
# data = stochastic_ma_sell(data)
# plot_stochastic_ma(data, buy_price_usdt)