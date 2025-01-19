import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import pandas_ta as ta
import numpy as np

def rsi_macd_buy(data):
    """
    Identify strong buy signals based on MACD and RSI indicators.

    Parameters:
    - data (pd.DataFrame): DataFrame containing crypto data with technical indicators.

    Returns:
    - pd.DataFrame: DataFrame with an additional column indicating the strength of buy signals.
    """
    # Ensure that the necessary columns are available
    required_columns = ['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9', 'RSI_14']
    if not all(col in data.columns for col in required_columns):
        print("Data is missing required columns for signal identification.")
        return data


    # Identify buy signals
    for i in range(1, len(data)):
        # MACD Conditions
        macd_cross = (data['MACD_12_26_9'].iloc[i] > data['MACDs_12_26_9'].iloc[i]) and (data['MACD_12_26_9'].iloc[i-1] <= data['MACDs_12_26_9'].iloc[i-1])
        macd_hist_positive = data['MACDh_12_26_9'].iloc[i] > 0
        macd_hist_increasing = data['MACDh_12_26_9'].iloc[i] > data['MACDh_12_26_9'].iloc[i-1]
        
        # RSI Conditions
        rsi_below_50 = data['RSI_14'].iloc[i] < 50
        rsi_below_40 =  data['RSI_14'].iloc[i] < 40
        rsi_below_30 = data['RSI_14'].iloc[i] < 30
        rsi_below_20 = data['RSI_14'].iloc[i] < 20
        rsi_upward = data['RSI_14'].iloc[i] > data['RSI_14'].iloc[i-1]

        # Strong Buy Signal Criteria
        if (macd_cross and macd_hist_positive and macd_hist_increasing and rsi_below_30) or rsi_below_20:
            data.at[i, 'MACD_RSI'] = 'Strong Buy'
        elif macd_cross and macd_hist_positive and rsi_below_40:
            data.at[i, 'MACD_RSI'] = 'Buy'
        else:
            data.at[i, 'MACD_RSI'] = 'None'

    return data

def rsi_macd_sell(data):
    """
    Identify strong sell signals based on MACD and RSI indicators.

    Parameters:
    - data (pd.DataFrame): DataFrame containing crypto data with technical indicators.

    Returns:
    - pd.DataFrame: DataFrame with an additional column indicating the strength of sell signals.
    """
    # Ensure that the necessary columns are available
    required_columns = ['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9', 'RSI_14']
    if not all(col in data.columns for col in required_columns):
        print("Data is missing required columns for signal identification.")
        return data


    # Identify sell signals
    for i in range(1, len(data)):
        # MACD Conditions
        macd_cross = (data['MACD_12_26_9'].iloc[i] < data['MACDs_12_26_9'].iloc[i]) and (data['MACD_12_26_9'].iloc[i-1] >= data['MACDs_12_26_9'].iloc[i-1])
        macd_hist_negative = data['MACDh_12_26_9'].iloc[i] < 0
        macd_hist_decreasing = data['MACDh_12_26_9'].iloc[i] < data['MACDh_12_26_9'].iloc[i-1]
        
        # RSI Conditions
        rsi_above_60 = data['RSI_14'].iloc[i] > 60
        rsi_above_70 = data['RSI_14'].iloc[i] > 70
        rsi_above_80 = data['RSI_14'].iloc[i] > 80
        rsi_downward = data['RSI_14'].iloc[i] < data['RSI_14'].iloc[i-1]

        # Strong Sell Signal Criteria
        if (macd_cross and macd_hist_negative and macd_hist_decreasing and rsi_above_70) or rsi_above_80:
            data.at[i, 'MACD_RSI_Sell'] = 'Strong Sell'
        elif macd_cross and macd_hist_negative and rsi_above_60:
            data.at[i, 'MACD_RSI_Sell'] = 'Sell'
        else:
            data.at[i, 'MACD_RSI_Sell'] = 'None'

    return data


def plot_interactive_candlestick(data, buy_price_usdt):
    """
    Plot an interactive candlestick chart with additional technical indicators, including separate subplots for MACD and RSI.
    Highlights buy and sell signals on the candlestick chart, and includes rejection candles.

    Parameters:
    - data (pd.DataFrame): DataFrame containing crypto data with 'Open time', 'Open', 'High', 'Low', 'Close' columns, 
                           and other technical indicators like SMA, EMA, MACD, RSI, and buy/sell signals.
    - buy_price_usdt (float): The buy price in USDT to plot a vertical line.

    Returns:
    - None: Displays the interactive plot.
    """
    # Ensure that the necessary columns are available
    required_columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'SMA_5', 'SMA_20', 'SMA_100', 
                        'EMA_5', 'EMA_20', 'EMA_100', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9', 
                        'RSI_14', 'MACD_RSI', 'MACD_RSI_Sell']
    if not all(col in data.columns for col in required_columns):
        print("Data is missing required columns for plotting.")
        return

    # Create subplots with three rows
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=('Candlestick Chart with Indicators', 'MACD', 'RSI'),
                        row_heights=[0.5, 0.25, 0.25])

    # Add Candlestick chart to the first subplot
    fig.add_trace(go.Candlestick(x=data['Open time'],
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name='Candlestick'),
                  row=1, col=1)

    # Add Simple Moving Averages (SMA) to the first subplot
    fig.add_trace(go.Scatter(x=data['Open time'], y=data['SMA_5'], mode='lines', name='SMA 5'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Open time'], y=data['SMA_20'], mode='lines', name='SMA 20'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Open time'], y=data['SMA_100'], mode='lines', name='SMA 100'),
                  row=1, col=1)

    # Add Exponential Moving Averages (EMA) to the first subplot
    fig.add_trace(go.Scatter(x=data['Open time'], y=data['EMA_5'], mode='lines', name='EMA 5'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Open time'], y=data['EMA_20'], mode='lines', name='EMA 20'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Open time'], y=data['EMA_100'], mode='lines', name='EMA 100'),
                  row=1, col=1)

    # Add MACD Line, Signal Line, and Histogram to the second subplot
    fig.add_trace(go.Scatter(x=data['Open time'], y=data['MACD_12_26_9'], mode='lines', name='MACD Line'),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=data['Open time'], y=data['MACDs_12_26_9'], mode='lines', name='Signal Line'),
                  row=2, col=1)
    fig.add_trace(go.Bar(x=data['Open time'], y=data['MACDh_12_26_9'], name='MACD Histogram'),
                  row=2, col=1)

    # Add RSI to the third subplot
    fig.add_trace(go.Scatter(x=data['Open time'], y=data['RSI_14'], mode='lines', name='RSI 14'),
                  row=3, col=1)

    # Highlight buy signals on the candlestick chart
    buy_signals = data[data['MACD_RSI'] == 'Strong Buy']
    fig.add_trace(go.Scatter(x=buy_signals['Open time'], 
                             y=buy_signals['Low'] * 0.99,
                             mode='markers',
                             marker=dict(symbol='triangle-up', color='red', size=10),
                             name='Strong Buy Signal'),
                  row=1, col=1)

    buy_signals = data[data['MACD_RSI'] == 'Buy']
    fig.add_trace(go.Scatter(x=buy_signals['Open time'], 
                             y=buy_signals['Low'] * 0.99,
                             mode='markers',
                             marker=dict(symbol='triangle-up', color='blue', size=10),
                             name='Buy Signal'),
                  row=1, col=1)

    # Highlight sell signals on the candlestick chart
    sell_signals = data[data['MACD_RSI_Sell'] == 'Strong Sell']
    fig.add_trace(go.Scatter(x=sell_signals['Open time'], 
                             y=sell_signals['High'] * 1.01,
                             mode='markers',
                             marker=dict(symbol='triangle-down', color='green', size=10),
                             name='Strong Sell Signal'),
                  row=1, col=1)

    sell_signals = data[data['MACD_RSI_Sell'] == 'Sell']
    fig.add_trace(go.Scatter(x=sell_signals['Open time'], 
                             y=sell_signals['High'] * 1.01,
                             mode='markers',
                             marker=dict(symbol='triangle-down', color='yellow', size=10),
                             name='Sell Signal'),
                  row=1, col=1)

    # Add vertical line for buy price
    fig.add_hline(y=buy_price_usdt, line=dict(color='white', width=2), row=1, col=1)

    # Update the layout for better visualization
    fig.update_layout(
        title='Interactive Candlestick Chart with Indicators, MACD, RSI, Buy/Sell Signals, and Rejection Candles',
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=900
    )

    # Update y-axis range for better scaling
    fig.update_yaxes(range=[data['Low'].min() * 0.95, data['High'].max() * 1.05], row=1, col=1)
    
    # Show the plot
    fig.show()