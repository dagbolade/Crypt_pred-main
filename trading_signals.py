import numpy as np


def calculate_rsi(prices, periods=14):
    # Calculate RSI using the prices Series
    delta = prices.diff()
    gain = (delta.mask(delta < 0, 0)).fillna(0)
    loss = (-delta.mask(delta > 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=periods).mean()
    avg_loss = loss.rolling(window=periods).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def generate_trading_signals(data, last_known_price, rsi_period=14, sentiment_threshold=0.1):
    # Calculate RSI using the 'Predicted_Close' column
    data['RSI'] = calculate_rsi(data['Predicted_Close'], periods=rsi_period)

    # Calculate Moving Averages using the 'Predicted_Close' column
    data['MA_Short'] = data['Predicted_Close'].rolling(window=7).mean()
    data['MA_Long'] = data['Predicted_Close'].rolling(window=30).mean().fillna(0)

    # Generate signals based on RSI and Moving Averages data['Signal'] = np.where((data['RSI'] < 30) & (data[
    # 'MA_Short'] > data['MA_Long']), 'Buy', np.where((data['RSI'] > 70) & (data['MA_Short'] < data['MA_Long']),
    # 'Sell', 'Hold', 'Hold'))

    # Generate signals based on the comparison between the last known price and the predicted price
    data['Signal'] = np.where((last_known_price < data['Predicted_Close'] * (1 - sentiment_threshold)), 'Buy',
                              np.where((last_known_price > data['Predicted_Close'] * (1 + sentiment_threshold)), 'Sell',
                                       'Hold'))
    # Drop rows where 'Signal' is NaN if you don't want 'Hold' signals to appear
    data.dropna(subset=['Signal'], inplace=True)
    return data


# using last known price to generate signals
def generate_prophet_trading_signals(data, last_known_price, rsi_period=14, sentiment_threshold=0.1):
    # Calculate RSI using the 'yhat' column
    data['RSI'] = calculate_rsi(data['yhat'], periods=rsi_period)

    # Calculate Moving Averages using the 'yhat' column
    data['MA_Short'] = data['yhat'].rolling(window=7).mean()
    data['MA_Long'] = data['yhat'].rolling(window=30).mean()

    # Generate signals based on the comparison between the last known price and the predicted price
    # Ensure the scalar is broadcast over the Series correctly
    data['Signal'] = np.where(
        (last_known_price < data['yhat'] * (1 - sentiment_threshold)), 'Buy',
        np.where(
            (last_known_price > data['yhat'] * (1 + sentiment_threshold)), 'Sell',
            'Hold'
        )
    )

    return data


# using rsi and moving averages to generate signals
def generate_prophet1_trading_signals(forecast):
    # Calculate RSI on forecasted 'yhat' values
    forecast['RSI'] = calculate_rsi(forecast['yhat'], periods=14)

    # Calculate short-term and long-term moving averages on forecasted 'yhat' values
    forecast['MA_Short'] = forecast['yhat'].rolling(window=7).mean()
    forecast['MA_Long'] = forecast['yhat'].rolling(window=30).mean()

    # Generate signals
    forecast['Signal'] = np.where(
        (forecast['RSI'] < 30) & (forecast['MA_Short'] > forecast['MA_Long']), 'Buy',
        np.where(
            (forecast['RSI'] > 70) & (forecast['MA_Short'] < forecast['MA_Long']), 'Sell',
            'Hold'
        )
    )

    return forecast


import plotly.graph_objects as go


def plot_forecast_with_signals(historical_data, forecast, signals):
    # Create a figure
    fig = go.Figure()

    # Plot historical data
    fig.add_trace(go.Scatter(
        x=historical_data['ds'],
        y=historical_data['y'],
        mode='lines',
        name='Historical Data'
    ))

    # Plot forecasted data
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecasted Data',
        line=dict(color='blue', dash='dot')  # Different style for forecast
    ))

    # Highlight the forecast intervals (optional)
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        fill='tonexty',  # fill area between trace0 and trace1
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        showlegend=False
    ))

    # Adding Buy signals
    buys = signals[signals['Signal'] == 'Buy']
    fig.add_trace(go.Scatter(
        x=buys['ds'],
        y=buys['yhat'],
        mode='markers',
        name='Buy Signal',
        marker=dict(color='green', size=10, symbol='triangle-up')
    ))

    # Adding Sell signals
    sells = signals[signals['Signal'] == 'Sell']
    fig.add_trace(go.Scatter(
        x=sells['ds'],
        y=sells['yhat'],
        mode='markers',
        name='Sell Signal',
        marker=dict(color='red', size=10, symbol='triangle-down')
    ))

    # Add titles and labels
    fig.update_layout(
        title='Forecast with Trading Signals',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Legend'
    )

    return fig


def plot_forecast_with_signals2(forecast):
    # Create a figure
    fig = go.Figure()

    # Plot the historical forecast line
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast'
    ))

    # Plot buy signals
    fig.add_trace(go.Scatter(
        x=forecast[forecast['Signal'] == 'Buy']['ds'],
        y=forecast[forecast['Signal'] == 'Buy']['yhat'],
        mode='markers',
        name='Buy Signal',
        marker=dict(color='green', size=10, symbol='triangle-up')
    ))

    # Plot sell signals
    fig.add_trace(go.Scatter(
        x=forecast[forecast['Signal'] == 'Sell']['ds'],
        y=forecast[forecast['Signal'] == 'Sell']['yhat'],
        mode='markers',
        name='Sell Signal',
        marker=dict(color='red', size=10, symbol='triangle-down')
    ))

    # Plot MA Short
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['MA_Short'],
        mode='lines',
        name='MA Short',
        line=dict(color='orange', width=2)
    ))

    # Plot MA Long
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['MA_Long'],
        mode='lines',
        name='MA Long',
        line=dict(color='purple', width=2)
    ))

    # Update layout to include titles and axis labels as necessary
    fig.update_layout(
        title='Forecast with Trading Signals and Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price',
        showlegend=True
    )

    return fig


def generate_arima_trading_signals(forecast_df, last_known_price, threshold=0.01):
    """
    Generate trading signals based on ARIMA forecasted prices.

    Parameters:
    forecast_df (pd.DataFrame): A DataFrame with 'Date' and 'Forecast' columns.
    last_known_price (float): The last known price of the asset.
    threshold (float): The threshold for determining signals.

    Returns:
    pd.DataFrame: The input DataFrame with an additional 'Signal' column.
    """
    # Calculate the percentage change between the forecast and the last known price
    forecast_df['Change'] = (forecast_df['Forecast'] - last_known_price) / last_known_price

    # Determine signals based on the change threshold
    forecast_df['Signal'] = forecast_df['Change'].apply(
        lambda x: 'Buy' if x > threshold else 'Sell' if x < -threshold else 'Hold')

    return forecast_df


def plot_arima_forecast_with_signals(arima_signals_df):
    """
    Plot ARIMA forecast with trading signals.

    Parameters:
    arima_signals_df (pd.DataFrame): The DataFrame with 'Date', 'Forecast', and 'Signal' columns.

    Returns:
    plotly.graph_objs._figure.Figure: The plotly figure object for the forecast with signals.
    """
    # Create the figure
    fig = go.Figure()

    # Add line for the forecast
    fig.add_trace(go.Scatter(x=arima_signals_df['Date'], y=arima_signals_df['Forecast'],
                             mode='lines', name='Forecast'))

    # Add markers for buy signals
    buy_signals = arima_signals_df[arima_signals_df['Signal'] == 'Buy']
    fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Forecast'],
                             mode='markers', name='Buy Signal',
                             marker=dict(color='green', size=10, symbol='triangle-up')))

    # Add markers for sell signals
    sell_signals = arima_signals_df[arima_signals_df['Signal'] == 'Sell']
    fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Forecast'],
                             mode='markers', name='Sell Signal',
                             marker=dict(color='red', size=10, symbol='triangle-down')))

    # Add markers for hold signals if needed
    # hold_signals = arima_signals_df[arima_signals_df['Signal'] == 'Hold']
    # fig.add_trace(go.Scatter(x=hold_signals['Date'], y=hold_signals['Forecast'],
    #                          mode='markers', name='Hold Signal', marker=dict(color='blue', size=10, symbol='circle')))

    # Update layout for readability
    fig.update_layout(title='ARIMA Forecast with Trading Signals',
                      xaxis_title='Date', yaxis_title='Price',
                      showlegend=True)

    return fig
