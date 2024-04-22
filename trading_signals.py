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
                              np.where((last_known_price > data['Predicted_Close'] * (1 + sentiment_threshold)), 'Sell', 'Hold'))
    # Drop rows where 'Signal' is NaN if you don't want 'Hold' signals to appear
    data.dropna(subset=['Signal'], inplace=True)
    return data
