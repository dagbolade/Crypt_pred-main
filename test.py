import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from pmdarima import auto_arima
from sklearn.model_selection import train_test_split

from Data_downolader import CryptoDataDownloader
from arima import predict_arima, fit_arima_model, find_best_arima

from bi_lstm_model import build_bi_lstm_model, train_bi_lstm_model
from clustering import select_cryptos_closest_to_centroids, plot_clusters, add_cluster_labels, apply_kmeans
from correlation import calculate_correlation_matrix, find_top_correlations, calculate_daily_returns
from data_preprocessing import convert_to_datetime
from data_transformation import scale_data, pivot_and_fill, remove_duplicates
from dimensionality_reduction import plot_explained_variance, apply_pca
from eda import plot_time_series, plot_rolling_statistics, plot_boxplot, plot_candlestick, \
    plot_volatility_clustering, plot_kde_of_closes, plot_candlestick_with_signals_and_ma
from feature_engineering import calculate_sma_ema_rsi
from lstm_model import prepare_lstm_data, build_lstm_model, train_lstm_model
from prophet_model import prepare_data_for_prophet, train_prophet_model, plot_forecast
from trading_signals import generate_trading_signals

from xgboost_model import train_xgboost_model

# Set up the sidebar menu
menu_options = [
    "Overview", "About", "Data Preprocessing", "Exploratory Data Analysis",
    "Correlation Analysis", "Prediction", "Trading Signals"
]
menu_choice = st.sidebar.radio("Menu", menu_options)


def show_overview():
    st.header("Overview")
    st.write("Welcome to the Cryptocurrency Analysis Dashboard!")


def show_about():
    st.header("About")
    st.write("This section provides information about the app and its functionalities.")


def data_preprocessing():
    st.header("Data Preprocessing")
    # Include all your data preprocessing code here
    st.title('Cryptocurrency Data Downloader')

    # Cache the data downloader function to prevent unnecessary downloads
    @st.cache_data
    def download_data(tickers, start_date, end_date, interval):
        downloader = CryptoDataDownloader(tickers, start_date=start_date, end_date=end_date, interval=interval)
        return downloader.download_data()

    # Sidebar for user inputs
    st.sidebar.header('Download Settings')
    all_cryptos = [
        'BNB-USD', 'ADA-USD', 'XRP-USD', 'DOGE-USD', 'DOT-USD', 'UNI-USD', 'BTC-USD',
        'BCH-USD', 'LTC-USD', 'BTC-USD', 'ETH-USD', 'LINK-USD', 'XLM-USD', 'USDC-USD', 'USDT-USD', 'VET-USD', 'ETC-USD',
        'FIL-USD', 'TRX-USD', 'EOS-USD', 'THETA-USD', 'XMR-USD', 'NEO-USD', 'AAVE-USD', 'ATOM-USD', 'WIF-USD', 'BONK'
                                                                                                               '-USD',
        'SHIB-USD', 'SOL-USD',
        'DGB-USD', 'CHZ-USD', 'ENJ-USD', 'MANA-USD', 'BAT-USD', 'SAND-USD'
    ]

    # Use top_30_cryptos as default in multiselect
    selected_tickers = st.sidebar.multiselect(
        'Select Cryptocurrencies, 30 cryptos have been added by default',
        options=all_cryptos,
        default=all_cryptos
    )

    start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2022-01-01'))
    end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('today'))
    interval = st.sidebar.selectbox('Select Interval', options=['1d', '1wk', '1mo'], index=0)
    download_button = st.sidebar.button('Download Data')

    # Initiate data downloading
    if download_button and selected_tickers:
        with st.spinner('Downloading data...'):
            st.session_state.crypto_data = download_data(selected_tickers, start_date, end_date, interval)
            st.success('Data downloaded successfully!')
            st.write("Data Sample:")
            st.dataframe(st.session_state.crypto_data.head())
    if 'crypto_data' in st.session_state and not st.session_state.crypto_data.empty:
        st.header("Data Preprocessing and Transformation")

        # If 'Date' is in the index, reset index to make 'Date' a column
        if 'Date' in st.session_state.crypto_data.index.names:
            st.session_state.crypto_data.reset_index(inplace=True)
            st.write("'Date' column reset from index.")

        # Check again after resetting index
        if 'Date' not in st.session_state.crypto_data.columns:
            st.error("'Date' column is missing from the data after reset. Please check the data source.")
            st.stop()

        st.session_state.crypto_data = remove_duplicates(st.session_state.crypto_data, subset=['Ticker', 'Date'])
        st.write("Duplicates removed.")

        processed_data = pivot_and_fill(st.session_state.crypto_data, index='Ticker', columns='Date', values='Close')
        st.write("Data pivoted and filled.")
        st.dataframe(processed_data.head())
        scaled_data = scale_data(processed_data)
        st.write("Data scaled.")
        reduced_data_df, pca = apply_pca(scaled_data, n_components=10)
        st.write("PCA applied with 10 components.")
        st.dataframe(reduced_data_df.head())

        # Display explained variance plot
        fig = plot_explained_variance(pca, n_components=10)
        st.pyplot(fig)

        # Perform clustering
        clusters, kmeans = apply_kmeans(reduced_data_df, n_clusters=4)
        processed_data = add_cluster_labels(processed_data, clusters)
        st.write("K-means clustering applied with 4 clusters.")
        st.dataframe(processed_data.head())

        # Show the counts of cryptocurrencies in each cluster
        st.write("Cluster Distribution:")
        st.write(processed_data['Cluster'].value_counts())

        # Visualize clusters
        fig = plot_clusters(reduced_data_df.values, clusters)
        st.pyplot(fig)

        # Automatically select cryptos closest to centroids
        selected_cryptos = select_cryptos_closest_to_centroids(reduced_data_df, clusters, kmeans.cluster_centers_)
        st.write("Cryptocurrencies closest to each cluster centroid:")
        st.dataframe(selected_cryptos)

        # Convert 'Date' column to datetime and set as index
        st.session_state.crypto_data = convert_to_datetime(st.session_state.crypto_data, 'Date').set_index('Date')

        # Drop every other column on selected cryptos, except the Ticker and cluster
        selected_cryptos = selected_cryptos[['Cluster']]
        st.write("Selected Cryptos:")
        st.write(selected_cryptos)

        # Merge the selected cryptocurrencies with the original data to get their full details including Date
        selected_cryptos_full = st.session_state.crypto_data.merge(selected_cryptos, left_on='Ticker', right_index=True)

        # Make sure the data is sorted by Date and its a column not an index
        selected_cryptos_full = selected_cryptos_full.reset_index().sort_values('Date')
        st.write("Selected Cryptos Full Details:")
        st.write(selected_cryptos_full)

        # Calculate daily returns for all cryptocurrencies
        daily_returns = calculate_daily_returns(st.session_state.crypto_data)

        # Apply the feature engineering to the selected cryptos
        selected_cryptos_full = calculate_sma_ema_rsi(selected_cryptos_full)
        st.write("Selected Cryptos Full Details with Feature Engineering:")
        st.write(selected_cryptos_full)

        # Store in session state
        st.session_state['selected_cryptos'] = selected_cryptos
        st.session_state['daily_returns'] = daily_returns
        st.session_state['selected_cryptos_full'] = selected_cryptos_full

        st.write("Preprocessing completed. Data stored in session state.")
    else:
        st.info("Please download data to proceed with transformation and analysis.")


def exploratory_data_analysis():
    st.header("Exploratory Data Analysis")
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Ensure the required data is available in the session state
    if 'selected_cryptos_full' in st.session_state:
        selected_cryptos_full = st.session_state['selected_cryptos_full']

        # Convert 'Date' if it's not already the index
        if 'Date' in selected_cryptos_full.columns:
            selected_cryptos_full['Date'] = pd.to_datetime(selected_cryptos_full['Date'])
            selected_cryptos_full.set_index('Date', inplace=True)

        # Create a dropdown menu to select a cryptocurrency
        unique_tickers = selected_cryptos_full['Ticker'].unique()
        selected_ticker = st.selectbox('Select a Cryptocurrency', unique_tickers)

        # Filter data for the selected cryptocurrency
        ticker_data = selected_cryptos_full[selected_cryptos_full['Ticker'] == selected_ticker]

        # Display plots for the selected cryptocurrency
        if not ticker_data.empty:
            # describe the data
            st.write(f"Descriptive statistics for {selected_ticker}:")
            st.write(ticker_data.describe())

            fig = plot_time_series(ticker_data, selected_ticker)
            st.pyplot(fig)

            fig = plot_rolling_statistics(ticker_data, selected_ticker)
            st.pyplot(fig)

            # fig = plot_distribution(ticker_data, selected_ticker)
            # st.pyplot(fig)

            fig = plot_boxplot(ticker_data, selected_ticker)
            st.pyplot(fig)

            fig = plot_candlestick(ticker_data, selected_ticker)
            st.plotly_chart(fig)

            fig = plot_volatility_clustering(ticker_data, selected_ticker)
            st.pyplot(fig)

            fig = plot_kde_of_closes(ticker_data, [selected_ticker])
            st.pyplot(fig)

            fig = plot_candlestick_with_signals_and_ma(ticker_data, selected_ticker)
            st.plotly_chart(fig)
        else:
            st.error("No data available for the selected cryptocurrency.")
    else:
        st.info("Please perform data preprocessing to generate and select cryptocurrencies for analysis.")


def correlation_analysis():
    if 'selected_cryptos' in st.session_state and 'daily_returns' in st.session_state:
        selected_cryptos = st.session_state['selected_cryptos']
        daily_returns = st.session_state['daily_returns']

        st.write("Selected Cryptocurrencies:", selected_cryptos)
        st.write("Daily Returns Sample:", daily_returns.head())

        # Perform correlation analysis using these
        correlation_matrix = calculate_correlation_matrix(daily_returns)
        top_correlations = find_top_correlations(correlation_matrix, selected_cryptos.index.tolist())

        # Display top correlations
        for ticker, correlation_data in top_correlations.items():
            st.write(f"Top correlations for {ticker}")
            st.write("Positive Correlations:", correlation_data['Positive'])
            st.write("Negative Correlations:", correlation_data['Negative'])
    else:
        st.error("Data not available. Please run the preprocessing first.")


import plotly.express as px


def generate_lstm_signals(current_price, predicted_prices, threshold=0.01):
    signals = []
    for predicted_price in predicted_prices:
        if predicted_price > current_price * (1 + threshold):
            signals.append('Buy')  # Buy signal
        elif predicted_price < current_price * (1 - threshold):
            signals.append('Sell')  # Sell signal
        else:
            signals.append('Hold')  # Hold signal
    return signals


import plotly.graph_objs as go


def prediction():
    st.header("Prediction")

    if 'selected_cryptos_full' in st.session_state:
        selected_cryptos_full = st.session_state['selected_cryptos_full']

        # Allowing user to select a model
        model_choice = st.selectbox('Select Prediction Model', ['LSTM', 'Prophet', 'BI-LSTM', 'ARIMA'])

        # User input for selecting cryptocurrency
        ticker = st.selectbox('Select a Cryptocurrency for Prediction:', selected_cryptos_full['Ticker'].unique())
        days_to_predict = st.number_input('Enter the number of days to predict:', min_value=1, max_value=365, value=30)

        # Filter data for the selected cryptocurrency
        crypto_data = selected_cryptos_full[selected_cryptos_full['Ticker'] == ticker]

        if st.button('Predict'):
            with st.spinner(f'Training {model_choice} model and making predictions...'):
                if model_choice == 'LSTM':
                    # Assuming the prepare_lstm_data and other model functions are adjusted to handle data appropriately
                    X, y, scaler = prepare_lstm_data(crypto_data, 'Close', sequence_length=60)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                    # reset index to make 'Date' a column
                    crypto_data = crypto_data.reset_index()

                    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
                    model, history = train_lstm_model(model, X_train, y_train, X_test, y_test)

                    # reset index to make 'Date' a column
                    crypto_data = crypto_data.reset_index()
                    # Ensure the crypto_data index is a DatetimeIndex
                    crypto_data.index = pd.to_datetime(crypto_data['Date'])

                    # Predict using the last available data to generate predictions for multiple days ahead
                    predictions = []
                    current_sequence = X_test[-1:]

                    for _ in range(days_to_predict):  # Number of days you want to predict
                        # Predict the next step and get the last predicted price
                        current_prediction = model.predict(current_sequence)
                        last_predicted_price = scaler.inverse_transform(current_prediction).flatten()[0]
                        predictions.append(last_predicted_price)

                        # Update the sequence with the last predicted price
                        current_sequence = np.roll(current_sequence, -1)
                        current_sequence[:, -1, :] = current_prediction

                    predicted_prices = np.array(predictions)

                    # Get the last known price from the historical data
                    last_known_price = scaler.inverse_transform(X_test[-1:, -1, 0].reshape(-1, 1)).flatten()[0]

                    # Calculate the prediction dates starting from the day after the last date in the index
                    prediction_dates = pd.date_range(start=crypto_data['Date'].iloc[-1] + pd.Timedelta(days=1),
                                                     periods=days_to_predict, freq='D')

                    # Generate a DataFrame with predictions and the corresponding dates
                    predicted_prices_df = pd.DataFrame({'Predicted_Close': predicted_prices}, index=prediction_dates)

                    # Generate trading signals based on the predicted prices
                    predicted_signals = generate_trading_signals(predicted_prices_df, last_known_price)

                    # Concatenate historical close prices with predicted prices
                    combined_data = pd.concat([crypto_data[['Close']], predicted_prices_df['Predicted_Close']])

                    # Display predicted prices and signals
                    st.write(f"Predicted prices for the next {days_to_predict} days with LSTM:")
                    st.dataframe(predicted_signals[['Predicted_Close', 'Signal']])

                    # Plot the historical and predicted prices
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=crypto_data['Date'], y=crypto_data['Close'], mode='lines',
                                             name='Historical Close'))
                    fig.add_trace(
                        go.Scatter(x=prediction_dates, y=predicted_prices, mode='lines', name='Predicted Close'))
                    fig.update_layout(title=f"{ticker} - Historical vs Predicted Close Prices",
                                      xaxis_title='Date', yaxis_title='Price')
                    st.plotly_chart(fig, use_container_width=True)

                    # show signals on the plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=crypto_data['Date'], y=crypto_data['Close'], mode='lines',
                                             name='Historical Close'))
                    fig.add_trace(
                        go.Scatter(x=prediction_dates, y=predicted_prices, mode='lines', name='Predicted Close'))
                    fig.add_trace(
                        go.Scatter(x=prediction_dates, y=predicted_prices, mode='markers', name='Predicted Close',
                                   marker=dict(color=predicted_signals['Signal'].map(
                                       {'Buy': 'green', 'Sell': 'red', 'Hold': 'blue'}))))

                    fig.update_layout(title=f"{ticker} - Historical vs Predicted Close Prices",
                                      xaxis_title='Date', yaxis_title='Price')
                    st.plotly_chart(fig, use_container_width=True)

                    # determine the signals and give advice
                    if predicted_signals['Signal'].str.contains('Buy').any():
                        st.success(f"🟢 Buy Advice: Consider buying {ticker}.")
                    elif predicted_signals['Signal'].str.contains('Sell').any():
                        st.error(f"🔴 Sell Advice: Consider selling {ticker}.")
                    else:
                        st.info(f"🔵 Hold Advice: Maintain your position in {ticker}.")


                elif model_choice == 'Prophet':
                    # Assuming the data is daily data
                    df_prophet = prepare_data_for_prophet(crypto_data, ticker)
                    model = train_prophet_model(df_prophet)
                    future = model.make_future_dataframe(periods=days_to_predict)
                    forecast = model.predict(future)

                    # Display forecast
                    st.write(f"Prophet forecast for the next {days_to_predict} days:")

                    # Historical data is everything in df_prophet up to the last actual date
                    historical = df_prophet[df_prophet['ds'] <= df_prophet['ds'].max()]

                    # Display the plot for the forecast
                    fig = plot_forecast(historical, forecast, days_to_predict)
                    st.plotly_chart(fig, use_container_width=True)

                    # Display the predicted prices
                    st.write(forecast[['ds', 'yhat']].tail(days_to_predict))

                elif model_choice == 'BI-LSTM':
                    # Assuming the prepare_lstm_data and other model functions are adjusted to handle data appropriately
                    X, y, scaler = prepare_lstm_data(crypto_data, 'Close', sequence_length=60)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                    bi_model = build_bi_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
                    bi_model, history = train_bi_lstm_model(bi_model, X_train, y_train, X_test, y_test)

                    # reset index to make 'Date' a column
                    crypto_data = crypto_data.reset_index()
                    # Ensure the crypto_data index is a DatetimeIndex
                    crypto_data.index = pd.to_datetime(crypto_data['Date'])

                    # Predict using the last available data to generate predictions for multiple days ahead
                    predictions = []
                    current_sequence = X_test[-1:]

                    for _ in range(days_to_predict):  # Number of days you want to predict
                        # Predict the next step and get the last predicted price
                        current_prediction = bi_model.predict(current_sequence)
                        last_predicted_price = scaler.inverse_transform(current_prediction).flatten()[0]
                        predictions.append(last_predicted_price)

                        # Update the sequence with the last predicted price
                        current_sequence = np.roll(current_sequence, -1)
                        current_sequence[:, -1, :] = current_prediction

                    predicted_prices = np.array(predictions)

                    # Get the last known price from the historical data
                    last_known_price = scaler.inverse_transform(X_test[-1:, -1, 0].reshape(-1, 1)).flatten()[0]

                    # Calculate the prediction dates starting from the day after the last date in the index
                    prediction_dates = pd.date_range(start=crypto_data['Date'].iloc[-1] + pd.Timedelta(days=1),
                                                     periods=days_to_predict, freq='D')

                    # Generate a DataFrame with predictions and the corresponding dates
                    predicted_prices_df = pd.DataFrame({'Predicted_Close': predicted_prices}, index=prediction_dates)

                    # Generate trading signals based on the predicted prices
                    predicted_signals = generate_trading_signals(predicted_prices_df, last_known_price)

                    # Concatenate historical close prices with predicted prices
                    combined_data = pd.concat([crypto_data[['Close']], predicted_prices_df['Predicted_Close']])

                    # Display predicted prices and signals
                    st.write(f"Predicted prices for the next {days_to_predict} days with Bi-LSTM:")
                    st.dataframe(predicted_signals[['Predicted_Close', 'Signal']])

                    # Plot the historical and predicted prices
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=crypto_data['Date'], y=crypto_data['Close'], mode='lines',
                                                name='Historical Close'))
                    fig.add_trace(
                        go.Scatter(x=prediction_dates, y=predicted_prices, mode='lines', name='Predicted Close'))
                    fig.update_layout(title=f"{ticker} - Historical vs Predicted Close Prices",
                                        xaxis_title='Date', yaxis_title='Price')
                    st.plotly_chart(fig, use_container_width=True)

                    # show trading signals on the plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=crypto_data['Date'], y=crypto_data['Close'], mode='lines',
                                             name='Historical Close'))
                    fig.add_trace(
                        go.Scatter(x=prediction_dates, y=predicted_prices, mode='lines', name='Predicted Close'))
                    fig.add_trace(
                        go.Scatter(x=prediction_dates, y=predicted_prices, mode='markers', name='Predicted Close',
                                   marker=dict(color=predicted_signals['Signal'].map(
                                       {'Buy': 'green', 'Sell': 'red', 'Hold': 'blue'}))))

                    fig.update_layout(title=f"{ticker} - Historical vs Predicted Close Prices",
                                        xaxis_title='Date', yaxis_title='Price')
                    st.plotly_chart(fig, use_container_width=True)

                    # determine the signals and give advice
                    if predicted_signals['Signal'].str.contains('Buy').any():
                        st.success(f"🟢 Buy Advice: Consider buying {ticker}.")
                    elif predicted_signals['Signal'].str.contains('Sell').any():
                        st.error(f"🔴 Sell Advice: Consider selling {ticker}.")
                    else:
                        st.info(f"🔵 Hold Advice: Maintain your position in {ticker}.")





                elif model_choice == 'ARIMA':
                    # reset index to make 'Date' a column
                    time_series_data = crypto_data.reset_index()

                    # Ensure the time series is indexed properly and is a Series
                    time_series_data.index = pd.to_datetime(time_series_data['Date'])
                    time_series_data = time_series_data['Close']
                    print(time_series_data.head())

                    # Ensure the time series is indexed properly and is a Series
                    time_series_data.index = pd.to_datetime(time_series_data.index)

                    # Find the best ARIMA model
                    auto_model = find_best_arima(time_series_data, seasonal=False)

                    # Fit the ARIMA model
                    model_fit = fit_arima_model(time_series_data, auto_model.order)

                    # Predict future prices
                    forecast = predict_arima(model_fit, days_to_predict)
                    # Generate forecast index starting from the last known date in the series
                    last_known_date = time_series_data.index.max()
                    forecast_index = pd.date_range(start=last_known_date + pd.Timedelta(days=1),
                                                   periods=days_to_predict, freq='D')
                    # Create forecast DataFrame
                    forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecast': forecast})
                    # Display the forecast
                    st.write(f"ARIMA forecast for the next {days_to_predict} days:")
                    fig = px.line(forecast_df, x='Date', y='Forecast', title=f"{ticker} - ARIMA Forecast")
                    st.plotly_chart(fig, use_container_width=True)


    else:
        st.error("Please ensure the cryptocurrency data is loaded and preprocessed.")


def trading_signals():
    st.header("Trading Signals")
    # Include code for generating trading signals
    st.write("Trading signal steps are shown here.")


# Conditional navigation based on sidebar choice
if menu_choice == "Overview":
    show_overview()
elif menu_choice == "About":
    show_about()
elif menu_choice == "Data Preprocessing":
    data_preprocessing()
elif menu_choice == "Exploratory Data Analysis":
    exploratory_data_analysis()
elif menu_choice == "Correlation Analysis":
    correlation_analysis()
elif menu_choice == "Prediction":
    prediction()
elif menu_choice == "Trading Signals":
    trading_signals()

# def train_arima_model(data, ticker, forecast_date):
#     # Make sure the index is a datetime index
#     data.index = pd.to_datetime(data.index)
#     close_prices = data['Close']
#
#     # Determine the train size and split the data
#     train_size = int(len(close_prices) * 0.8)
#     train, test = close_prices[:train_size], close_prices[train_size:]
#
#     # Fit the ARIMA model on the training set
#     model = auto_arima(train, seasonal=False, stepwise=True,
#                        suppress_warnings=True, error_action='ignore',
#                        max_order=None, trace=True)
#     model.fit(train)
#
#     # Forecast period includes the test set and the days to predict
#     forecast_period = len(test) + forecast_date
#
#     # Generate forecast index starting from the day after the last date in the test set
#     forecast_index = pd.date_range(start=test.index[-1] + pd.Timedelta(days=1), periods=forecast_period, freq='D')
#
#     # Forecast the values
#     forecast_values = model.predict(n_periods=forecast_period)
#     forecast = pd.Series(forecast_values[-forecast_date:], index=forecast_index, name='Forecast')
#
#     return train, test, forecast
