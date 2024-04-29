# app/pages/data_preprocessing.py
import streamlit as st
import pandas as pd
import clustering
import correlation
import data_preprocessing
import data_transformation
import dimensionality_reduction
import feature_engineering
from Data_downolader import CryptoDataDownloader


def data_preprocessing_page():
    st.header("Data Preprocessing")

    # Cryptocurrency Data Downloader
    # Cache the data downloader function to prevent unnecessary downloads
    @st.cache_data
    def download_data(tickers, start_date, end_date, interval):
        downloader = CryptoDataDownloader(tickers, start_date=start_date, end_date=end_date, interval=interval)
        return downloader.download_data()

    # Sidebar for user inputs
    st.sidebar.header('Download Settings')
    all_cryptos = ['BNB-USD', 'ADA-USD', 'XRP-USD', 'DOGE-USD', 'DOT-USD', 'UNI-USD', 'BTC-USD', 'BCH-USD', 'LTC-USD', 'ETH-USD', 'LINK-USD', 'XLM-USD', 'USDC-USD', 'USDT-USD', 'VET-USD', 'ETC-USD', 'FIL-USD', 'TRX-USD', 'EOS-USD', 'THETA-USD', 'XMR-USD', 'NEO-USD', 'AAVE-USD', 'ATOM-USD', 'WIF-USD', 'BONK-USD', 'SHIB-USD', 'SOL-USD', 'DGB-USD', 'CHZ-USD', 'ENJ-USD', 'MANA-USD', 'BAT-USD', 'SAND-USD']

    # Use top_30_cryptos as default in multiselect
    selected_tickers = st.sidebar.multiselect('Select Cryptocurrencies, 30 cryptos have been added by default', options=all_cryptos, default=all_cryptos)

    start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2020-01-01'))
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
        st.write("Data Preprocessing and Transformation")

        # If 'Date' is in the index, reset index to make 'Date' a column
        if 'Date' in st.session_state.crypto_data.index.names:
            st.session_state.crypto_data.reset_index(inplace=True)
            st.write("'Date' column reset from index.")

        # Check again after resetting index
        if 'Date' not in st.session_state.crypto_data.columns:
            st.error("'Date' column is missing from the data after reset. Please check the data source.")
            st.stop()

        st.session_state.crypto_data = data_transformation.remove_duplicates(st.session_state.crypto_data, subset=['Ticker', 'Date'])
        st.write("Duplicates removed.")

        processed_data = data_transformation.pivot_and_fill(st.session_state.crypto_data, index='Ticker', columns='Date', values='Close')
        st.write("Data pivoted and filled.")
        st.dataframe(processed_data.head())

        scaled_data = data_transformation.scale_data(processed_data)
        st.write("Data scaled.")

        reduced_data_df, pca = dimensionality_reduction.apply_pca(scaled_data, n_components=10)
        st.write("PCA applied with 10 components.")
        st.dataframe(reduced_data_df.head())

        # Display explained variance plot
        fig = dimensionality_reduction.plot_explained_variance(pca, n_components=10)
        st.pyplot(fig)

        # Perform clustering
        clusters, kmeans = clustering.apply_kmeans(reduced_data_df, n_clusters=4)
        processed_data = clustering.add_cluster_labels(processed_data, clusters)
        st.write("K-means clustering applied with 4 clusters.")
        st.dataframe(processed_data.head())

        # Show the counts of cryptocurrencies in each cluster
        st.write("Cluster Distribution:")
        st.write(processed_data['Cluster'].value_counts())

        # Visualize clusters
        fig = clustering.plot_clusters(reduced_data_df.values, clusters)
        st.pyplot(fig)

        # Automatically select cryptos closest to centroids
        selected_cryptos = clustering.select_cryptos_closest_to_centroids(reduced_data_df, clusters, kmeans.cluster_centers_)
        st.write("Cryptocurrencies closest to each cluster centroid:")
        st.dataframe(selected_cryptos)

        # Convert 'Date' column to datetime and set as index
        st.session_state.crypto_data = data_preprocessing.convert_to_datetime(st.session_state.crypto_data, 'Date').set_index('Date')

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
        daily_returns = correlation.calculate_daily_returns(st.session_state.crypto_data)

        # Apply the feature engineering to the selected cryptos
        selected_cryptos_full = feature_engineering.calculate_sma_ema_rsi(selected_cryptos_full)
        st.write("Selected Cryptos Full Details with Feature Engineering:")
        st.write(selected_cryptos_full)

        # covert to datetime and set as index in selected cryptos full
        selected_cryptos_full = data_preprocessing.convert_to_datetime(selected_cryptos_full, 'Date').set_index('Date')
        st.write("Selected Cryptos Full Details with Feature Engineering:")
        st.write(selected_cryptos_full)

        # Store in session state
        st.session_state['selected_cryptos'] = selected_cryptos
        st.session_state['daily_returns'] = daily_returns
        st.session_state['selected_cryptos_full'] = selected_cryptos_full

        st.write("Preprocessing completed. Data stored in session state.")
    else:
        st.info("Please download data to proceed with transformation and analysis.")