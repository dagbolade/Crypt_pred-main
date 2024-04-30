import base64
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor
from keras._tf_keras import keras
from matplotlib import pyplot as plt
from pmdarima import auto_arima
from scipy.stats import stats, norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks

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
from trading_metrics import calculate_sharpe_ratio, calculate_max_drawdown, calculate_sortino_ratio
from trading_signals import generate_trading_signals, \
    generate_prophet1_trading_signals, plot_forecast_with_signals2, generate_arima_trading_signals, \
    plot_arima_forecast_with_signals, generate_lstm_trading_signals

from app.pages import data_preprocessing, eda, prediction, highest_return, trading_strategy, news, correlation, \
    desired_profit, model_evalaution


def show_overview():

    st.title("Cryptocurrency Analysis Dashboard")
    st.write("Welcome to the Cryptocurrency Analysis Dashboard!")
    st.write("This dashboard provides various tools and insights for analyzing cryptocurrency data.")
    st.write("Use the sidebar navigation to explore different sections of the app.")


def show_about():
    st.title("About")
    st.write(
        "This Cryptocurrency Analysis Dashboard is designed to help users of SoliGence analyze and visualize "
        "cryptocurrency data.")
    st.write("The app provides the following functionalities:")
    st.write("- Data Preprocessing: Clean and prepare cryptocurrency data for analysis.")
    st.write("- Correlation Analysis: Examine the correlations between different cryptocurrencies.")
    st.write("- Exploratory Data Analysis: Perform exploratory analysis and visualize cryptocurrency data.")
    st.write("- Prediction: Predict future prices of cryptocurrencies using machine learning models.")
    st.write("- Desired Profit: Calculate the required investment to achieve a desired profit.")
    st.write("- Highest Return Prediction: Identify cryptocurrencies with the highest potential return.")
    st.write("- Trading Strategy: Develop and backtest trading strategies for cryptocurrencies.")
    st.write("- News: Stay updated with the latest news and articles related to cryptocurrencies.")
    st.write("- Model Evaluation: Evaluate the performance of machine learning models used for predictions.")


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(jpg_file):
    bin_str = get_base64(jpg_file)
    page_bg_img = f'''
    <style>
    .stApp {{
      background-image: url("data:image/jpg;base64,{bin_str}");
      background-size: cover;
      background-position: center;
      background-attachment: fixed;
      background-repeat: no-repeat;
      background-color: #f5f5f5;
      
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


def main():
    # Set page configuration
    st.set_page_config(page_title="Cryptocurrency Analysis Dashboard", page_icon=":chart_with_upwards_trend:",
                       layout="wide")

    # Set the background image
    set_background('app/assets/background.jpg')
    # Apply custom styles to the sidebar
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Adding solent logo to the sidebar
    logo_path = "https://www.solent.ac.uk/graphics/logo/rebrandLogo.svg"
    st.sidebar.image(logo_path, width=150)

    st.sidebar.title("Navigation")
    pages = {
        "Overview": show_overview,
        "About": show_about,
        "Data Preprocessing": data_preprocessing.data_preprocessing_page,
        "Correlation Analysis": correlation.correlation_page,
        "Exploratory Data Analysis": eda.eda_page,
        "Prediction": prediction.prediction_page,
        "Desired Profit": desired_profit.desired_profit_page,
        "Highest Return Prediction": highest_return.highest_return_page,
        "Trading Strategy": trading_strategy.trading_strategy_page,
        "News": news.news_page,
        "Model Evaluation": model_evalaution.model_evaluation_page,
    }

    selection = st.sidebar.radio("Go to", list(pages.keys()))
    pages[selection]()


if __name__ == "__main__":
    main()
