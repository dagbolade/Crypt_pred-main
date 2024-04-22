# crypto_data.py

import yfinance as yf
import pandas as pd
import datetime


class CryptoDataDownloader:
    def __init__(self, tickers, start_date=None, end_date=None, interval='1d'):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval

    def download_data(self):
        all_crypto_data = pd.DataFrame()

        if self.start_date is None or self.end_date is None:
            self.start_date = datetime.datetime.now() - datetime.timedelta(days=831)  # Default to last year
            self.end_date = datetime.datetime.now()

        for ticker in self.tickers:
            print(f'Downloading historical data for {ticker}')
            try:
                data = yf.download(ticker, start=self.start_date.strftime('%Y-%m-%d'),
                                   end=self.end_date.strftime('%Y-%m-%d'), interval=self.interval)
                if not data.empty:
                    data['Ticker'] = ticker
                    all_crypto_data = pd.concat([all_crypto_data, data], ignore_index=False)
            except Exception as e:
                print(f"Failed to download data for {ticker}: {e}")

        return all_crypto_data

    def save_to_csv(self, dataframe, file_path):
        dataframe.to_csv(file_path, index=True)
        print(f"Data successfully saved to {file_path}")


# Use the class in your main workflow script
if __name__ == "__main__":
    # Define the list of cryptocurrency tickers
    crypto_tickers = [
        'BNB-USD', 'ADA-USD', 'XRP-USD', 'DOGE-USD', 'DOT-USD', 'UNI-USD', 'BTC-USD',
        'BCH-USD', 'LTC-USD', 'BTC-USD', 'ETH-USD', 'LINK-USD', 'XLM-USD', 'USDC-USD', 'USDT-USD', 'VET-USD', 'ETC-USD',
        'FIL-USD', 'TRX-USD', 'EOS-USD', 'THETA-USD', 'XMR-USD', 'NEO-USD', 'AAVE-USD', 'ATOM-USD', 'WIF-USD', 'BONK'
                                                                                                               '-USD', 'SHIB-USD', 'SOL-USD',
        'DGB-USD', 'CHZ-USD', 'ENJ-USD', 'MANA-USD', 'BAT-USD', 'SAND-USD',

    ]

    downloader = CryptoDataDownloader(crypto_tickers)
    historical_data = downloader.download_data() # call the download_data method to download the data
    csv_file_path = 'top_new_cryptos_historical_data.csv'
    downloader.save_to_csv(historical_data, csv_file_path)
