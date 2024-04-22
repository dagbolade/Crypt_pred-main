import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score


def preprocess_data(data, features, target):
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    #data['Daily_Return'] = data[target].pct_change()  # Create a new feature
    return data.dropna()  # Remove rows with NaN values


def split_data(data, test_size=0.2):
    # Assuming 'Date' is the index of the data
    data = data.reset_index()  # Reset index to turn the Date column into a regular column
    train_size = int(len(data) * (1 - test_size))
    train, test = data.iloc[:train_size], data.iloc[train_size:]
    return train.set_index('Date'), test.set_index('Date')


def train_xgboost_model(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                             max_depth=5, alpha=10, n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    return model, preds, rmse, r2


def plot_xg_predictions(dates, y_test, y_pred, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(dates, y_test, label='Actual Price', color='blue')
    plt.plot(dates, y_pred, label='Predicted Price', color='red')
    plt.title(f'{ticker} - Actual vs Predicted Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def generate_features_for_next_prediction(last_row):
    # Create a new row for prediction with the necessary feature transformation
    new_row = last_row.copy()

    return new_row


def forecast_future_days(model, last_known_data, days):
    future_predictions = []
    current_features = last_known_data.copy() # Copy the last known data

    for _ in range(days):
        # Predict the next day
        y_pred = model.predict(current_features)
        future_predictions.append(y_pred[0])

        # Create new features for the next prediction
        next_features = generate_features_for_next_prediction(current_features.iloc[-1])

        # Append the new features for the next prediction
        # Ensuring that 'current_features' remains a DataFrame
        current_features = pd.concat([current_features, pd.DataFrame([next_features])]).reset_index(drop=True)

    return future_predictions
