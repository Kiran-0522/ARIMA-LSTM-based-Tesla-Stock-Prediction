import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Title
st.title("Tesla Stock Price Prediction - LSTM vs ARIMA")

# Load CSV directly
df = pd.read_csv("Tesla_2020_2025.csv")  # File must be in same folder
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# EDA
st.subheader("Close Price Over Time")
st.line_chart(df['Close'])

st.subheader("Basic Statistics")
st.write(df['Close'].describe())

# Predict button
if st.button("Predict"):
    # Close price
    close_prices = df['Close']
    data = close_prices.values.reshape(-1, 1)

    # LSTM
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    y_pred_lstm = lstm_model.predict(X)
    predicted_lstm = scaler.inverse_transform(y_pred_lstm).reshape(-1)
    actual_lstm = scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)

    # ARIMA
    train_arima = close_prices[:-30]
    test_arima = close_prices[-30:]
    arima_model = ARIMA(train_arima, order=(5, 1, 0))
    arima_fit = arima_model.fit()
    forecast_arima = arima_fit.forecast(steps=30)

    actual_arima = test_arima.values.reshape(-1)
    predicted_arima = forecast_arima.values.reshape(-1)

    # Evaluation
    rmse_lstm = np.sqrt(mean_squared_error(actual_lstm[-30:], predicted_lstm[-30:]))
    mae_lstm = mean_absolute_error(actual_lstm[-30:], predicted_lstm[-30:])
    rmse_arima = np.sqrt(mean_squared_error(actual_arima, predicted_arima))
    mae_arima = mean_absolute_error(actual_arima, predicted_arima)

    st.subheader("Model Evaluation")
    st.write(f"LSTM  -> RMSE: {rmse_lstm:.2f}, MAE: {mae_lstm:.2f}")
    st.write(f"ARIMA -> RMSE: {rmse_arima:.2f}, MAE: {mae_arima:.2f}")

    # Plots
    st.subheader("ARIMA Forecast vs Actual")
    fig1 = plt.figure(figsize=(10, 4))
    plt.plot(train_arima, label='Train')
    plt.plot(test_arima.index, test_arima, label='Actual')
    plt.plot(test_arima.index, forecast_arima, label='Forecast')
    plt.legend()
    st.pyplot(fig1)

    st.subheader("LSTM Prediction vs Actual")
    fig2 = plt.figure(figsize=(10, 4))
    plt.plot(actual_lstm[-30:], label="Actual")
    plt.plot(predicted_lstm[-30:], label="Predicted")
    plt.legend()
    st.pyplot(fig2)
