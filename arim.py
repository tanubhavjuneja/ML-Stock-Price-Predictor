import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import time
def fetch_stock_data(ticker_symbol, start_date, end_date):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    return stock_data
def preprocess_data(data):
    data['Close'] = data['Close'].fillna(method='ffill')
    return data['Close']
def predict_next(model, history):
    order = (5, 1, 0) 
    model_fit = model.fit()
    output = model_fit.forecast()
    next_day_prediction = output[0]
    return next_day_prediction
start = time.time()
ticker_symbol = "BHARTIARTL.BO"
start_date_training = "2021-01-01"
end_date_training = "2023-03-31"
start_date_prediction = "2023-04-01"
end_date_prediction = "2023-12-31"
stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
closing_prices_train = preprocess_data(stock_data_training)
num_iterations = 10
offset_dict = {}
for _ in range(num_iterations):
    working_day_counter = 1
    history = list(closing_prices_train)
    model = ARIMA(history, order=(5, 1, 0))  # Initialize ARIMA model
    stock_data_2024 = fetch_stock_data(ticker_symbol, start_date_prediction, end_date_prediction)
    closing_prices_test = preprocess_data(stock_data_2024)
    predicted_prices = []
    actual_prices = []
    for price in closing_prices_test:
        next_day_prediction = predict_next(model, history)
        predicted_prices.append(next_day_prediction)
        actual_price_today = price  # Assuming actual price is available
        percentage_difference = ((next_day_prediction - actual_price_today) / actual_price_today) * 100
        offset_dict[working_day_counter] = offset_dict.get(working_day_counter, 0) + percentage_difference
        history.append(actual_price_today)
        working_day_counter += 1
average_offset_dict = {day: offset / num_iterations for day, offset in offset_dict.items()}
working_day_counter = 1
start_date_training = "2021-01-01"
end_date_training = "2023-12-31"
stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
closing_prices_train = preprocess_data(stock_data_training)
start_date_prediction = "2024-01-01"
end_date_prediction = "2024-02-29"
stock_data_2024 = fetch_stock_data(ticker_symbol, start_date_prediction, end_date_prediction)
closing_prices_test = preprocess_data(stock_data_2024)
predicted_prices = []
actual_prices = []
final_offset_dict = {}
history = list(closing_prices_train)
model = ARIMA(history, order=(5, 1, 0))  
for price in closing_prices_test:
    next_day_prediction = predict_next(model, history)
    predicted_prices.append(next_day_prediction)
    actual_price_today = price 
    final_price = next_day_prediction * ((100 - average_offset_dict[working_day_counter]) / 100)
    percentage_difference = ((final_price - actual_price_today) / actual_price_today) * 100
    final_offset_dict[working_day_counter] = final_offset_dict.get(working_day_counter, 0) + percentage_difference
    history.append(actual_price_today)
    working_day_counter += 1
end = time.time()
print(end-start)
print(final_offset_dict)