import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import time
def fetch_stock_data(ticker_symbol, start_date, end_date):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    return stock_data
def preprocess_data(data):
    data['Next_Open'] = data['Open'].shift(-1)
    data['Next_High'] = data['High'].shift(-1)
    data['Next_Low'] = data['Low'].shift(-1)
    data['Next_Close'] = data['Close'].shift(-1)
    data['Next_Volume'] = data['Volume'].shift(-1)
    data.dropna(inplace=True)
    X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = data[['Next_Open', 'Next_High', 'Next_Low', 'Next_Close', 'Next_Volume']]
    return X, y
def predict_next(model, day_data):
    feature = np.array(day_data).reshape(1, -1)
    next_day_data = model.predict(feature)[0]
    return next_day_data
start = time.time()
ticker_symbol = "BHARTIARTL.BO"
start_date_training = "2021-01-01"
end_date_training = "2023-03-31"
start_date_prediction = "2023-04-01"
end_date_prediction = "2023-12-31"
stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
X_train, y_train = preprocess_data(stock_data_training)
num_iterations = 10
offset_dict = {}
for _ in range(num_iterations):
    working_day_counter = 1
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    stock_data_2024 = fetch_stock_data(ticker_symbol, start_date_prediction, end_date_prediction)
    prediction_dates = pd.date_range(start=start_date_prediction, end=end_date_prediction, freq='D').strftime('%Y-%m-%d')
    prediction_dates_filtered = [date for date in prediction_dates if date in stock_data_2024.index]
    current_day_data = stock_data_2024.iloc[0][['Open', 'High', 'Low', 'Close', 'Volume']]
    predicted_prices = []
    actual_prices = []
    for date in prediction_dates_filtered:
        next_day_data = predict_next(model, current_day_data)
        predicted_prices.append(next_day_data.tolist())
        current_day_data = pd.Series(next_day_data, index=current_day_data.index)
        actual_price_today = stock_data_2024.loc[date][['Open', 'High', 'Low', 'Close', 'Volume']]
        percentage_difference = ((next_day_data[3] - actual_price_today[3]) / actual_price_today[3]) * 100
        offset_dict[working_day_counter] = offset_dict.get(working_day_counter, 0) + percentage_difference
        working_day_counter += 1
average_offset_dict = {day: offset / num_iterations for day, offset in offset_dict.items()}
working_day_counter = 1
start_date_training = "2021-01-01"
end_date_training = "2023-12-31"
stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
X_train, y_train = preprocess_data(stock_data_training)
model = RandomForestRegressor() 
model.fit(X_train, y_train)
start_date_prediction = "2024-01-01"
end_date_prediction = "2024-02-29"
stock_data_2024 = fetch_stock_data(ticker_symbol, start_date_prediction, end_date_prediction)
prediction_dates = pd.date_range(start=start_date_prediction, end=end_date_prediction, freq='D').strftime('%Y-%m-%d')
prediction_dates_filtered = [date for date in prediction_dates if date in stock_data_2024.index]
current_day_data = stock_data_2024.iloc[0][['Open', 'High', 'Low', 'Close', 'Volume']]
predicted_prices = []
actual_prices = []
final_offset_dict = {}
for date in prediction_dates_filtered:
    next_day_data = predict_next(model, current_day_data)
    predicted_prices.append(next_day_data.tolist())
    current_day_data = pd.Series(next_day_data, index=current_day_data.index)
    actual_price_today = stock_data_2024.loc[date][['Open', 'High', 'Low', 'Close', 'Volume']]
    final_price = next_day_data[3] * ((100 - average_offset_dict[working_day_counter]) / 100)
    percentage_difference = ((final_price - actual_price_today[3]) / actual_price_today[3]) * 100
    final_offset_dict[working_day_counter] = final_offset_dict.get(working_day_counter, 0) + percentage_difference
    working_day_counter += 1
end = time.time()
print(end-start)
print(final_offset_dict)