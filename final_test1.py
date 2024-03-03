import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import datetime as dt
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
def main(StockName):
        
    start=time.time()
    ticker_symbol = StockName
    start_date_training = "1990-01-01"
    end_date_training = "2023-12-31"
    start_date_prediction = "2023-01-01"
    end_date_prediction = "2023-12-31"
    stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
    X_train, y_train = preprocess_data(stock_data_training)
    num_iterations = 10
    offset_dict = {}
    for _ in range(num_iterations):
        working_day_counter = 1
        model = LinearRegression()
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
    print("\n\n\n\n\n\n\n\n\n\n\n\n",len(average_offset_dict),"\n\n\n\n\n\n\n\n\n\n\n\n")
    working_day_counter = 1
    start_date_training = "1990-01-01"
    end_date_training = "2023-12-31"
    stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
    X_train, y_train = preprocess_data(stock_data_training)
    model = LinearRegression()
    model.fit(X_train, y_train)
    start_date_prediction = "2024-01-01"
    end_date_prediction = "2024-02-29"
    stock_data_2024 = fetch_stock_data(ticker_symbol, start_date_prediction, end_date_prediction)
    prediction_dates = pd.date_range(start=start_date_prediction, end=end_date_prediction, freq='D').strftime('%Y-%m-%d')
    prediction_dates_filtered = [date for date in prediction_dates if date in stock_data_2024.index]
    current_day_data = stock_data_2024.iloc[0][['Open', 'High', 'Low', 'Close', 'Volume']]
    predicted_prices = []
    actual_prices = []
    plus_offset_dict = {}
    minus_offset_dict = {}
    for date in prediction_dates_filtered:
        print(working_day_counter)
        print(date)
        next_day_data = predict_next(model, current_day_data)
        predicted_prices.append(next_day_data.tolist())
        current_day_data = pd.Series(next_day_data, index=current_day_data.index)
        actual_price_today = stock_data_2024.loc[date][['Open', 'High', 'Low', 'Close', 'Volume']]
        final_price = next_day_data[3] * ((100 - average_offset_dict[working_day_counter]) / 100)
        percentage_difference = ((final_price - actual_price_today[3]) / actual_price_today[3]) * 100
        minus_offset_dict[working_day_counter] = plus_offset_dict.get(working_day_counter, 0) + percentage_difference
        final_price = next_day_data[3] * ((100 - average_offset_dict[working_day_counter]) / 100)
        percentage_difference = ((final_price - actual_price_today[3]) / actual_price_today[3]) * 100
        plus_offset_dict[working_day_counter] = minus_offset_dict.get(working_day_counter, 0) + percentage_difference
        working_day_counter += 1
    plus=0
    minus=0
    for i in range (1,len(minus_offset_dict)):
        if math.fabs(minus_offset_dict[i])>math.fabs(plus_offset_dict[i]):
            plus+=1
        else:
            minus+=1
    working_day_counter = 1
    start_date_training = "1990-01-01"
    end_date_training = "2024-02-29"
    stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
    X_train, y_train = preprocess_data(stock_data_training)
    model = LinearRegression()
    model.fit(X_train, y_train)
    current_day_data = stock_data_training.iloc[-1][['Open', 'High', 'Low', 'Close', 'Volume']]
    predicted_prices = []
    if plus>minus:
        for _ in range(60):
            next_day_data = predict_next(model, current_day_data)
            price=next_day_data[3] * ((100 - average_offset_dict[working_day_counter]) / 100)
            predicted_prices.append(price)
            current_day_data = pd.Series(next_day_data, index=current_day_data.index)
            working_day_counter += 1
    else:
        for _ in range(60):
            next_day_data = predict_next(model, current_day_data)
            price=next_day_data[3] * ((100 + average_offset_dict[working_day_counter]) / 100)
            predicted_prices.append(price)
            current_day_data = pd.Series(next_day_data, index=current_day_data.index)
            working_day_counter += 1
    
    return predicted_prices
    # end=time.time()


def Find_Dates(prices):
    dates = [dt.datetime.now() - dt.timedelta(days=i) for i in range(len(prices))]
    return dates
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(dates, predicted_prices, marker='o', linestyle='-')
# ax.set_xlabel('Date')
# ax.set_ylabel('Value')
# ax.set_title('Graph of Values')
# ax.invert_xaxis()
# fig.savefig("C:/Users/khann/OneDrive/Desktop/images/graph.png")