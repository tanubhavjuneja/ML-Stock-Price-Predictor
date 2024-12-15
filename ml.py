import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import math
import numpy as np
from datetime import datetime,timedelta
import time
start=time.time()
def fetch_stock_data(ticker_symbol, start_date, end_date):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    return stock_data
def predict_lin_reg(model, day_data):
    feature = np.array(day_data).reshape(1, -1)
    next_day_data = model.predict(feature)[0]
    return next_day_data
def predict_lin_reg_mod(model1,model2,model3,model4,model5,day_data):
    feature = np.array(day_data).reshape(1, -1)
    next_day_open = model1.predict(feature)[0]
    next_day_high = model2.predict(feature)[0]
    next_day_low = model3.predict(feature)[0]
    next_day_close = model4.predict(feature)[0]
    next_day_volume = model5.predict(feature)[0]
    next_day_data=[next_day_open,next_day_high,next_day_low,next_day_close,next_day_volume]
    return next_day_data
def predict_gb(model1,model2,model3,model4,model5,day_data):
    feature = np.array(day_data).reshape(1, -1)
    next_day_open = model1.predict(feature)[0]
    next_day_high = model2.predict(feature)[0]
    next_day_low = model3.predict(feature)[0]
    next_day_close = model4.predict(feature)[0]
    next_day_volume = model5.predict(feature)[0]
    next_day_data=[next_day_open,next_day_high,next_day_low,next_day_close,next_day_volume]
    return next_day_data
def predict_ran_for_reg(model, day_data):
    feature = np.array(day_data).reshape(1, -1)
    next_day_data = model.predict(feature)[0]
    return next_day_data
def predict_arima(model1,model2,model3,model4,model5):
    output1 = model1.forecast(steps=1)
    next_day_open = output1[0]
    output2 = model2.forecast(steps=1)
    next_day_high = output2[0]
    output3 = model3.forecast(steps=1)
    next_day_low = output3[0]
    output4 = model4.forecast(steps=1)
    next_day_close = output4[0]
    output5 = model5.forecast(steps=1)
    next_day_volume = output5[0]
    next_day_data=[next_day_open,next_day_high,next_day_low,next_day_close,next_day_volume]
    return next_day_data
def check_offset(offset_list,working_day_counter,percentage_difference,next_day_data):
    for l in range(len(percentage_difference[working_day_counter])):
        if math.fabs(percentage_difference[working_day_counter][l])>1:
            next_day_data[l]*=(100-percentage_difference[working_day_counter][l])/100
            offset_list[working_day_counter][l]=percentage_difference[working_day_counter][l]
            percentage_difference[working_day_counter][l]=0
    return next_day_data,offset_list
def apply_offset(offset_list_,offset_list,next_day_data,working_day_counter):
    for el in range(len(offset_list[working_day_counter])):
        next_day_data[el]*=((100-((offset_list[working_day_counter][el]+(4*offset_list_[working_day_counter][el]))/5))/100)
    return next_day_data
def apply_final_offset(offset_list_2023,offset_list_2024,offset_list,next_day_data,working_day_counter):
    for el in range(len(offset_list_2024[working_day_counter])):
        next_day_data[el]*=((100-(((2*offset_list[working_day_counter][el])+(4*offset_list_2023[working_day_counter][el])+(4*offset_list_2024[working_day_counter][el]))/10))/100)
    return next_day_data
def check_lin_reg_model(iterations):
    def preprocess_lin_reg_data(data):
        data['Next_Open'] = data['Open'].shift(-1)
        data['Next_High'] = data['High'].shift(-1)
        data['Next_Low'] = data['Low'].shift(-1)
        data['Next_Close'] = data['Close'].shift(-1)
        data['Next_Volume'] = data['Volume'].shift(-1)
        data.dropna(inplace=True)
        X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        y = data[['Next_Open', 'Next_High', 'Next_Low', 'Next_Close', 'Next_Volume']]
        return X, y
    start_date_training = "2008-01-01"
    end_date_training = "2012-12-31"
    start_date_prediction = "2013-01-01"
    end_date_prediction = "2013-12-31"
    start_date_training = datetime.strptime(start_date_training, "%Y-%m-%d")
    end_date_training = datetime.strptime(end_date_training, "%Y-%m-%d")
    start_date_prediction = datetime.strptime(start_date_prediction, "%Y-%m-%d")
    end_date_prediction = datetime.strptime(end_date_prediction, "%Y-%m-%d")
    lin_reg_percentage_difference = [[0] * 5 for _ in range(250)]
    lin_reg_offset_list=[[0] * 5 for _ in range(250)]
    for _ in range(iterations):
        working_day_counter = 1
        stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
        stock_data_2024 = fetch_stock_data(ticker_symbol, start_date_prediction, end_date_prediction)
        prediction_dates = pd.date_range(start=start_date_prediction, end=end_date_prediction, freq='D').strftime('%Y-%m-%d')
        prediction_dates_filtered = [date for date in prediction_dates if date in stock_data_2024.index]
        X_train, y_train = preprocess_lin_reg_data(stock_data_training)
        model = LinearRegression()
        model.fit(X_train, y_train)
        current_day_data = stock_data_2024.iloc[0][['Open', 'High', 'Low', 'Close', 'Volume']]
        percentage_difference = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
        offset_list = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
        for date in prediction_dates_filtered:
            next_day_data = predict_lin_reg(model, current_day_data)
            actual_data_today = stock_data_2024.loc[date][['Open', 'High', 'Low', 'Close', 'Volume']]
            for k in range(len(next_day_data)):
                percentage_difference[working_day_counter][k] += ((next_day_data[k] - actual_data_today[k]) / next_day_data[k]) * 100
            next_day_data,offset_list=check_offset(offset_list,working_day_counter,percentage_difference,next_day_data)
            current_day_data = pd.Series(next_day_data, index=current_day_data.index)
            working_day_counter += 1
        for i in range(len(percentage_difference)):
            for j in range(len(next_day_data)):
                lin_reg_percentage_difference[i][j] += percentage_difference[i][j]/iterations
                lin_reg_offset_list[i][j] += offset_list[i][j]/iterations
        start_date_prediction += timedelta(days=365)
        end_date_prediction += timedelta(days=365)
        start_date_training += timedelta(days=365)
        end_date_training += timedelta(days=365)
    working_day_counter = 1
    start_date_training = "2020-01-01"
    end_date_training = "2023-03-08"
    start_date_prediction = "2023-03-09"
    end_date_prediction = "2023-12-31"
    stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
    stock_data_2024 = fetch_stock_data(ticker_symbol, start_date_prediction, end_date_prediction)
    prediction_dates = pd.date_range(start=start_date_prediction, end=end_date_prediction, freq='D').strftime('%Y-%m-%d')
    prediction_dates_filtered = [date for date in prediction_dates if date in stock_data_2024.index]
    X_train, y_train = preprocess_lin_reg_data(stock_data_training)
    model = LinearRegression()
    model.fit(X_train, y_train)
    current_day_data = stock_data_2024.iloc[0][['Open', 'High', 'Low', 'Close', 'Volume']]
    percentage_difference = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    lin_reg_offset_list_2023 = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    for date in prediction_dates_filtered:
        next_day_data = predict_lin_reg(model, current_day_data)
        actual_data_today = stock_data_2024.loc[date][['Open', 'High', 'Low', 'Close', 'Volume']]
        next_day_data=apply_offset(offset_list,lin_reg_offset_list,next_day_data,working_day_counter)
        for l in range(len(next_day_data)):
            percentage_difference[working_day_counter][l] += ((next_day_data[l] - actual_data_today[l]) / next_day_data[l]) * 100
        next_day_data,lin_reg_offset_list_2023=check_offset(lin_reg_offset_list_2023,working_day_counter,percentage_difference,next_day_data)
        current_day_data = pd.Series(next_day_data, index=current_day_data.index)
        working_day_counter += 1
    start_date_training = "2023-01-01"
    end_date_training = "2023-12-31"
    start_date_prediction = "2024-01-01"
    end_date_prediction = "2024-03-08"
    working_day_counter = 1
    stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
    stock_data_2024 = fetch_stock_data(ticker_symbol, start_date_prediction, end_date_prediction)
    prediction_dates = pd.date_range(start=start_date_prediction, end=end_date_prediction, freq='D').strftime('%Y-%m-%d')
    prediction_dates_filtered = [date for date in prediction_dates if date in stock_data_2024.index]
    X_train, y_train = preprocess_lin_reg_data(stock_data_training)
    model = LinearRegression()
    model.fit(X_train, y_train)
    current_day_data = stock_data_2024.iloc[0][['Open', 'High', 'Low', 'Close', 'Volume']]
    percentage_difference = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    lin_reg_offset_list_2024 = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    for date in prediction_dates_filtered:
        next_day_data = predict_lin_reg(model, current_day_data)
        actual_data_today = stock_data_2024.loc[date][['Open', 'High', 'Low', 'Close', 'Volume']]
        next_day_data=apply_offset(lin_reg_offset_list_2023,lin_reg_offset_list,next_day_data,working_day_counter)
        for l in range(len(next_day_data)):
            percentage_difference[working_day_counter][l] += ((next_day_data[l] - actual_data_today[l]) / next_day_data[l]) * 100
        next_day_data,lin_reg_offset_list_2024=check_offset(lin_reg_offset_list_2024,working_day_counter,percentage_difference,next_day_data)
        current_day_data = pd.Series(next_day_data, index=current_day_data.index)
        working_day_counter += 1
    lin_reg_list=[[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    for i in range(len(lin_reg_offset_list_2024)):
        for j in range(len(lin_reg_offset_list_2024[i])):
            lin_reg_list[i][j]=lin_reg_offset_list_2024[i][j]+percentage_difference[i][j]
    start_date_training = "2023-01-01"
    end_date_training = "2024-03-08"
    working_day_counter = 1
    stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
    X_train, y_train = preprocess_lin_reg_data(stock_data_training)
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(X_train, y_train)
    return lin_reg_list,lin_reg_model,lin_reg_offset_list,lin_reg_offset_list_2023,lin_reg_offset_list_2024
def check_lin_reg_mod_model(iterations):
    def preprocess_lin_reg_mod_data(data):
        data['Next_Open'] = data['Open'].shift(-1)
        data['Next_High'] = data['High'].shift(-1)
        data['Next_Low'] = data['Low'].shift(-1)
        data['Next_Close'] = data['Close'].shift(-1)
        data['Next_Volume'] = data['Volume'].shift(-1)
        data.dropna(inplace=True)
        X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        y1 = data['Next_Open']
        y2 = data['Next_High']
        y3 = data['Next_Low']
        y4 = data['Next_Close']
        y5 = data['Next_Volume']
        return X,y1,y2,y3,y4,y5
    start_date_training = "2008-01-01"
    end_date_training = "2012-12-31"
    start_date_prediction = "2013-01-01"
    end_date_prediction = "2013-12-31"
    start_date_training = datetime.strptime(start_date_training, "%Y-%m-%d")
    end_date_training = datetime.strptime(end_date_training, "%Y-%m-%d")
    start_date_prediction = datetime.strptime(start_date_prediction, "%Y-%m-%d")
    end_date_prediction = datetime.strptime(end_date_prediction, "%Y-%m-%d")
    lin_reg_mod_percentage_difference = [[0] * 5 for _ in range(250)]
    lin_reg_mod_offset_list=[[0] * 5 for _ in range(250)]
    for _ in range(iterations):
        working_day_counter = 1
        stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
        stock_data_2024 = fetch_stock_data(ticker_symbol, start_date_prediction, end_date_prediction)
        prediction_dates = pd.date_range(start=start_date_prediction, end=end_date_prediction, freq='D').strftime('%Y-%m-%d')
        prediction_dates_filtered = [date for date in prediction_dates if date in stock_data_2024.index]
        X_train, y1_train, y2_train, y3_train, y4_train, y5_train= preprocess_lin_reg_mod_data(stock_data_training)
        model1 = LinearRegression()
        model1.fit(X_train, y1_train)
        model2 = LinearRegression()
        model2.fit(X_train, y2_train)
        model3 = LinearRegression()
        model3.fit(X_train, y3_train)
        model4 = LinearRegression()
        model4.fit(X_train, y4_train)
        model5 = LinearRegression()
        model5.fit(X_train, y5_train)
        current_day_data = stock_data_2024.iloc[0][['Open', 'High', 'Low', 'Close', 'Volume']]
        percentage_difference = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
        offset_list = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
        for date in prediction_dates_filtered:
            next_day_data = predict_lin_reg_mod(model1,model2,model3,model4,model5,current_day_data)
            actual_data_today = stock_data_2024.loc[date][['Open', 'High', 'Low', 'Close', 'Volume']]
            for k in range(len(next_day_data)):
                percentage_difference[working_day_counter][k] += ((next_day_data[k] - actual_data_today[k]) / next_day_data[k]) * 100
            next_day_data,offset_list=check_offset(offset_list,working_day_counter,percentage_difference,next_day_data)
            current_day_data = pd.Series(next_day_data, index=current_day_data.index)
            working_day_counter += 1
        for i in range(len(percentage_difference)):
            for j in range(len(next_day_data)):
                lin_reg_mod_percentage_difference[i][j] += percentage_difference[i][j]/iterations
                lin_reg_mod_offset_list[i][j] += offset_list[i][j]/iterations
        start_date_prediction += timedelta(days=365)
        end_date_prediction += timedelta(days=365)
        start_date_training += timedelta(days=365)
        end_date_training += timedelta(days=365)
    working_day_counter = 1
    start_date_training = "2020-01-01"
    end_date_training = "2023-03-08"
    start_date_prediction = "2023-03-09"
    end_date_prediction = "2023-12-31"
    stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
    stock_data_2024 = fetch_stock_data(ticker_symbol, start_date_prediction, end_date_prediction)
    prediction_dates = pd.date_range(start=start_date_prediction, end=end_date_prediction, freq='D').strftime('%Y-%m-%d')
    prediction_dates_filtered = [date for date in prediction_dates if date in stock_data_2024.index]
    X_train, y1_train, y2_train, y3_train, y4_train, y5_train= preprocess_lin_reg_mod_data(stock_data_training)
    model1 = LinearRegression()
    model1.fit(X_train, y1_train)
    model2 = LinearRegression()
    model2.fit(X_train, y2_train)
    model3 = LinearRegression()
    model3.fit(X_train, y3_train)
    model4 = LinearRegression()
    model4.fit(X_train, y4_train)
    model5 = LinearRegression()
    model5.fit(X_train, y5_train)
    current_day_data = stock_data_2024.iloc[0][['Open', 'High', 'Low', 'Close', 'Volume']]
    percentage_difference = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    lin_reg_mod_offset_list_2023 = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    for date in prediction_dates_filtered:
        next_day_data = predict_lin_reg_mod(model1,model2,model3,model4,model5,current_day_data)
        actual_data_today = stock_data_2024.loc[date][['Open', 'High', 'Low', 'Close', 'Volume']]
        next_day_data=apply_offset(offset_list,lin_reg_mod_offset_list,next_day_data,working_day_counter)
        for l in range(len(next_day_data)):
            percentage_difference[working_day_counter][l] += ((next_day_data[l] - actual_data_today[l]) / next_day_data[l]) * 100
        next_day_data,lin_reg_mod_offset_list_2023=check_offset(lin_reg_mod_offset_list_2023,working_day_counter,percentage_difference,next_day_data)
        current_day_data = pd.Series(next_day_data, index=current_day_data.index)
        working_day_counter += 1
    start_date_training = "2023-01-01"
    end_date_training = "2023-12-31"
    start_date_prediction = "2024-01-01"
    end_date_prediction = "2024-03-08"
    working_day_counter = 1
    stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
    stock_data_2024 = fetch_stock_data(ticker_symbol, start_date_prediction, end_date_prediction)
    prediction_dates = pd.date_range(start=start_date_prediction, end=end_date_prediction, freq='D').strftime('%Y-%m-%d')
    prediction_dates_filtered = [date for date in prediction_dates if date in stock_data_2024.index]
    X_train, y1_train, y2_train, y3_train, y4_train, y5_train= preprocess_lin_reg_mod_data(stock_data_training)
    model1 = LinearRegression()
    model1.fit(X_train, y1_train)
    model2 = LinearRegression()
    model2.fit(X_train, y2_train)
    model3 = LinearRegression()
    model3.fit(X_train, y3_train)
    model4 = LinearRegression()
    model4.fit(X_train, y4_train)
    model5 = LinearRegression()
    model5.fit(X_train, y5_train)
    current_day_data = stock_data_2024.iloc[0][['Open', 'High', 'Low', 'Close', 'Volume']]
    percentage_difference = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    lin_reg_mod_offset_list_2024 = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    for date in prediction_dates_filtered:
        next_day_data = predict_lin_reg_mod(model1,model2,model3,model4,model5,current_day_data)
        actual_data_today = stock_data_2024.loc[date][['Open', 'High', 'Low', 'Close', 'Volume']]
        next_day_data=apply_offset(lin_reg_mod_offset_list_2023,lin_reg_mod_offset_list,next_day_data,working_day_counter)
        for l in range(len(next_day_data)):
            percentage_difference[working_day_counter][l] += ((next_day_data[l] - actual_data_today[l]) / next_day_data[l]) * 100
        next_day_data,lin_reg_mod_offset_list_2024=check_offset(lin_reg_mod_offset_list_2024,working_day_counter,percentage_difference,next_day_data)
        current_day_data = pd.Series(next_day_data, index=current_day_data.index)
        working_day_counter += 1
    lin_reg_mod_list=[[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    for i in range(len(lin_reg_mod_offset_list_2024)):
        for j in range(len(lin_reg_mod_offset_list_2024[i])):
            lin_reg_mod_list[i][j]=lin_reg_mod_offset_list_2024[i][j]+percentage_difference[i][j]
    start_date_training = "2023-01-01"
    end_date_training = "2024-03-08"
    working_day_counter = 1
    stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
    X_train, y1_train, y2_train, y3_train, y4_train, y5_train= preprocess_lin_reg_mod_data(stock_data_training)
    lin_reg_mod_model1 = LinearRegression()
    lin_reg_mod_model1.fit(X_train, y1_train)
    lin_reg_mod_model2 = LinearRegression()
    lin_reg_mod_model2.fit(X_train, y2_train)
    lin_reg_mod_model3 = LinearRegression()
    lin_reg_mod_model3.fit(X_train, y3_train)
    lin_reg_mod_model4 = LinearRegression()
    lin_reg_mod_model4.fit(X_train, y4_train)
    lin_reg_mod_model5 = LinearRegression()
    lin_reg_mod_model5.fit(X_train, y5_train)
    return lin_reg_mod_list,lin_reg_mod_model1,lin_reg_mod_model2,lin_reg_mod_model3,lin_reg_mod_model4,lin_reg_mod_model5,lin_reg_mod_offset_list,lin_reg_mod_offset_list_2023,lin_reg_mod_offset_list_2024
def check_gra_boo_model(iterations):
    def preprocess_gra_boo_data(data):
        data['Next_Open'] = data['Open'].shift(-1)
        data['Next_High'] = data['High'].shift(-1)
        data['Next_Low'] = data['Low'].shift(-1)
        data['Next_Close'] = data['Close'].shift(-1)
        data['Next_Volume'] = data['Volume'].shift(-1)
        data.dropna(inplace=True)
        X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        y1 = data['Next_Open']
        y2 = data['Next_High']
        y3 = data['Next_Low']
        y4 = data['Next_Close']
        y5 = data['Next_Volume']
        return X,y1,y2,y3,y4,y5
    start_date_training = "2008-01-01"
    end_date_training = "2012-12-31"
    start_date_prediction = "2013-01-01"
    end_date_prediction = "2013-12-31"
    start_date_training = datetime.strptime(start_date_training, "%Y-%m-%d")
    end_date_training = datetime.strptime(end_date_training, "%Y-%m-%d")
    start_date_prediction = datetime.strptime(start_date_prediction, "%Y-%m-%d")
    end_date_prediction = datetime.strptime(end_date_prediction, "%Y-%m-%d")
    gra_boo_percentage_difference = [[0] * 5 for _ in range(250)]
    gra_boo_offset_list=[[0] * 5 for _ in range(250)]
    for _ in range(iterations):
        working_day_counter = 1
        stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
        stock_data_2024 = fetch_stock_data(ticker_symbol, start_date_prediction, end_date_prediction)
        prediction_dates = pd.date_range(start=start_date_prediction, end=end_date_prediction, freq='D').strftime('%Y-%m-%d')
        prediction_dates_filtered = [date for date in prediction_dates if date in stock_data_2024.index]
        X_train, y1_train, y2_train, y3_train, y4_train, y5_train= preprocess_gra_boo_data(stock_data_training)
        model1 = GradientBoostingRegressor(n_estimators=500,learning_rate=0.01,max_depth=7,min_samples_split=2,min_samples_leaf=1,max_features='log2')
        model1.fit(X_train, y1_train)
        model2 = GradientBoostingRegressor(n_estimators=500,learning_rate=0.01,max_depth=7,min_samples_split=2,min_samples_leaf=1,max_features='log2')
        model2.fit(X_train, y2_train)
        model3 = GradientBoostingRegressor(n_estimators=500,learning_rate=0.01,max_depth=7,min_samples_split=2,min_samples_leaf=1,max_features='log2')
        model3.fit(X_train, y3_train)
        model4 = GradientBoostingRegressor(n_estimators=500,learning_rate=0.01,max_depth=7,min_samples_split=2,min_samples_leaf=1,max_features='log2')
        model4.fit(X_train, y4_train)
        model5 = GradientBoostingRegressor(n_estimators=500,learning_rate=0.01,max_depth=7,min_samples_split=2,min_samples_leaf=1,max_features='log2')
        model5.fit(X_train, y5_train)
        current_day_data = stock_data_2024.iloc[0][['Open', 'High', 'Low', 'Close', 'Volume']]
        percentage_difference = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
        offset_list = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
        for date in prediction_dates_filtered:
            next_day_data = predict_gb(model1,model2,model3,model4,model5,current_day_data)
            actual_data_today = stock_data_2024.loc[date][['Open', 'High', 'Low', 'Close', 'Volume']]
            for k in range(len(next_day_data)):
                percentage_difference[working_day_counter][k] += math.fabs((next_day_data[k] - actual_data_today[k]) / next_day_data[k]) * 100
            next_day_data,offset_list=check_offset(offset_list,working_day_counter,percentage_difference,next_day_data)
            current_day_data = pd.Series(next_day_data, index=current_day_data.index)
            working_day_counter += 1
        for i in range(len(percentage_difference)):
            for j in range(len(next_day_data)):
                gra_boo_percentage_difference[i][j] += percentage_difference[i][j] / iterations
                gra_boo_offset_list[i][j] += offset_list[i][j] / iterations
        start_date_prediction += timedelta(days=365)
        end_date_prediction += timedelta(days=365)
        start_date_training += timedelta(days=365)
        end_date_training += timedelta(days=365)
    working_day_counter = 1
    start_date_training = "2020-01-01"
    end_date_training = "2023-03-08"
    start_date_prediction = "2023-03-09"
    end_date_prediction = "2023-12-31"
    stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
    stock_data_2024 = fetch_stock_data(ticker_symbol, start_date_prediction, end_date_prediction)
    prediction_dates = pd.date_range(start=start_date_prediction, end=end_date_prediction, freq='D').strftime('%Y-%m-%d')
    prediction_dates_filtered = [date for date in prediction_dates if date in stock_data_2024.index]
    X_train, y1_train, y2_train, y3_train, y4_train, y5_train= preprocess_gra_boo_data(stock_data_training)
    model1 = GradientBoostingRegressor(n_estimators=500,learning_rate=0.01,max_depth=7,min_samples_split=2,min_samples_leaf=1,max_features='log2')
    model1.fit(X_train, y1_train)
    model2 = GradientBoostingRegressor(n_estimators=500,learning_rate=0.01,max_depth=7,min_samples_split=2,min_samples_leaf=1,max_features='log2')
    model2.fit(X_train, y2_train)
    model3 = GradientBoostingRegressor(n_estimators=500,learning_rate=0.01,max_depth=7,min_samples_split=2,min_samples_leaf=1,max_features='log2')
    model3.fit(X_train, y3_train)
    model4 = GradientBoostingRegressor(n_estimators=500,learning_rate=0.01,max_depth=7,min_samples_split=2,min_samples_leaf=1,max_features='log2')
    model4.fit(X_train, y4_train)
    model5 = GradientBoostingRegressor(n_estimators=500,learning_rate=0.01,max_depth=7,min_samples_split=2,min_samples_leaf=1,max_features='log2')
    model5.fit(X_train, y5_train)
    current_day_data = stock_data_2024.iloc[0][['Open', 'High', 'Low', 'Close', 'Volume']]
    percentage_difference = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    gra_boo_offset_list_2023 = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    for date in prediction_dates_filtered:
        next_day_data = predict_gb(model1,model2,model3,model4,model5,current_day_data)
        actual_data_today = stock_data_2024.loc[date][['Open', 'High', 'Low', 'Close', 'Volume']]
        next_day_data=apply_offset(offset_list,gra_boo_offset_list,next_day_data,working_day_counter)
        for l in range(len(next_day_data)):
            percentage_difference[working_day_counter][l] += ((next_day_data[l] - actual_data_today[l]) / next_day_data[l]) * 100
        next_day_data,gra_boo_offset_list_2023=check_offset(gra_boo_offset_list_2023,working_day_counter,percentage_difference,next_day_data)
        current_day_data = pd.Series(next_day_data, index=current_day_data.index)
        working_day_counter += 1
    start_date_training = "2023-01-01"
    end_date_training = "2023-12-31"
    start_date_prediction = "2024-01-01"
    end_date_prediction = "2024-03-08"
    working_day_counter = 1
    stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
    stock_data_2024 = fetch_stock_data(ticker_symbol, start_date_prediction, end_date_prediction)
    prediction_dates = pd.date_range(start=start_date_prediction, end=end_date_prediction, freq='D').strftime('%Y-%m-%d')
    prediction_dates_filtered = [date for date in prediction_dates if date in stock_data_2024.index]
    X_train, y1_train, y2_train, y3_train, y4_train, y5_train= preprocess_gra_boo_data(stock_data_training)
    model1 = GradientBoostingRegressor(n_estimators=500,learning_rate=0.01,max_depth=7,min_samples_split=2,min_samples_leaf=1,max_features='log2')
    model1.fit(X_train, y1_train)
    model2 = GradientBoostingRegressor(n_estimators=500,learning_rate=0.01,max_depth=7,min_samples_split=2,min_samples_leaf=1,max_features='log2')
    model2.fit(X_train, y2_train)
    model3 = GradientBoostingRegressor(n_estimators=500,learning_rate=0.01,max_depth=7,min_samples_split=2,min_samples_leaf=1,max_features='log2')
    model3.fit(X_train, y3_train)
    model4 = GradientBoostingRegressor(n_estimators=500,learning_rate=0.01,max_depth=7,min_samples_split=2,min_samples_leaf=1,max_features='log2')
    model4.fit(X_train, y4_train)
    model5 = GradientBoostingRegressor(n_estimators=500,learning_rate=0.01,max_depth=7,min_samples_split=2,min_samples_leaf=1,max_features='log2')
    model5.fit(X_train, y5_train)
    current_day_data = stock_data_2024.iloc[0][['Open', 'High', 'Low', 'Close', 'Volume']]
    percentage_difference = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    gra_boo_offset_list_2024 = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    for date in prediction_dates_filtered:
        next_day_data = predict_gb(model1,model2,model3,model4,model5,current_day_data)
        actual_data_today = stock_data_2024.loc[date][['Open', 'High', 'Low', 'Close', 'Volume']]
        next_day_data=apply_offset(gra_boo_offset_list_2023,gra_boo_offset_list,next_day_data,working_day_counter)
        for l in range(len(next_day_data)):
            percentage_difference[working_day_counter][l] += ((next_day_data[l] - actual_data_today[l]) / next_day_data[l]) * 100
        next_day_data,gra_boo_offset_list_2024=check_offset(gra_boo_offset_list_2024,working_day_counter,percentage_difference,next_day_data)
        current_day_data = pd.Series(next_day_data, index=current_day_data.index)
        working_day_counter += 1
    gra_boo_list=[[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    for i in range(len(gra_boo_offset_list_2024)):
        for j in range(len(gra_boo_offset_list_2024[i])):
            gra_boo_list[i][j]=gra_boo_offset_list_2024[i][j]+percentage_difference[i][j]
    start_date_training = "2023-01-01"
    end_date_training = "2024-03-08"
    working_day_counter = 1
    stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
    X_train, y1_train, y2_train, y3_train, y4_train, y5_train= preprocess_gra_boo_data(stock_data_training)
    gra_boo_model1 = GradientBoostingRegressor(n_estimators=500,learning_rate=0.01,max_depth=7,min_samples_split=2,min_samples_leaf=1,max_features='log2')
    gra_boo_model1.fit(X_train, y1_train)
    gra_boo_model2 = GradientBoostingRegressor(n_estimators=500,learning_rate=0.01,max_depth=7,min_samples_split=2,min_samples_leaf=1,max_features='log2')
    gra_boo_model2.fit(X_train, y2_train)
    gra_boo_model3 = GradientBoostingRegressor(n_estimators=500,learning_rate=0.01,max_depth=7,min_samples_split=2,min_samples_leaf=1,max_features='log2')
    gra_boo_model3.fit(X_train, y3_train)
    gra_boo_model4 = GradientBoostingRegressor(n_estimators=500,learning_rate=0.01,max_depth=7,min_samples_split=2,min_samples_leaf=1,max_features='log2')
    gra_boo_model4.fit(X_train, y4_train)
    gra_boo_model5 = GradientBoostingRegressor(n_estimators=500,learning_rate=0.01,max_depth=7,min_samples_split=2,min_samples_leaf=1,max_features='log2')
    gra_boo_model5.fit(X_train, y5_train)
    return gra_boo_list,gra_boo_model1,gra_boo_model2,gra_boo_model3,gra_boo_model4,gra_boo_model5,gra_boo_offset_list,gra_boo_offset_list_2023,gra_boo_offset_list_2024,stock_data_training
def check_ran_for_reg_model(iterations):
    def preprocess_ran_for_reg_data(data):
        data['Next_Open'] = data['Open'].shift(-1)
        data['Next_High'] = data['High'].shift(-1)
        data['Next_Low'] = data['Low'].shift(-1)
        data['Next_Close'] = data['Close'].shift(-1)
        data['Next_Volume'] = data['Volume'].shift(-1)
        data.dropna(inplace=True)
        X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        y = data[['Next_Open', 'Next_High', 'Next_Low', 'Next_Close', 'Next_Volume']]
        return X, y
    start_date_training = "2008-01-01"
    end_date_training = "2012-12-31"
    start_date_prediction = "2013-01-01"
    end_date_prediction = "2013-12-31"
    start_date_training = datetime.strptime(start_date_training, "%Y-%m-%d")
    end_date_training = datetime.strptime(end_date_training, "%Y-%m-%d")
    start_date_prediction = datetime.strptime(start_date_prediction, "%Y-%m-%d")
    end_date_prediction = datetime.strptime(end_date_prediction, "%Y-%m-%d")
    ran_for_reg_percentage_difference = [[0] * 5 for _ in range(250)]
    ran_for_reg_offset_list = [[0] * 5 for _ in range(250)]
    for _ in range(iterations):
        working_day_counter = 1
        stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
        stock_data_2024 = fetch_stock_data(ticker_symbol, start_date_prediction, end_date_prediction)
        prediction_dates = pd.date_range(start=start_date_prediction, end=end_date_prediction, freq='D').strftime('%Y-%m-%d')
        prediction_dates_filtered = [date for date in prediction_dates if date in stock_data_2024.index]
        X_train, y_train = preprocess_ran_for_reg_data(stock_data_training)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        current_day_data = stock_data_2024.iloc[0][['Open', 'High', 'Low', 'Close', 'Volume']]
        percentage_difference = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
        offset_list = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
        for date in prediction_dates_filtered:
            next_day_data = predict_ran_for_reg(model, current_day_data)
            actual_data_today = stock_data_2024.loc[date][['Open', 'High', 'Low', 'Close', 'Volume']]
            for k in range(len(next_day_data)):
                percentage_difference[working_day_counter][k] += math.fabs((next_day_data[k] - actual_data_today[k]) / actual_data_today[k]) * 100
            next_day_data,offset_list=check_offset(offset_list,working_day_counter,percentage_difference,next_day_data)
            current_day_data = pd.Series(next_day_data, index=current_day_data.index)
            working_day_counter += 1
        for i in range(len(percentage_difference)):
            for j in range(len(next_day_data)):
                ran_for_reg_percentage_difference[i][j] += percentage_difference[i][j]/iterations
                ran_for_reg_offset_list[i][j] += offset_list[i][j]/iterations
        start_date_prediction += timedelta(days=365)
        end_date_prediction += timedelta(days=365)
        start_date_training += timedelta(days=365)
        end_date_training += timedelta(days=365)
    working_day_counter = 1
    start_date_training = "2020-01-01"
    end_date_training = "2023-03-08"
    start_date_prediction = "2023-03-09"
    end_date_prediction = "2023-12-31"
    stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
    stock_data_2024 = fetch_stock_data(ticker_symbol, start_date_prediction, end_date_prediction)
    prediction_dates = pd.date_range(start=start_date_prediction, end=end_date_prediction, freq='D').strftime('%Y-%m-%d')
    prediction_dates_filtered = [date for date in prediction_dates if date in stock_data_2024.index]
    X_train, y_train = preprocess_ran_for_reg_data(stock_data_training)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    current_day_data = stock_data_2024.iloc[0][['Open', 'High', 'Low', 'Close', 'Volume']]
    percentage_difference = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    ran_for_reg_offset_list_2023 = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    for date in prediction_dates_filtered:
        next_day_data = predict_ran_for_reg(model, current_day_data)
        actual_data_today = stock_data_2024.loc[date][['Open', 'High', 'Low', 'Close', 'Volume']]
        next_day_data=apply_offset(offset_list,ran_for_reg_offset_list,next_day_data,working_day_counter)
        for l in range(len(next_day_data)):
            percentage_difference[working_day_counter][l] += ((next_day_data[l] - actual_data_today[l]) / next_day_data[l]) * 100
        next_day_data,ran_for_reg_offset_list_2023=check_offset(ran_for_reg_offset_list_2023,working_day_counter,percentage_difference,next_day_data)
        current_day_data = pd.Series(next_day_data, index=current_day_data.index)
        working_day_counter += 1
    start_date_training = "2023-01-01"
    end_date_training = "2023-12-31"
    start_date_prediction = "2024-01-01"
    end_date_prediction = "2024-03-08"
    working_day_counter = 1
    stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
    stock_data_2024 = fetch_stock_data(ticker_symbol, start_date_prediction, end_date_prediction)
    prediction_dates = pd.date_range(start=start_date_prediction, end=end_date_prediction, freq='D').strftime('%Y-%m-%d')
    prediction_dates_filtered = [date for date in prediction_dates if date in stock_data_2024.index]
    X_train, y_train = preprocess_ran_for_reg_data(stock_data_training)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    current_day_data = stock_data_2024.iloc[0][['Open', 'High', 'Low', 'Close', 'Volume']]
    percentage_difference = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    ran_for_reg_offset_list_2024 = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    for date in prediction_dates_filtered:
        next_day_data = predict_ran_for_reg(model, current_day_data)
        actual_data_today = stock_data_2024.loc[date][['Open', 'High', 'Low', 'Close', 'Volume']]
        next_day_data=apply_offset(ran_for_reg_offset_list_2023,ran_for_reg_offset_list,next_day_data,working_day_counter)
        for l in range(len(next_day_data)):
            percentage_difference[working_day_counter][l] += ((next_day_data[l] - actual_data_today[l]) / next_day_data[l]) * 100
        next_day_data,ran_for_reg_offset_list_2024=check_offset(ran_for_reg_offset_list_2024,working_day_counter,percentage_difference,next_day_data)
        current_day_data = pd.Series(next_day_data, index=current_day_data.index)
        working_day_counter += 1
    ran_for_reg_list=[[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    for i in range(len(ran_for_reg_offset_list_2024)):
        for j in range(len(ran_for_reg_offset_list_2024[i])):
            ran_for_reg_list[i][j]=ran_for_reg_offset_list_2024[i][j]+percentage_difference[i][j]
    start_date_training = "2023-01-01"
    end_date_training = "2024-03-08"
    working_day_counter = 1
    stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
    X_train, y_train = preprocess_ran_for_reg_data(stock_data_training)
    ran_for_reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    ran_for_reg_model.fit(X_train, y_train)
    return ran_for_reg_list,ran_for_reg_model,ran_for_reg_offset_list,ran_for_reg_offset_list_2023,ran_for_reg_offset_list_2024
def check_arima_model(iterations):
    global arima_percentage_difference,arima_offset_list
    def preprocess_arima_data(data):
        data['Open'] = data['Open'].fillna(method='ffill')
        data['High'] = data['High'].fillna(method='ffill')
        data['Low'] = data['Low'].fillna(method='ffill')
        data['Close'] = data['Close'].fillna(method='ffill')
        data['Volume'] = data['Volume'].fillna(method='ffill')
        return data['Open'],data['High'],data['Low'],data['Close'],data['Volume']
    start_date_training = "2008-01-01"
    end_date_training = "2012-12-31"
    start_date_prediction = "2013-01-01"
    end_date_prediction = "2013-12-31"
    start_date_training = datetime.strptime(start_date_training, "%Y-%m-%d")
    end_date_training = datetime.strptime(end_date_training, "%Y-%m-%d")
    start_date_prediction = datetime.strptime(start_date_prediction, "%Y-%m-%d")
    end_date_prediction = datetime.strptime(end_date_prediction, "%Y-%m-%d")
    arima_percentage_difference = [[0] * 5 for _ in range(250)]
    arima_offset_list=[[0] * 5 for _ in range(250)]
    for _ in range(iterations):
        working_day_counter = 1
        stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
        stock_data_2024 = fetch_stock_data(ticker_symbol, start_date_prediction, end_date_prediction)
        prediction_dates = pd.date_range(start=start_date_prediction, end=end_date_prediction, freq='D').strftime('%Y-%m-%d')
        prediction_dates_filtered = [date for date in prediction_dates if date in stock_data_2024.index]
        y1_train,y2_train,y3_train,y4_train,y5_train = preprocess_arima_data(stock_data_training)
        p, d, q = 0, 0, 2 
        history1 = list(y1_train)
        history2 = list(y2_train)
        history3 = list(y3_train)
        history4 = list(y4_train)
        history5 = list(y5_train)
        model1 = ARIMA(history1, order=(p, d, q))
        model1_fit=model1.fit()
        history2 = list(y2_train)
        model2 = ARIMA(history2, order=(p, d, q))
        model2_fit=model2.fit()
        history3 = list(y3_train)
        model3 = ARIMA(history3, order=(p, d, q))
        model3_fit=model3.fit()
        history4 = list(y4_train)
        model4 = ARIMA(history4, order=(p, d, q))
        model4_fit=model4.fit()
        history5 = list(y5_train)
        model5 = ARIMA(history5, order=(p, d, q))
        model5_fit=model5.fit()
        current_day_data = stock_data_2024.iloc[0][['Open', 'High', 'Low', 'Close', 'Volume']]
        percentage_difference = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
        offset_list = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
        for date in prediction_dates_filtered:
            next_day_data = predict_arima(model1_fit,model2_fit,model3_fit,model4_fit,model5_fit)
            actual_data_today = stock_data_2024.loc[date][['Open', 'High', 'Low', 'Close', 'Volume']]
            for k in range(len(next_day_data)):
                percentage_difference[working_day_counter][k] += math.fabs((next_day_data[k] - actual_data_today[k]) / actual_data_today[k]) * 100
            next_day_data,offset_list=check_offset(offset_list,working_day_counter,percentage_difference,next_day_data)
            current_day_data = pd.Series(next_day_data, index=current_day_data.index)
            working_day_counter += 1
        for i in range(len(percentage_difference)):
            for j in range(len(next_day_data)):
                arima_percentage_difference[i][j] += percentage_difference[i][j]/iterations
                arima_offset_list[i][j] += offset_list[i][j]/iterations
        start_date_prediction += timedelta(days=365)
        end_date_prediction += timedelta(days=365)
        start_date_training += timedelta(days=365)
        end_date_training += timedelta(days=365)
    working_day_counter = 1
    start_date_training = "2020-01-01"
    end_date_training = "2023-03-08"
    start_date_prediction = "2023-03-09"
    end_date_prediction = "2023-12-31"
    stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
    stock_data_2024 = fetch_stock_data(ticker_symbol, start_date_prediction, end_date_prediction)
    prediction_dates = pd.date_range(start=start_date_prediction, end=end_date_prediction, freq='D').strftime('%Y-%m-%d')
    prediction_dates_filtered = [date for date in prediction_dates if date in stock_data_2024.index]
    y1_train,y2_train,y3_train,y4_train,y5_train = preprocess_arima_data(stock_data_training)
    p, d, q = 0, 0, 2 
    history1 = list(y1_train)
    history2 = list(y2_train)
    history3 = list(y3_train)
    history4 = list(y4_train)
    history5 = list(y5_train)
    model1 = ARIMA(history1, order=(p, d, q))
    model1_fit=model1.fit()
    history2 = list(y2_train)
    model2 = ARIMA(history2, order=(p, d, q))
    model2_fit=model2.fit()
    history3 = list(y3_train)
    model3 = ARIMA(history3, order=(p, d, q))
    model3_fit=model3.fit()
    history4 = list(y4_train)
    model4 = ARIMA(history4, order=(p, d, q))
    model4_fit=model4.fit()
    history5 = list(y5_train)
    model5 = ARIMA(history5, order=(p, d, q))
    model5_fit=model5.fit()
    percentage_difference = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    arima_offset_list_2023 = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    for date in prediction_dates_filtered:
        next_day_data = predict_arima(model1_fit,model2_fit,model3_fit,model4_fit,model5_fit)
        next_day_data=apply_offset(offset_list,arima_offset_list,next_day_data,working_day_counter)
        actual_data_today = stock_data_2024.loc[date][['Open', 'High', 'Low', 'Close', 'Volume']]
        for k in range(len(next_day_data)):
            percentage_difference[working_day_counter][k] += math.fabs((next_day_data[k] - actual_data_today[k]) / actual_data_today[k]) * 100
        next_day_data,offset_list=check_offset(offset_list,working_day_counter,percentage_difference,next_day_data)
        current_day_data = pd.Series(next_day_data, index=current_day_data.index)
        working_day_counter += 1
    start_date_training = "2023-01-01"
    end_date_training = "2023-12-31"
    start_date_prediction = "2024-01-01"
    end_date_prediction = "2024-03-08"
    working_day_counter = 1
    stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
    stock_data_2024 = fetch_stock_data(ticker_symbol, start_date_prediction, end_date_prediction)
    prediction_dates = pd.date_range(start=start_date_prediction, end=end_date_prediction, freq='D').strftime('%Y-%m-%d')
    prediction_dates_filtered = [date for date in prediction_dates if date in stock_data_2024.index]
    y1_train,y2_train,y3_train,y4_train,y5_train = preprocess_arima_data(stock_data_training)
    p, d, q = 0, 0, 2 
    history1 = list(y1_train)
    history2 = list(y2_train)
    history3 = list(y3_train)
    history4 = list(y4_train)
    history5 = list(y5_train)
    model1 = ARIMA(history1, order=(p, d, q))
    model1_fit=model1.fit()
    history2 = list(y2_train)
    model2 = ARIMA(history2, order=(p, d, q))
    model2_fit=model2.fit()
    history3 = list(y3_train)
    model3 = ARIMA(history3, order=(p, d, q))
    model3_fit=model3.fit()
    history4 = list(y4_train)
    model4 = ARIMA(history4, order=(p, d, q))
    model4_fit=model4.fit()
    history5 = list(y5_train)
    model5 = ARIMA(history5, order=(p, d, q))
    model5_fit=model5.fit()
    percentage_difference = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    arima_offset_list_2024 = [[0] * 5 for _ in range(len(prediction_dates_filtered) + 1)]
    for date in prediction_dates_filtered:
        next_day_data = predict_arima(model1_fit,model2_fit,model3_fit,model4_fit,model5_fit)
        next_day_data=apply_offset(offset_list,arima_offset_list,next_day_data,working_day_counter)
        actual_data_today = stock_data_2024.loc[date][['Open', 'High', 'Low', 'Close', 'Volume']]
        for k in range(len(next_day_data)):
            percentage_difference[working_day_counter][k] += math.fabs((next_day_data[k] - actual_data_today[k]) / actual_data_today[k]) * 100
        next_day_data,offset_list=check_offset(offset_list,working_day_counter,percentage_difference,next_day_data)
        current_day_data = pd.Series(next_day_data, index=current_day_data.index)
        working_day_counter += 1
    arima_list=[0] * (len(prediction_dates_filtered)+1)
    for i in range(len(arima_offset_list_2024)):
        arima_list[i]=arima_offset_list_2024[i]+percentage_difference[i]
    start_date_training = "2023-01-01"
    end_date_training = "2024-03-08"
    stock_data_training = fetch_stock_data(ticker_symbol, start_date_training, end_date_training)
    y1_train,y2_train,y3_train,y4_train,y5_train = preprocess_arima_data(stock_data_training)
    p, d, q = 0, 0, 2 
    history1 = list(y1_train)
    history2 = list(y2_train)
    history3 = list(y3_train)
    history4 = list(y4_train)
    history5 = list(y5_train)
    arima_model1 = ARIMA(history1, order=(p, d, q))
    arima_model1_fit=arima_model1.fit()
    history2 = list(y2_train)
    arima_model2 = ARIMA(history2, order=(p, d, q))
    arima_model2_fit=arima_model2.fit()
    history3 = list(y3_train)
    arima_model3 = ARIMA(history3, order=(p, d, q))
    arima_model3_fit=arima_model3.fit()
    history4 = list(y4_train)
    arima_model4 = ARIMA(history4, order=(p, d, q))
    arima_model4_fit=arima_model4.fit()
    history5 = list(y5_train)
    arima_model5 = ARIMA(history5, order=(p, d, q))
    arima_model5_fit=arima_model5.fit()
    return arima_list,arima_model1_fit,arima_model2_fit,arima_model3_fit,arima_model4_fit,arima_model5_fit,arima_offset_list,arima_offset_list_2023,arima_offset_list_2024
ticker_symbol = "HDFCBANK.BO"
#arima_list,arima_model1,arima_model2,arima_model3,arima_model4,arima_model5,arima_offset_list,arima_offset_list_2023,arima_offset_list_2024=check_arima_model(10)
lin_reg_list,lin_reg_model,lin_reg_offset_list,lin_reg_offset_list_2023,lin_reg_offset_list_2024=check_lin_reg_model(10)
lin_reg_mod_list,lin_reg_mod_model1,lin_reg_mod_model2,lin_reg_mod_model3,lin_reg_mod_model4,lin_reg_mod_model5,lin_reg_mod_offset_list,lin_reg_mod_offset_list_2023,lin_reg_mod_offset_list_2024=check_lin_reg_mod_model(10)
ran_for_reg_list,ran_for_reg_model,ran_for_reg_offset_list,ran_for_reg_offset_list_2023,ran_for_reg_offset_list_2024=check_ran_for_reg_model(10)
gra_boo_list,gra_boo_model1,gra_boo_model2,gra_boo_model3,gra_boo_model4,gra_boo_model5,gra_boo_offset_list,gra_boo_offset_list_2023,gra_boo_offset_list_2024,stock_data_training=check_gra_boo_model(10)
current_day_data = stock_data_training.iloc[-1][['Open', 'High', 'Low', 'Close', 'Volume']]
prediction=[]
for i in range(len(lin_reg_list)):
    next_day_data=[0]*5
    lin_reg_data=[0]*5
    lin_reg_mod_data=[0]*5
    ran_for_reg_data=[0]*5
    gra_boo_data=[0]*5
    arima_data=[0]*5
    for j in range(len(lin_reg_list[i])):
        list=[math.fabs(lin_reg_list[i][j]),math.fabs(gra_boo_list[i][j]),math.fabs(lin_reg_mod_list[i][j]),math.fabs(ran_for_reg_list[i][j])]
        if list.index(min(list))==0:
            if all(ele > 0 for ele in lin_reg_data)==False:
                data=predict_lin_reg(lin_reg_model,current_day_data)
                lin_reg_data=apply_final_offset(lin_reg_offset_list_2023,lin_reg_offset_list_2024,lin_reg_offset_list,data,i)
            data=lin_reg_data
        elif list.index(min(list))==1:
            if all(ele > 0 for ele in gra_boo_data)==False:
                data=predict_gb(gra_boo_model1,gra_boo_model2,gra_boo_model3,gra_boo_model4,gra_boo_model5,current_day_data)
                gra_boo_data=apply_final_offset(gra_boo_offset_list_2023,gra_boo_offset_list_2024,gra_boo_offset_list,data,i)
            data=gra_boo_data
        elif list.index(min(list))==2:
            if all(ele > 0 for ele in lin_reg_mod_data)==False:
                data=predict_lin_reg_mod(lin_reg_mod_model1,lin_reg_mod_model2,lin_reg_mod_model3,lin_reg_mod_model4,lin_reg_mod_model5,current_day_data)
                lin_reg_mod_data=apply_final_offset(lin_reg_mod_offset_list_2023,lin_reg_mod_offset_list_2024,lin_reg_mod_offset_list,data,i)
            data=lin_reg_mod_data
        else:
            if all(ele > 0 for ele in ran_for_reg_data)==False:
                data=predict_ran_for_reg(ran_for_reg_model,current_day_data)
                ran_for_reg_data=apply_final_offset(ran_for_reg_offset_list_2023,ran_for_reg_offset_list_2024,ran_for_reg_offset_list,data,i)
            data=ran_for_reg_data
        next_day_data[j]=data[j]
    prediction.append(next_day_data)
    current_day_data = pd.Series(next_day_data, index=current_day_data.index)
print(prediction)
end=time.time()
print(end-start)