# Stock Price Prediction Using Machine Learning

This project predicts future stock prices dynamically using multiple machine learning models. The backend is built with **Bottle** (Python), and the frontend is created using **HTML**, **CSS**, and **JavaScript**.

## Tech Stack
- **Backend**: Python with **Bottle** framework
- **Machine Learning Models**: Linear Regression, Modified Linear Regression, Random Forest Regression, Gradient Boosting
- **Frontend**: HTML, CSS, JavaScript
- **Libraries**: pandas, math, sklearn

## Features
- **Dynamic Stock Price Prediction**: The algorithm predicts the next day's stock prices based on historical data, using multiple machine learning models such as Linear Regression, Modified Linear Regression, Random Forest Regression, and Gradient Boosting.
  
- **Model Selection**: The system tries different machine learning models for the given stock and selects the model that provides the most accurate predictions. Each model is evaluated based on its prediction accuracy, and the model with the least error is chosen to forecast the stock's future price. The algorithm uses the historical data for various features such as Open, High, Low, Close, and Volume to make these predictions.

- **Bias Adjustment**: After selecting the best-performing model, the algorithm further refines its predictions by adding a bias or offset. This helps improve the accuracy of the forecast by accounting for trends or recurring patterns that may not be captured perfectly by the machine learning model alone. The offset values are dynamically calculated and applied to make the predictions more precise.

- **Interactive Frontend**: The frontend allows users to input the stock ticker symbol (e.g., `HDFCBANK.BO`) and retrieve predictions for the next day's stock prices. The predictions are displayed for several price points, including Open, High, Low, Close, and Volume.

- **Prediction Results**: For each stock, the system predicts various key metrics, including the opening price, the highest price of the day, the lowest price, the closing price, and the trading volume. The results are presented in a clear, user-friendly interface, allowing users to easily understand the predicted market behavior.
