Stock Price Prediction Using Machine Learning
This project predicts future stock prices dynamically using multiple machine learning models. The backend is built with Bottle (Python), and the frontend is created using HTML, CSS, and JavaScript.

Tech Stack
Backend: Python with Bottle framework
Machine Learning Models: Linear Regression, Modified Linear Regression, Random Forest Regression, Gradient Boosting
Frontend: HTML, CSS, JavaScript
Libraries: pandas, math, sklearn
Features
Predicts next-day stock prices using multiple models.
Dynamic adjustments with offsets.
Interactive frontend for stock ticker input.
Displays predicted stock data (Open, High, Low, Close, Volume).
Project Structure
bash
Copy code
/stock-prediction
│
├── /static
│   ├── /css
│   ├── /js
│
├── /templates
│   └── index.html
│
├── /models
│   └── stock_prediction.py
│
└── server.py
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Start the server:

bash
Copy code
python server.py
Open http://localhost:8080 in your browser.

Usage
Enter a stock ticker symbol (e.g., HDFCBANK.BO).
Click "Predict" to get next-day stock price predictions.
