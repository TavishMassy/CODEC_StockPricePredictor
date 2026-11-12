# Apple (AAPL) Stock Price Prediction Using Machine Learning

## Project Overview
This project implements an end-to-end financial machine learning pipeline to forecast the next-day closing price of Apple Inc. (`AAPL`) stock. Using 7 years of historical daily market data fetched via `yfinance`, the notebook performs technical indicator engineering, mutual information feature selection, chronological train-test splitting, and Linear Regression modeling.

---

## Features & Technical Indicators
Starting from raw OHLCV (`Open`, `High`, `Low`, `Close`, `Volume`) daily data, the following domain-specific financial features were engineered:

* **Monthly VWAP (Volume Weighted Average Price):** Institutional price benchmark anchored to the 1st of each calendar month.
* **RSI (Relative Strength Index - 14-Period):** Momentum oscillator measuring overbought ($\ge 70$) and oversold ($\le 30$) market conditions.
* **ATR (Average True Range - 14-Period):** Volatility indicator computed from true price ranges across daily gaps and highs/lows.
* **Simple Moving Averages (SMAs):** 21-day, 42-day, and 63-day trend indicators.
* **Bollinger Bands (21-Day, $\pm 2\sigma$):** Upper and lower volatility bands around the 21-day SMA.
* **Lagged Closing Prices (`Close_Lag_1` to `Close_Lag_14`):** 14-day historical price context for autoregressive time-series forecasting.
* **Daily Return:** Percentage price change from the previous trading day.

---

## Methodology
1. **Data Collection:** Retrieved 7 years of historical daily stock prices for `AAPL` (1,759 trading days) using the `yfinance` API.
2. **Feature Selection:** Evaluated feature relevance against the next-day closing price (`Target_close`) using Mutual Information Regression (`mutual_info_regression`), retaining 28 out of 29 features above a `0.05` threshold.
3. **Chronological Validation Split:** Applied a strict time-series split without data leakage:
   * **Training Set:** 1,417 historical trading days.
   * **Validation Set:** Latest 252 trading days (1 full trading year).
4. **Model Training & Evaluation:** Trained a `LinearRegression` model (`scikit-learn`) and evaluated out-of-sample predictions against actual closing prices.

---

## Model Performance
On the out-of-sample 1-year validation set (252 trading days):
* **Mean Absolute Error (MAE):** `$7.41`
* **Average Actual Price:** `$257.62`
* **$R^2$ Score:** `0.8865`

---

## Visualizations Included
* **Institutional & Momentum Dashboard:** 3-panel chart showing Closing Price vs. Monthly VWAP, 14-period RSI with overbought/oversold zones, and 14-period ATR volatility expansion.
* **Trend & Volatility Envelope:** 2-panel chart displaying Closing Price with 21-day SMA and Bollinger Bands alongside daily trading volume.
* **Mutual Information Ranking:** Horizontal bar chart ranking all selected features by predictive information score.
* **Actual vs. Predicted Price Plot:** 1-year out-of-sample comparison between actual `AAPL` closing prices and Linear Regression forecasts.

---

## Tech Stack & Dependencies
* **Python 3.x**
* `yfinance` (Market data retrieval)
* `pandas` & `numpy` (Data manipulation & indicator calculation)
* `matplotlib` & `seaborn` (Financial data visualization)
* `scikit-learn` (Mutual Information feature selection, Linear Regression, and evaluation metrics)

---

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/TavishMassy/CODEC_StockPricePredictor
   cd CODEC_StockPricePredictor
   ```
2. Install the required libraries:
   ```bash
   pip install yfinance numpy pandas matplotlib seaborn scikit-learn
   ```
3. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
