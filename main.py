import streamlit as st
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import os
import datetime
from prophet import Prophet
import requests
import pandas as pd


def get_stock_data_by_symbol(symbol: str):
    url = "https://alpha.financeapi.net/symbol/get-chart?period=MAX&symbol=" + symbol

    payload = {}
    headers = {
        'X-API-KEY': 'FXl3Ntu3hA4UehOXR018C5Lbinf5ADsS9s6cQVGt'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    response_json = response.json()
    if 'attributes' in response_json:
        return pd.DataFrame(response_json['attributes']).T
    else:
        return pd.DataFrame()


def fill_missing_values(dataframe):
    return dataframe.ffill().bfill()


def get_data(symbols, dates):
    df_final = pd.DataFrame(index=dates)
    for symbol in symbols:
        df_temp = get_stock_data_by_symbol(symbol)
        if not df_temp.empty:
            df_temp = df_temp[['close']]
            df_temp.index = pd.to_datetime(df_temp.index)
            df_temp = df_temp.rename(columns={"close": symbol})
            df_final = df_final.join(df_temp)

    return df_final


def get_data_for_training(symbol, start_date, end_date):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df_final = pd.DataFrame(index=pd.date_range(start_date, end_date))
    #  file_path = symbol_to_path(symbol)
    df_temp = get_stock_data_by_symbol(symbol)
    if df_temp.empty:
        raise 'Symbol invalid or with empty stock data!'
    df_temp = df_temp[['close']]
    df_temp.index = pd.to_datetime(df_temp.index)
    df_final = df_final.join(df_temp)
    df_final = df_final.ffill().bfill()
    return df_final


def compute_daily_returns(df):
    """Compute and return the daily return values."""
    # TODO: Your code here
    # Note: Returned DataFrame must have the same number of rows
    return df[:-1] / df[1:].values - 1


def get_data_for_training(symbol, start_date, end_date):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df_final = pd.DataFrame(index=pd.date_range(start_date, end_date))
    df_temp = get_stock_data_by_symbol(symbol)
    if df_temp.empty:
        raise 'Symbol invalid or with empty stock data!'
    df_temp = df_temp[['close']]
    df_temp.index = pd.to_datetime(df_temp.index)
    df_final = df_final.join(df_temp)
    df_final = df_final.ffill().bfill()
    return df_final


st.title("Capstone Project Data Scientist: Investment and Trading")
st.info("Student: Paulo Mota")
tab_compare, tab_predict = st.tabs(["Compare Stocks", "Predictor"])

with tab_compare:
    st.write("""Closing price and daily returns comparator.""")
    today = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)
    yesterday = today - datetime.timedelta(days=1)
    start_default = yesterday - datetime.timedelta(days=30)
    start_date = st.date_input('Start date', datetime.date(1993, 1, 1), min_value=datetime.date(1993, 1, 1))
    end_date = st.date_input('End date', yesterday)
    if not start_date < end_date:
        st.error('Error: End date must fall after start date.')

    if end_date > yesterday:
        st.error('Error: End date must be equal or older than yesterday.')

    # get symbols from user
    symbol_list = st.text_input("Enter Symbols Separated by comma (,)", value='GOOG,TSLA,MSFT',
                                key='symbols').upper().split(',')

    if st.button('Run'):
        valid_input = True
        if not start_date < end_date:
            st.error('Error: End date must fall after start date.')
            valid_input = False

        if end_date > yesterday:
            st.error('Error: End date must be equal or older than yesterday.')
            valid_input = False

        if symbol_list[0] == '':
            st.error('Error: You must have at least one stock symbol.')
            valid_input = False

        with st.spinner("Processing..."):
            # check if all fields are filled
            if valid_input:
                dates = pd.date_range(start_date, end_date)  # date range as index
                df_data = get_data(symbol_list, dates)  # get data for each symbol

                # Fill missing values
                df_data = fill_missing_values(df_data)

                st.header('Closing price by stock symbol')
                st.line_chart(df_data)
                with st.expander("What is closing price?"):
                    st.write(
                        """The closing price is the raw price or cash value of the last transacted price in a security before the market officially closes for normal trading. It is often the reference point used by investors to compare a stock's performance since the previous day""")

                daily_returns = compute_daily_returns(df_data)
                st.header('Daily returns by stock symbol')
                st.line_chart(daily_returns)
                with st.expander("What is daily return?"):
                    st.write(
                        """Daily return is calculated by subtracting the opening price from the closing price. If you are calculating for a per-share gain, you simply multiply the result by your share amount. If you are calculating for percentages, you divide by the opening price, then multiply by 100.""")
            else:
                st.error("Please fill all fields!")

with tab_predict:
    st.write(
        """Closing price predictor with Prophet Model from Facebook, more info https://facebook.github.io/prophet/ """)
    stock_symbol = st.text_input('Stock Symbol', 'TSLA')

    if st.button('Predict Stock'):
        with st.spinner("Processing..."):
            today = datetime.date.today()
            start_date = today - datetime.timedelta(days=1095)
            end_date = today - datetime.timedelta(days=1)
            predict_days = 365
            df_train = get_data_for_training(stock_symbol, start_date, end_date)
            # Preparing training data
            df_train = df_train[['close']]
            df_train = df_train.reset_index()
            df_train = df_train.rename(columns={'index': 'ds', 'close': 'y'})
            df_train.tail()

            # Create Prophet model
            model = Prophet()

            # Train model
            model.fit(df_train)

            # Make predictions on test data
            future = model.make_future_dataframe(periods=predict_days, freq='D')
            predictions = model.predict(future)

            st.line_chart(predictions[['ds', 'yhat']], x='ds', y='yhat')
