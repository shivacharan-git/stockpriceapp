import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential                   # Sequential model
from tensorflow.keras.layers import Dense             # For fully connected layers
from tensorflow.keras.layers import LSTM              # For LSTM layers
from sklearn.preprocessing import MinMaxScaler        # Scaling the data
from keras.models import load_model
min_max_scaler = MinMaxScaler()

##streamlit run app.py

st.write("""
# Simple Stock Price Prediction App
""")

#define the ticker symbol
st.sidebar.header('Stock Selection')
nse = ['BAJAJ-AUTO.NS', 'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS',
 'GAIL.NS', 'GRASIM.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'ITC.NS',
 'KOTAKBANK.NS', 'LT.NS', 'MARUTI.NS', 'MM.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'RELIANCE.NS', 'SHREECEM.NS',
 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS']


stock = st.sidebar.selectbox('Select Stock',nse,index=0)

st.write("""
Shown are the stock **LTP**, **Closing price** ,  & **Predicted Price** of""", stock)

tickerData = yf.Ticker(stock)
tickerDf = tickerData.history(period='max')

stocks_df = pd.DataFrame(tickerDf.Close)
stocks_df.dropna(axis=0,how='any',inplace=True)

st.write("""
## LTP
""")
st.write(stocks_df['Close'][-1])

st.write("""
## Closing Price
""")
st.dataframe(stocks_df)

prediction_days = 30
ts_train= stocks_df[:-prediction_days]       # Remove 30 days from the end for Training data
ts_test= stocks_df[-prediction_days:]

# We are going to make prediction
# Preprocess the test data
test_set = ts_test.values

inputs = np.reshape(test_set, (len(test_set), 1))               # Reshape before passing in the input
inputs = min_max_scaler.fit_transform(inputs)#transform()                       # Scaling the data
inputs = np.reshape(inputs, (len(inputs), 1, 1))

model = load_model('StockPrice.h5')
result = model.predict(inputs)

predicted_price = model.predict(inputs)                              # Make predictions on the test data
predicted_price = min_max_scaler.inverse_transform(predicted_price)  # Inverse transform the predicted price
st.write("""
## Predicted Price
""")
predicted=pd.DataFrame(predicted_price,columns=["Predicted price"],index=ts_test.index)       # Calculate the error
st.dataframe(predicted)
st.write("""
\n\n\n
""")
# Plot the Actual price and the predicted price
plt.figure(figsize=(12, 12), dpi=80, facecolor = 'w', edgecolor = 'k')

plt.plot(test_set[:, 0], color='red', label='Real {} Price'.format(stock))                      # Actual Price
plt.plot(predicted_price[:, 0], color = 'blue', label = 'Predicted Close Price')      # Predicted Price

plt.title('{} Price Prediction from {} to {}'.format(stock,stocks_df.index[-prediction_days].date(),stocks_df.index[-1].date()), fontsize = 20)
plt.xlabel('Time', fontsize=30)
plt.ylabel('{} Price'.format(stock), fontsize = 30)
plt.legend(loc = 'best')
# plt.show()
st.pyplot(plt)
