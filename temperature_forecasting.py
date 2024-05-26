import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load data
df = pd.read_csv('MaunaLoaDailyTemps.csv', index_col='DATE', parse_dates=True)
df = df.dropna()
print('Shape of data', df.shape)

# Plotting data
df['AvgTemp'].plot(figsize=(12,5), title='Average Temperature Over Time')
plt.xlabel('Date')
plt.ylabel('AvgTemp')
plt.show()

# Check for Stationarity
def adf_test(dataset):
    dftest = adfuller(dataset, autolag='AIC')
    print("1. ADF : ", dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Num Of Lags : ", dftest[2])
    print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
    print("5. Critical Values :")
    for key, val in dftest[4].items():
        print("\t", key, ": ", val)

adf_test(df['AvgTemp'])

# Determining order for ARIMA model
stepwise_fit = auto_arima(df['AvgTemp'], trace=True, suppress_warnings=True)
stepwise_fit.summary()

# Splitting data into training and testing
train = df.iloc[:-30]
test = df.iloc[-30:]
print(train.shape, test.shape)
print(test.iloc[0], test.iloc[-1])

# Training the model
model = ARIMA(train['AvgTemp'], order=(1, 0, 5))
model = model.fit()
print(model.summary())

# Make predictions on test set
start = len(train)
end = len(train) + len(test) - 1
index_future_dates = pd.date_range(start='2018-12-01', end='2018-12-30')
pred = model.predict(start=start, end=end, typ='levels').rename('ARIMA Predictions')
pred.index = index_future_dates
pred.plot(legend=True)
test['AvgTemp'].plot(legend=True)
plt.show()

# Calculate RMSE
rmse = sqrt(mean_squared_error(pred, test['AvgTemp']))
print('RMSE:', rmse)

# Train the model on entire dataset
model2 = ARIMA(df['AvgTemp'], order=(1, 0, 5))
model2 = model2.fit()
print(df.tail())

# Prediction of future dates
index_future_dates = pd.date_range(start='2018-12-30', end='2019-01-29')
pred = model2.predict(start=len(df), end=len(df) + 30, typ='levels').rename('ARIMA Predictions')
pred.index = index_future_dates
print(pred)

pred.plot(figsize=(12,5), legend=True, title='Future Temperature Predictions')
plt.xlabel('Date')
plt.ylabel('AvgTemp')
plt.show()
