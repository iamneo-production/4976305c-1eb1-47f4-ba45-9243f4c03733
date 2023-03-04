import requests

# data
urls = {
    2022: "https://tspcb.cgg.gov.in/CentralLab/MonthlyAQI2022.xls",
    2021: "https://tspcb.cgg.gov.in/CentralLab/MonthlyAQI2021.xls",
    2020: "https://tspcb.cgg.gov.in/CentralLab/MonthlyAQI2020.xls",
    2019: "https://tspcb.cgg.gov.in/Environment/MonthlyAQI2019.xlsx",
    2018: "https://tspcb.cgg.gov.in/Environment/Monthly%20AQI%20Jan%20-%20Dec%202018.xlsx",
}

# download to file
def get_workbook(url: str, filepath: str):
  with open(filepath ,'wb') as file:
    file.write(requests.get(url).content)

!pip install xlrd

import xlrd


def read_for_year(y):
  # get workbool
  get_workbook(urls[y], str(y) + ".xls")

  # read
  book = xlrd.open_workbook(str(y) + ".xls")

  print("The number of worksheets is {0}".format(book.nsheets))
  print("Worksheet name(s): {0}".format(book.sheet_names()))

  # Collect sheets
  sheets = []
  for i in range(0, book.nsheets):
    sheets.append(
        book.sheet_by_index(i)
    )
  print(f"collected: {len(sheets)} sheets")
  return sheets
  
def get_row(sheet, rows):
  r = []
  for i in range(rows[0], rows[1]):
    r.append(sheet.row_values(i))
  return r

sheet_2022 = read_for_year(2022)[1]
rows_2022 = get_row( sheet_2022, (53, 60))

len(rows_2022)
sheet_2021 = read_for_year(2021)[1]
rows_2021 = get_row( sheet_2021, (6, 13))

# check 2 rows
rows_2022[:2]
rows_2021[:2]

# removing preceding 2 columns
rows_2022 = [ r[2:] for r in rows_2022 ]
rows_2021 = [ r[2:] for r in rows_2021 ]

import pandas as pd

# new dataframe
df = pd.DataFrame()

from datetime import datetime

# add a column for time
row_mon = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
row_time = []

for y in [2021, 2022]:
  for row in row_mon:
    row = row + "_" + str(y)
    dt = datetime.strptime(row, '%b_%Y').date()
    row_time.append(
        pd.to_datetime(dt)
    )

# concat
df = pd.concat([
    df,
    pd.DataFrame(row_time, columns=['month'])
])

# print
df

import numpy as np
def add_aqi(rows):
  # create a zero matrix with 1 column
  avg = np.zeros(shape=(len(rows[0]), 1))
  # add values to the matrix
  for i in rows:
    for j in range(0, len(i)):
      if i[j] == '-':
        i[j] = 0
      avg[j] = avg[j] + i[j]
  # calulate average
  avg = np.divide(avg, len(rows))
  # add to dataframe
  return avg

# vertically stack the rows
avg_values = np.vstack((add_aqi(rows_2022), add_aqi(rows_2021)))
print(avg_values)

df['aqi'] = avg_values

# print the pandas data
print(df)
# save to csv
df.to_csv('data.csv', index=False)

import matplotlib.pyplot as plt 
import numpy as np 

# change size
plt.figure(figsize=(15,5))
# title
plt.title("Air Quality Index in Respect to Historical Time")

# plot
plt.plot(df.month, df.aqi)

# label
plt.xlabel("Month")
plt.ylabel("AQI")
plt.draw()
plt.show()

# copy
df = pd.read_csv('data.csv')

# convert month to timestamp
df['month'] = pd.to_datetime(df['month']).astype(int) // 10**9

# Train Test Dividing Ratio

train_size = int(len(df) * 0.80)
#test_size = len(df) - train_size # not requied

# Training Data
x_train = df['month'][:train_size]
y_train = df['aqi'][:train_size]
x_test = df['month'][train_size:]
y_test = df['aqi'][train_size:]

training_data = df['aqi'].values

from sklearn.preprocessing import MinMaxScaler

# RESHAPING PROBLEM

array.reshape(-1, 1)
training_data = MinMaxScaler().fit_transform(training_data)

import pandas as pd
import tensorflow as tf


tf.random.set_seed(7)
# load the dataset
dataframe = pd.read_csv('data.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 3

# reshape input to be [samples, time steps, features]
train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_x, train_x, epochs=100, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(train_x)
testPredict = model.predict(test_x)