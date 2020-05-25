import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import keras
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import sklearn


import pymysql
from sqlalchemy import create_engine

connection_string = "THIS STRING HELD THE CONNECTION TO THE SERVER FOR SQL QUERIES"
connec = create_engine(connection_string)

df = pd.read_sql('SELECT * FROM retail_uk_ecom LIMIT 100000', con = connec)
df.info()


##General Data Selection/Cleaning: Only want date of sale and aggregate amount sold.
df['sales_date'] = pd.to_datetime(df['sales_date'])
df['items_quantity'] = pd.to_numeric(df['items_quantity'])
SALES_EVERYDAY =  pd.DataFrame(df.groupby('sales_date')['items_quantity'].sum())
SALES_EVERYDAY = SALES_EVERYDAY.reset_index()

SALES_EVERYDAY_DEL = SALES_EVERYDAY.copy()
SALES_EVERYDAY_DEL['prev'] = SALES_EVERYDAY_DEL['items_quantity'].shift(1)
SALES_EVERYDAY_DEL = SALES_EVERYDAY_DEL.dropna()
SALES_EVERYDAY_DEL['del'] = (SALES_EVERYDAY_DEL['items_quantity'] - SALES_EVERYDAY_DEL['prev'])

SUPERVISED = SALES_EVERYDAY_DEL.drop(['prev'], axis = 1)

for x in range(1,30):
    extra_col = 'lbp-' + str(x)
    SUPERVISED[extra_col] = SUPERVISED['del'].shift(x)

SUPERVISED = SUPERVISED.dropna().reset_index(drop = True)

##Setting Test and Training Samples. Lookback period was chosen to be past 30 days.

from sklearn.preprocessing import MinMaxScaler
modeldf = SUPERVISED.drop(['items_quantity','sales_date'], axis = 1)

train_samples, test_samples = modeldf[0:-30].values, modeldf[-30:].values

SCALER = MinMaxScaler(feature_range = (-1,1))
SCALER = SCALER.fit(train_samples)

train_samples = train_samples.reshape(train_samples.shape[0], train_samples.shape[1])
training_scaled = SCALER.transform(train_samples)
test_samples = test_samples.reshape(test_samples.shape[0], test_samples.shape[1])
test_scaled = SCALER.transform(test_samples)

x_train, y_train = training_scaled[:,1:], training_scaled[:,0:1]
x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1])
x_test , y_test = test_scaled[:,1:], test_scaled[:,0:1]
x_test = x_test.reshape(x_test.shape[0], 1,x_test.shape[1])

##Creation of Sequential Model LSTM Neural Network with one dense layer

model = Sequential()
model.add(LSTM(4,batch_input_shape= (1,x_train.shape[1],x_train.shape[2]), stateful = True))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.fit(x_train, y_train, batch_size = 1, shuffle= False, epochs = 100)

preds = model.predict(x_test,batch_size = 1)

##Reconstructing predictuions into interpretable sales data by inverse transform.
preds = preds.reshape(30,1,1)
rebuilt_test =[]
for x in range(0,30):
    rebuilt_test.append(np.concatenate([preds[x],x_test[x]], axis = 1))
rebuilt_test = np.array(rebuilt_test)
rebuilt_test = rebuilt_test.reshape(rebuilt_test.shape[0], rebuilt_test.shape[2])
rebuilt_test_inv = SCALER.inverse_transform(rebuilt_test)

results = []
true_dates = list(SALES_EVERYDAY[-31:].sales_date)
true_sales = list(SALES_EVERYDAY[-31:].items_quantity)
y = 0
while y < len(rebuilt_test_inv):
    temp_dict = {}
    temp_dict['pvalue'] = int(rebuilt_test_inv[y][0] + act_sales[y])
    temp_dict['sales_date'] = sales_dates[y+1]
    results.append(temp_dict)
    y+=1   
resultdf = pd.DataFrame(results)

##Plotting the Data


import plotly.offline as po 
import plotly.graph_objs as graph

prediction_sales = pd.merge(SALES_EVERYDAY, resultdf, on= 'sales_date', how = 'left')

plot_data = [go.Scatter(x = prediction_sales['sales_date'], y = prediction_sales['items_quantity'], name = 'True Sales'), go.Scatter(x = prediction_sales['sales_date'], y = prediction_sales['pvalue'],name = 'Predicted')]

plot_layout = go.Layout(title = 'Sales Prediction')
fig = go.Figure(data = plot_data, layout = plot_layout)
po.iplot(fig)




###SKU Correlation Heatmap

import pymysql
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np

connection_string = 'mysql+pymysql://datathon_team1:mR}Aw7W@34.227.65.68/datathon'
connec = create_engine(connection_string)
df = pd.read_sql('SELECT * FROM retail_uk_ecom LIMIT 1000', con = connec)
df.info()

dummy = pd.get_dummies(df['SKU'])
dummy['transaction_number'] = df['transaction_number']
CORRELATION_MATRIX_BASKET = dummy.groupby('transaction_number').sum().corr()
sns.heatmap(CORRELATION_MATRIX_BASKET,, vmin = 0.4, vmax = 0.99, cmap = 'Blues')
