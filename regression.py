from sklearn.datasets import load_boston
import pandas as pd
import numpy as np


Data = load_boston()
Data_Set = pd.DataFrame(Data.data,columns = Data.feature_names)
Data_Set['Target'] = Data.target

x = Data.data
y = Data.target

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,train_size= 0.75,random_state = 76)

from sklearn.preprocessing import MinMaxScaler
    
scaler = MinMaxScaler(feature_range=(0,1))

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

y_train = y_train.reshape(-1,1)
y_train = scaler.fit_transform(y_train)

#Regression

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train,y_train)
y_predict = model.predict(x_test)

y_predict = scaler.inverse_transform(y_predict)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math

AE1 = mean_absolute_error(y_test, y_predict)
SE2 = mean_squared_error(y_test,y_predict)
RSE = math.sqrt(SE2)
R2 = r2_score(y_test,y_predict)

def mean_absulute_percentage_error(y_true,y_pred):
    y_true,y_pred = np.array(y_true),np.array(y_pred)
    return np.mean(np.abs((y_true-y_pred)/y_true))*100

MAPE = mean_absulute_percentage_error(y_test,y_predict)
