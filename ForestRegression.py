#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 20:19:16 2021

@author: suraj
"""

#Random_Forest

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

from sklearn.ensemble import RandomForestRegressor

Random_F = RandomForestRegressor(n_estimators=100,max_depth = 30 , random_state = 33)
Random_F.fit(x_train,y_train)

y_predicted = Random_F.predict(x_test)
y_predicted = y_predicted.reshape(-1,1)
y_predicted = scaler.inverse_transform(y_predicted)

from sklearn.metrics import r2_score

accuracy = r2_score(y_test,y_predicted)




from sklearn.svm import SVR

model_SVR = SVR(kernel = 'rbf')

model_SVR.fit(x_train,y_train)
y_predicted = model_SVR.predict(x_test)
y_predicted = y_predicted.reshape(-1,1)
y_predicted = scaler.inverse_transform(y_predicted)

from sklearn.metrics import r2_score
accuracySVR = r2_score(y_test,y_predicted)
