#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 13:36:00 2021

@author: suraj
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

Data = load_boston()
Data_Set = pd.DataFrame(Data.data,columns=Data.feature_names) #DataFrame

x = Data.data[:,5]
y = Data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,train_size = 0.75,random_state = 76)



Poly_P = PolynomialFeatures(degree=2) 
x_train = x_train.reshape(-1,1)
x_train = Poly_P.fit_transform(x_train)


model = LinearRegression()
model.fit(x_train,y_train)

x_test = x_test.reshape(-1,1)
x_test = Poly_P.fit_transform(x_test)

y_predicted = model.predict(x_test)


Accuracy_Score = r2_score(y_test,y_predicted)
Accuracy = Accuracy_Score*100