# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 02:52:51 2021

@author: yajur
"""

import pandas as pd 
import numpy as np 
import pickle 

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


da = pd.read_csv("book.csv")

da.dtypes 

x = da.iloc[:, :5].values
y = da.iloc[:, 5].values


xtrain, xtest, ytrain, ytest=train_test_split(x, y, random_state=12, 
             test_size=0.15)



gbr = GradientBoostingRegressor(n_estimators=600, 
    max_depth=5, 
    learning_rate=0.01, 
    min_samples_split=3)

gbr.fit(xtrain, ytrain)

ypred = gbr.predict(xtest)
mse = mean_squared_error(ytest,ypred)

mse 


x_ax = range(len(ytest))
plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")

pickle.dump(gbr,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

