#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:26:30 2019

@author: katrinahoefflinger
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from rfpimp import *
import matplotlib.pyplot as plt


import pandas as pd # Import the library and give a short alias: pd
# Read the house data, print the first five lines
print("Reading house data...")
house = pd.read_csv("data/kc_house_data.csv")
print(house.head(5))
# Get average house price.
prices = house['price']
avg_price = prices.mean()
print(f"Average price is ${avg_price:.0f}")

X, y = house[['bedrooms','bathrooms', 'zipcode', 'lat', 'long', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']], house['price']
 # 20% of data goes into test set, 80% into training set 
 
byzipcode = house.groupby(['zipcode']).mean()
byzipcode = byzipcode.reset_index()
bybedrooms = house.groupby(['bedrooms']).mean()
bybedrooms = bybedrooms.reset_index()
bybath = house.groupby(['bathrooms']).mean()
bybath = bybath.reset_index()

bybath.plot.line('bathrooms','price', style='-o')
plt.show()
bybedrooms.plot.line('bedrooms','price', style='-o')
plt.show()
byzipcode.plot.line('zipcode','price', style='-o')
plt.show()

 
numRuns = 4
sum = 0
for i in range(numRuns):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
    rf = RandomForestRegressor(n_estimators=100) 
    rf.fit(X_train, y_train)
    validation_e = mean_absolute_error(y_valid, rf.predict(X_valid))
    sum += validation_e

#I = importances(rf, X_valid, y_valid)
#plot_importances(I, color='#4575b4', vscale=1.8)
I = importances(rf, X_valid, y_valid, features=['bedrooms','bathrooms',['lat','long'], 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15'])
plot_importances(I, color='#4575b4', vscale=1.8)
sum = sum/numRuns
print(f"${sum:.0f} average error; {sum*100.0/y.mean():.2f}% error")
