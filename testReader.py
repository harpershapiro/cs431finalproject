#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:26:30 2019

@author: katrinahoefflinger
"""


import pandas as pd # Import the library and give a short alias: pd
# Read the house data, print the first five lines
print("Reading house data...")
house = pd.read_csv("data/kc_house_data.csv")
print(house.head(5))
# Get average house price.
prices = house['price']
avg_price = prices.mean()
print(f"Average price is ${avg_price:.0f}")

# Split the data
from sklearn.model_selection import train_test_split
X, y = house[['bedrooms','bathrooms','zipcode', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'sqft_living15', 'sqft_lot15']], house['price']
 # 20% of data goes into test set, 80% into training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

 #rf = RandomForestRegressor(n_estimators=10)
 #rf.fit(X_train, y_train)
 #validation_e = mean_absolute_error(y_test, rf.predict(X_test))
 # print(f"${validation_e:.0f} average error; {validation_e*100.0/y.mean():.2f}% error")
