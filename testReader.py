#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:26:30 2019

@author: katrinahoefflinger
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import learning_curve
from rfpimp import *
import matplotlib.pyplot as plt

import pandas as pd # Import the library and give a short alias: pd
# Read the house data, print the first five lines
#DEFINE PLOT LEARNING CURVE
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

##########################################################################


print("Reading house data...")
house = pd.read_csv("data/kc_house_data.csv")
print(house.head(5))
# Get average house price.
prices = house['price']
avg_price = prices.mean()
print(f"Average price is ${avg_price:.0f}")
feature_list = ['bedrooms','bathrooms','lat','long', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']
feature_list_coordGroup = ['bedrooms','bathrooms',['lat','long'], 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']

X, y = house[feature_list], house['price']
 # 20% of data goes into test set, 80% into training set 
 
bysqft_living = house.groupby(['sqft_living']).mean()
bysqft_living = bysqft_living.reset_index()
bybedrooms = house.groupby(['bedrooms']).mean()
bybedrooms = bybedrooms.reset_index()
bybath = house.groupby(['bathrooms']).mean()
bybath = bybath.reset_index()

bybath.plot.line('bathrooms','price', style='-o')
plt.show()
bybedrooms.plot.line('bedrooms','price', style='-o')
plt.show()
bysqft_living.plot.line('sqft_living','price', style='-o')
plt.show()

#TRAINING THE CLASSIFIER
#numRuns = 4
#sum = 0
#for i in range(numRuns):
print("Training the classifier...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
rf = RandomForestRegressor(n_estimators=100) 
rf.fit(X_train, y_train)
print("Validation Evaluation...")
validation_e = mean_absolute_error(y_valid, rf.predict(X_valid))
#sum += validation_e

#FINAL SCORES AFTER TRAINING
    
#UNGROUPED LAT LONG
#I = importances(rf, X_valid, y_valid) 
#plot_importances(I, color='#4575b4', vscale=1.8)

#GROUPED LAT LONG
#I = importances(rf, X_valid, y_valid, features=feature_list_coordGroup)
#plot_importances(I, color='#FF0000', vscale=1.8)

#LEARNING CURVE FOR TRAINING
plot_learning_curve(
       rf,"LEARNING CURVE: RANDOM FOREST",X_train,y_train)
#sum = sum/numRuns
#print(f"${sum:.0f} average error; {sum*100.0/y.mean():.2f}% error")
print(f"${validation_e:.0f} average error; {validation_e*100.0/y.mean():.2f}% error")


#TESTING
print("Test results for the classifier...")
test_I = importances(rf, X_test, y_test, features=feature_list_coordGroup)
plot_importances(test_I, color='#4575b4',vscale=1.8)
test_mae = mean_absolute_error(y_test,rf.predict(X_test))
test_mse = mean_squared_error(y_test, rf.predict(X_test))
print(f"${test_mae:.0f} average mean absolute error; {test_mae*100.0/y.mean():.2f}% error" )
 
