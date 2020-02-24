#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn import LinearRegression

### CHEN: LinearRegression should be imported from sklearn.linear_model.
### Thus, it should be corrected to: from sklearn.linear_model import LinearRegression

from sklearn.cross_validation import cross_val_score

### CHEN: Again, cross_val_score is from sklearn.model_selection
### Thus it should be corrected to: from sklearn.model_selection import cross_val_score

# Load data
d = pd.read_csv('../data/train.csv')

### CHEN: The data are loaded into variable d. However, x1 and x2 are referenced to data.
### consider using: data = pd.read_csv('../data/train.csv')

# Setup data for prediction
x1 = data.SalaryNormalized
x2 = pd.get_dummies(data.ContractType)

### CHEN: while there is no issue of using x1 and x2 from a program prospective, 
### it's a good practice to separate dependent and independent variables using y and x respectively.
### In this case, you should rename your x1 as y.

# Setup model
model = LinearRegression()

# Evaluate model
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split

### CHEN: Note that there is no need import cross_val_score, since you have already import this previously.
### Furthermore, train_test_split was not used. Thus these two lines can be omitted
### For furture reference, train_test_split is a part of sklearn.model_selection.
### Thus, the code should be: from sklearn.model_selection import train_test_split

scores = cross_val_score(model, x2, x1, cv=1, scoring='mean_absolute_error')
print(scores.mean())

### CHEN: One fold cross validation is the same as having no cross validation.
### Remember, in k-fold, data is split into k groups. So here, the data are not split

### CHEN: To use mean_absolute_error scoring function, the proper string should be
### scoring = 'neg_mean_absolute_error'
