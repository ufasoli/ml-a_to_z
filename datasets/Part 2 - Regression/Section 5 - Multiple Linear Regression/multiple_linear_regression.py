# Multiple Linear Regression
# In this example we'll try to find out
# if there's a correlation between profit and money spent in
# R&D Admin, Marketing State
# y = dependent variable (profit))
# y= b0 + b1*x1 (R&D spend) + b2*x2 (admin spend) + b3*x3 (Marketing spend) + ??? (state is a categorical variables)
# when dealing with categorical values 'dummy variables' need to be created -->b4*D1 (dummy variable)


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values  # independent variables (observations)
y = dataset.iloc[:, 4].values  # dependent variable --> profit

# Encoding categorical data
# Encoding the independent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])  # encode categorical variables (string into numbers) in this case the state

# transpose values encoded into columns otherwise simple label encoding will assign values with significance (dummy variables)
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# avoiding the dummy variable trap
X = X[:, 1:]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Multiple linear regression to the training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)  # fit regressor to trainning set

# predicting the test set results
y_pred = regressor.predict(X_test)

# backwards elimination regression

import statsmodels.api as sm

# we need to add the constant variable of the linear regression equation (see linear regression equation b0)

# add an array of the size of the data (array filled with 1 as a column (axis=1))
X = np.append(arr=np.ones((50, 1)), values=X.astype(int), axis=1)
X_optimal = X[:, [0, 1, 2, 3, 4, 5]]  # optimal matrix of features (variables that are statistically signifiant (i.e. SL > 0.05)
regressor_OLS = sm.OLS(endog=y, exog=X_optimal).fit()
regressor_OLS.summary()

X_optimal = X[:, [0, 1, 3, 4, 5]]  # optimal matrix of features (variables that are statistically signifiant)
regressor_OLS = sm.OLS(endog=y, exog=X_optimal).fit()

regressor_OLS.summary()


X_optimal = X[:, [0, 3, 4, 5]]  # optimal matrix of features (variables that are statistically signifiant)
regressor_OLS = sm.OLS(endog=y, exog=X_optimal).fit()

regressor_OLS.summary()


X_optimal = X[:, [0, 3, 5]]  # optimal matrix of features (variables that are statistically signifiant)
regressor_OLS = sm.OLS(endog=y, exog=X_optimal).fit()

regressor_OLS.summary()


X_optimal = X[:, [0, 3]]  # optimal matrix of features (variables that are statistically signifiant)
regressor_OLS = sm.OLS(endog=y, exog=X_optimal).fit()

regressor_OLS.summary()



