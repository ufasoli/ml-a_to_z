# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting simple Linear regresion to the training set
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
#regressor learn correlation between sets (in our case years of experience --> salary
regressor.fit(X_train,y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test) #vector of predictions for the dependent variable (in this case salaries) using test data

#visualising the TRAINING  results
plt.scatter(X_train, y_train, color='red') #create a scatter plot
plt.plot(X_train, regressor.predict(X_train), color='blue')#plot and predict using train data

plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#visualising the TEST results
plt.scatter(X_test, y_test, color='red') #create a scatter plot
plt.plot(X_train, regressor.predict(X_train), color='blue')#plot and predict using train data

plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()