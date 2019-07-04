# Decision Tree Regression



# In this example we are an HR team
# hiring a new employee and negotiation is taken place
# to make an offer regarding salary expectation
# employee is asking for at least 160k --> based on the dataset
# salaries are not a linear function but polynomial (employee is a 6.5 level)


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# create tree regressor
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)  # random_stat=0 to have same seed as tutorial (and so same result)
regressor.fit(X, y)

#predict
y_pred = regressor.predict([[6.5]])

# high resolution model needed to plot decision trees
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid)), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or bluff (Regression Tree)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()