# SVR

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

# Feature scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))

# Fitting SVR into dataset
from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(X, y)  # x = Matrix of features --> y dependent variables

''' 
we need to scale the value due to the fact we did feature scaling on the processor sx_X.transform and then apply the 
inverse transform function to get the value in the correct scale
'''

y_pred = sc_y.inverse_transform(
    regressor.predict(
        sc_X.transform(np.array([[6.5]]))
    ))

# apply inverse transform method to obtain the value without the scaling


# Visualise svr test results
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# visualize svr test results in 'high' resolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Thruth or Bluff (SVR Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
