# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor_SVR = SVR(kernel = 'rbf')
regressor_SVR.fit(X, y)

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor_DTR = DecisionTreeRegressor(random_state = 0)
regressor_DTR.fit(X, y)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor_RFR = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor_RFR.fit(X, y)

"""
#6.5 can be replaced accordingly with required level
# Predicting a new result with SVR
y_pred_SVR = regressor_SVR.predict(sc_X.transform(np.array([[6.5]])))
y_pred_SVR = sc_y.inverse_transform(y_pred_SVR)

# Predicting a new result with Decision Tree Regression 
y_pred_DTR = regressor_DTR.predict(sc_X.transform(np.array([[6.5]])))
y_pred_DTR = sc_y.inverse_transform(y_pred_DTR)

# Predicting a new result with Random Forest Regression
y_pred_RFR = regressor_RFR.predict(sc_X.transform(np.array([[6.5]])))
y_pred_RFR = sc_y.inverse_transform(y_pred_RFR)"""

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor_SVR.predict(X_grid), color = 'blue')
plt.plot(X_grid, regressor_RFR.predict(X_grid), color = 'black')
plt.plot(X_grid, regressor_DTR.predict(X_grid), color = 'yellow')
plt.title('SVR = Blue\nDecision Tree Regression = Yellow\nRandom Forest Regression = Black')
plt.xlabel('Position level(Feature Scaled)')
plt.ylabel('Salary(Feature Scaled)')
plt.show()

