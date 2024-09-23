import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import linear_model

# Load data
X_train = np.load('./X_train.npy') # Air temperature, Water temperature, Wind speed, Wind direction, Illumination
y_train = np.load('./y_train.npy') # Concentration of toxic Algae
X_test = np.load('./X_test.npy')

## 1. LINEAR REGRESSION
# Create linear regression object, train the model and make predictions
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
algae_y_pred = regr.predict(X_test)

print(algae_y_pred)
print("Coefficients: \n", regr.coef_)
print("Interception: \n", regr.intercept_)

# Plot outputs
#plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(X_test, algae_y_pred, color="blue", linewidth=1)

plt.xticks(())
plt.yticks(())

plt.show()

## 2. OUTLAIER REMOVAL

## 2. REGULARIZATION