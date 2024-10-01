import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error

class ARXModel(BaseEstimator, RegressorMixin):
    def __init__(self, n=1, m=1, d=1):
        self.n = n  # AR order
        self.m = m  # Input order
        self.d = d  # Delay
        self.model = LinearRegression()

    def build_arx_regressor(self, u, y):
        N = len(y)
        phi = []

        for k in range(max(self.n, self.d + self.m), N):
            row = []
            # AutoRegressive terms (past output values)
            row.extend(y[k-1:k-self.n-1:-1])  # y(k-1), ..., y(k-n)

            # eXogenous terms (past input values)
            row.extend(u[k-self.d:k-self.d-self.m:-1])  # u(k-d), ..., u(k-d-m)

            phi.append(row)
        
        return np.array(phi)

    def fit(self, X, y):
        u, y_data = X[:, 0], X[:, 1]
        
        # Build regressor matrix
        phi = self.build_arx_regressor(u, y_data)
        y_trimmed = y_data[max(self.n, self.d + self.m):]

        # Fit the Linear Regression model
        self.model.fit(phi, y_trimmed)
        return self

    def predict(self, X):
        u, y_data = X[:, 0], X[:, 1]
        phi = self.build_arx_regressor(u, y_data)
        return self.model.predict(phi)

    def score(self, X, y_true):
        # Trim y_true to match the length of y_pred
        u, y_data = X[:, 0], X[:, 1]
        phi = self.build_arx_regressor(u, y_data)
        y_pred = self.model.predict(phi)
        
        # Align lengths for scoring
        y_true_trimmed = y_true[max(self.n, self.d + self.m):]
        
        # Now both y_true_trimmed and y_pred have consistent lengths
        return -mean_squared_error(y_true_trimmed, y_pred)

# Load data
u_train = np.load('./input/u_train.npy')
y_train = np.load('./input/output_train.npy')
u_test = np.load('./input/u_test.npy')

# Combine input and output into a single dataset for GridSearchCV
X_train = np.column_stack((u_train, y_train))

# Define the parameter grid for n, m, and d
param_grid = {
    'n': range(1, 10), # Number of past outputs 
    'm': range(1, 10), # Number of past inputs
    'd': range(1, 10)  # Delay parameter
}

# Fit model using GridSearchCV
arx = ARXModel()
grid_search = GridSearchCV(estimator=arx, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)

# Get best model parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"Best parameters: {best_params}")
print(f"Best MSE: {-best_score}")

# Now use the best model to predict on the test set
X_test = np.column_stack((u_test, np.zeros(len(u_test))))  # y_test is unknown, but 0s are used to fill the y column
y_test_pred = best_model.predict(X_test)

# Extract last 400 samples for submission
y_submission = y_test_pred[-400:]

# Save the submission file
#np.save('y_test_submission.npy', y_submission)
