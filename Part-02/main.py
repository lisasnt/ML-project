import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

'''
Concepts:
Y=Xθ arx model in matrix form
X (or ϕ) = regressor matrix 
θ = obtained after fit() the linear model
Y = obtained after predict() 
'''
def calculate_sse(y_true, y_pred):
    sse = np.sum((y_true - y_pred) ** 2)
    return sse

class ARXModel(BaseEstimator, RegressorMixin):
    def __init__(self, n=1, m=1, d=1):
        self.n = n  # AR order (past output)
        self.m = m  # Input order (past input)
        self.d = d  # Delay
        self.model = LinearRegression()

    def build_arx_regressor(self, u, y):
        N = len(y)
        phi = []

        for k in range(max(self.n, self.d + self.m), N):
            row = []
            # AutoRegressive terms (past output values)
            if self.n > 0:
                row.extend(y[k-1:k-self.n-1:-1])  # y(k-1), ..., y(k-n)

            # Exogenous terms (past input values)
            if self.m > 0:
                row.extend(u[k-self.d:k-self.d-self.m:-1])  # u(k-d), ..., u(k-d-m)

            # Append the row to phi (regressor matrix)
            if len(row) == (self.n + self.m):
                phi.append(row)
        
        return np.array(phi)

    def fit(self, u, y):
        # Build the regressor matrix
        phi = self.build_arx_regressor(u, y)
        
        # The target values are y[max(self.n, self.d + self.m):]
        y_target = y[max(self.n, self.d + self.m):]
        
        # Fit the linear regression model
        self.model.fit(phi, y_target)
        
        return self

    def predict(self, u):
        return self.model.predict(u)

def brute_force_arx(u_train, y_train, u_test, y_test, n_range, m_range, d_range):
    best_sse = float('inf')
    best_params = None

    for n in n_range:
        for m in m_range:
            for d in d_range:
                model = ARXModel(n, m, d)
                model.fit(u_train, y_train)
                
                # Predict on test set
                y_pred = model.predict(u_test)
                
                # Calculate SSE
                sse = calculate_sse(y_test, y_pred)
                
                if sse < best_sse:
                    best_sse = sse
                    best_params = (n, m, d)
                
                print(f"n={n}, m={m}, d={d}: SSE={sse}")

    return best_params, best_sse

# Load data
u_train = np.load('./input/u_train.npy')
y_train = np.load('./input/output_train.npy')
u_test = np.load('./input/u_test.npy')

n_range = range(1, 10)  # Number of past outputs 
m_range = range(1, 10)  # Number of past inputs
d_range = range(1, 10)  # Delay parameter

test_size = 510
u_train_train = u_train[:-test_size]
u_train_test = u_train[-test_size:]
y_train_train = y_train[:-test_size]
y_train_test = y_train[-test_size:]

# Verify the size of test sets
print(f"u_train_test size: {len(u_train_test)}")
print(f"y_train_test size: {len(y_train_test)}")

# Brute force search for the best ARX parameters
best_model = brute_force_arx(u_train_train, y_train_train, u_train_test, y_train_test, n_range, m_range, d_range)

print("Best model parameters:", best_model)

'''
train_set = np.column_stack(u_train, y_train) #TODO ASK [:, :-1]))  # Exclude the last y value
# Search for the best {n,m,d} set
grid_search = GridSearchCV(estimator=arx, param_grid=param_grid,\
                            cv=3, scoring='neg_mean_squared_error')
grid_search.fit(u_train, y_train)

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
'''

