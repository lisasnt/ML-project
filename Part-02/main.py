import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

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
        self.phi = None  # Placeholder for storing the regressor matrix

    def build_arx_regressor(self, u, y):
        N = len(y)
        phi = []

        # Iterate from max(self.n, self.d + self.m) to ensure we have enough past data
        for k in range(max(self.n, self.d + self.m), N):
            row = []
            
            # AutoRegressive terms (past output values)
            if self.n > 0:
                row.extend(y[k-1:k-self.n-1:-1])  # y(k-1), ..., y(k-n)

            # Exogenous terms (past input values)
            if self.m > 0:
                row.extend(u[k-self.d:k-self.d-self.m:-1])  # u(k-d), ..., u(k-d-m)

            # Append the row to the regressor matrix (ϕ)
            phi.append(row)
        
        return np.array(phi)

    def fit(self, u, y):
        # Build the ARX regressor matrix using past input (u) and output (y)
        self.phi = self.build_arx_regressor(u, y)  # Store regressor matrix in the instance
        
        # The target values are y[max(self.n, self.d + self.m):]
        y_target = y[max(self.n, self.d + self.m):]
        
        # Fit the linear regression model with the regressor matrix (X) and target values (Y)
        self.model.fit(self.phi, y_target)
        
        return self

    def predict(self, u, phi0, uold):
        # Use the pre-built regressor matrix from fit
        if self.phi is None:
            raise ValueError("Model has not been fitted yet.")
        
        phiI = phi0
        
        predictions = []
        print(len(u))
        for i in range(len(u)):
            print(i)
            yI = self.model.predict(phiI)
            predictions.append(yI)

            if i < (self.d + self.m):
                phiI = yI.append(phiI)
                phiI[self.n] = uold[-1]
                phiI.pop()
                uold.pop()

            else:
                phiI = yI.append(phiI)
                phiI[self.n] = u[i-self.d-self.m]
                phiI.pop()
                

        
        # Use the saved phi for prediction (no need to rebuild the regressor matrix)
        return predictions

def brute_force_arx(u_train, y_train, u_test, y_test, n_range, m_range, d_range):
    best_sse = float('inf')
    best_params = None

    for n in n_range:
        for m in m_range:
            for d in d_range:
                model = ARXModel(n, m, d)
                
                # Fit the model with training data
                model.fit(u_train, y_train)
                
                # Predict using the saved regressor matrix
                # Step 1: Take the last 'n' elements from vector y
                phi0 = model.y[-n:].tolist()

                # Step 2: Add elements from vector u, from index end-d to end-d-m
                u_part = model.u[-(d+1):-(d+m+1):-1].tolist()

                # Step 3: Concatenate both parts
                phi0.extend(u_part)

                uold = model.u[-d:].tolist()

                

                y_pred = model.predict(u_test,phi0,uold)  # Using the stored phi
                
                # Calculate SSE (Sum of Squared Errors)
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

# Manually split train/test data
test_size = 400  
u_train_train = u_train[:-test_size]
u_train_test = u_train[-test_size:]
y_train_train = y_train[:-test_size]
y_train_test = y_train[-test_size:]

# Brute force search for the best ARX parameters
best_params, best_sse = brute_force_arx(u_train_train, y_train_train, u_train_test, y_train_test, n_range, m_range, d_range)

print("Best model parameters:", best_params)
print("Best SSE:", best_sse)
print()