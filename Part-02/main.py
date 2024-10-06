import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV    
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
        self.model = LinearRegression() #Tried RidgeCV, LinearRegression and LassoCV. Lasso was bad, Ridge best and Lin similar
        self.phi = None  # Placeholder for storing the regressor matrix
    
    def build_arx_regressor(self, u, y):
        N = len(y)
        phi = []
    
        # Determine the expected length of each row in phi
        expected_length = self.n + self.m + 1
    
        # Iterate from max(self.n, self.d + self.m) to ensure we have enough past data
        for k in range(max(self.n, self.d + self.m), N): #k is going from p to N-1
            row = []
        
            # AutoRegressive terms (past output values)
            #if self.n > 0:

            #row.extend(y[k-1:k-self.n-1:-1])  # y(k-1), ..., y(k-n)

            y_reverse = (y[k-self.n:k:1])[::-1] #This is needed to get the first row of phi
            row.extend(y_reverse)  # y(k-1), ..., y(k-n)
        
            # Exogenous terms (past input values)
            #if self.m > 0:
            
            #row.extend(u[k-self.d:k-self.d-self.m-1:-1])  # u(k-d), ..., u(k-d-m)
            u_reverse = (u[k-self.d-self.m:k-self.d+1:1])[::-1]
            row.extend(u_reverse)
            
        # Check if the row length matches the expected length
            if len(row) == expected_length:
                phi.append(row)
            else:
                
                print(f"Warning: Skipping row at k={k} when n={self.n}, d={self.d} and m={self.m} because of inconsistent length")

        return np.array(phi)

    def fit(self, u, y):
        # Build the ARX regressor matrix using past input (u) and output (y)
        self.phi = self.build_arx_regressor(u, y)  # Store regressor matrix in the instance
        
        # The target values are y[max(self.n, self.d + self.m):]
        y_target = y[max(self.n, self.d + self.m):]
        
        # Fit the linear regression model with the regressor matrix (X) and target values (Y)
        self.model.fit(self.phi, y_target)
        
        return self
    
    def check_stability(self):
        """
        Check the stability of the ARX model based on the roots of the AR characteristic equation.
        Stability criterion: All roots must lie inside the unit circle in the complex plane.
        """
        if self.ar_coefficients is None:
            raise ValueError("Model has not been fitted yet or AR coefficients are not available.")
        
        # Form the AR characteristic polynomial: 1 - θ_1 * z^{-1} - θ_2 * z^{-2} - ... - θ_n * z^{-n}
        # We need to find the roots of the polynomial with coefficients [1, -θ_1, -θ_2, ..., -θ_n]
        poly_coeffs = np.concatenate(([1], -self.ar_coefficients))
        
        # Calculate the roots of the characteristic polynomial
        roots = np.roots(poly_coeffs)
        
        # Check if all roots lie inside the unit circle
        stable = np.all(np.abs(roots) < 1)
        
        if stable:
            return True
        else:
            return False
        
    def predict(self, u, phi0, uold):
        # Use the pre-built regressor matrix from fit
        if self.phi is None:
            raise ValueError("Model has not been fitted yet.")
        
        phi_k = phi0.copy()  # Make sure we don't modify the original phi0 list
        predictions = []

        for k in range(len(u)):
            y_k = self.model.predict([phi_k])[0]  # Model expects 2D array
            
            predictions.append(y_k)
            
            # Update phiI (regressor matrix) for the next time step
            if len(uold) != 0:
                # Use elements from uold for the initial iterations
                phi_k = [y_k] + (phi_k[:-1])  # Add the latest prediction at the start and pop the last element
                phi_k[self.n] = uold[-1]  # Replace the relevant u term
                uold.pop()  # Update uold
            else:
                # Use elements from the u sequence after the initial iterations
                phi_k = [y_k] + (phi_k[:-1])  # Add the latest prediction at the start and pop the last element
                phi_k[self.n] = u[k - self.d]  # Replace the relevant u term
        
        return np.array(predictions)


def brute_force_arx(u_train, y_train, u_test, y_test, n_range, m_range, d_range):
    best_sse = float('inf')
    best_params = None
    best_ypred = []

    for n in n_range:
        for m in m_range:
            for d in d_range:
                model = ARXModel(n, m, d)
                
                # Fit the model with training data
                model.fit(u_train, y_train)

                if model.check_stability:
                    # Predict using the saved regressor matrix
                    # Step 1: Take the last 'n' elements from vector y
                    phi0 = y_train[-1:-(n+1):-1].tolist()

                    # Step 2: Add elements from vector u, from index end-d to end-d-m
                
                    u_part = u_train[-(d):-(d+m+1):-1].tolist()

                    # Step 3: Concatenate both parts
                    phi0.extend(u_part)

                    uold = u_train[-1:-d:-1].tolist()

                    y_pred = model.predict(u_test, phi0, uold)  # Using the stored phi
                
                    # Calculate SSE (Sum of Squared Errors)
                    sse = calculate_sse(y_test, y_pred)
                
                    if sse < best_sse:
                        best_sse = sse
                        best_params = (n, m, d)
                        best_ypred = y_pred
                        print('Current best parameters(n,m,d)',best_params)
                        print('Current best SSE, ', sse)
                        #print('Current best alpha', model.model.alpha_)
                
    return best_params, best_sse, best_ypred

def plot_validation(y_true, y_pred, u, test_size):
    
    # Generate x values corresponding to the index of the data points
    x_values_train = range(len(y_true))  # Last test_size points of y_train
    x_values_pred = range(len(y_true)-test_size, len(y_true))  # Points for best_ypred

    plt.figure(figsize=(10, 6))

    # Plot the last 400 points of y_train
    plt.plot(x_values_train, y_train, label='True y_train', color='blue')

    # Plot the predicted values
    plt.plot(x_values_pred, y_pred, label='Best Prediction (y_pred)', color='red', linestyle='--')

    # Plot input values
    plt.plot(x_values_train, u, label='Input values u', color='green')

    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Comparison of y_train and Best Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_results(y_train, best_ypred, u_train, u_test):
    # Generate x values corresponding to the index of the data points
    x_values_train = range(len(y_train))  # Last test_size points of y_train
    x_values_pred = range(len(y_train)+1, len(y_train) + len(best_ypred)+1)  # Points for best_ypred

    plt.figure(figsize=(10, 6))

    # Plot the last 400 points of y_train
    plt.plot(x_values_train, y_train, label='True y_train', color='blue')

    # Plot the predicted values
    plt.plot(x_values_pred, best_ypred, label='Best Prediction (y_pred)', color='red', linestyle='--')

    plt.plot(x_values_train, u_train, label='Input values u', color='green')

    plt.plot(x_values_pred, u_test, label='Input values u', color='green')

    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Comparison of y_train and Best Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()

# Load data
u_train = np.load('./input/u_train.npy')
y_train = np.load('./input/output_train.npy')
u_test = np.load('./input/u_test.npy')

n_range = range(1, 10)  # Number of past outputs 
m_range = range(1, 10)  # Number of past inputs
d_range = range(1, 10)  # Delay parameter

# Manually split train/test data
test_size = 510 #int(len(u_train)*0.2)
print('test size =',test_size)
u_train_train = u_train[:-test_size]
u_train_test = u_train[-test_size:]
y_train_train = y_train[:-test_size]
y_train_test = y_train[-test_size:]

# Brute force search for the best ARX parameters
best_params, best_sse, best_ypred = brute_force_arx(u_train_train, y_train_train, u_train_test, y_train_test, n_range, m_range, d_range)

print("Best model parameters:", best_params)
print("Best SSE:", best_sse)
print()
# Plotting the results: y_train (last 400 points) and best_ypred (prediction)
plot_validation(y_train, best_ypred,u_train, test_size)

model = ARXModel()
model.n, model.m, model.d = best_params
model.fit(u_train,y_train)

# Step 1: Take the last 'n' elements from vector y
phi0 = y_train[-1:-(model.n+1):-1].tolist()

# Step 2: Add elements from vector u, from index end-d to end-d-m
u_part = u_train[-(model.d):-(model.d+model.m+1):-1].tolist()

# Step 3: Concatenate both parts
phi0.extend(u_part)

uold = u_train[-1:-model.d:-1].tolist()

y_test = model.predict(u_test,phi0,uold)

y_submit = y_test[-400:]
print('Check submission has right amount of elements:', len(y_submit))

#np.save('y_submit.npy', y_submit) #Save last 400 values for submission

#print('Alpha=',model.model.alpha_)
plot_results(y_train,y_test, u_train, u_test)