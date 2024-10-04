import numpy as np
from sklearn.linear_model import LinearRegression

'''
Concepts:
Y = Xθ arx model in matrix form
X (or ϕ) = regressor matrix 
θ = obtained after fit() the linear model
Y = obtained after predict() 
'''
def calculate_sse(y_true, y_pred):
    """
    Calculate the Sum of Squared Errors (SSE)
    """
    sse = np.sum((y_true - y_pred) ** 2)
    return sse

def build_arx_regressor_row(u, y, n, m, d, k):
    """
    Build a single row of the ARX regressor matrix (ϕ) at time step k.
    - u: input data
    - y: output data
    - n: AR order (number of past outputs)
    - m: Input order (number of past inputs)
    - d: Delay
    - k: Current time step
    """
    row = []
    
    # AutoRegressive terms (past output values)
    if n > 0:
        if k >= n:  # Ensure there are enough past values
            row.extend(y[k-1:k-n-1:-1])  # y(k-1), ..., y(k-n)
        else:
            return None  # Not enough past values to build the row

    # Exogenous terms (past input values)
    if m > 0:
        if k >= d + m:  # Ensure there are enough past input values
            row.extend(u[k-d:k-d-m:-1])  # u(k-d), ..., u(k-d-m)
        else:
            return None  # Not enough past values to build the row

    return row if len(row) == (n + m) else None  # Ensure the row has the correct length

def fit_arx(u_train, y_train, n, m, d):
    """
    Fit the ARX model to the training data.
    - u_train: input data (training)
    - y_train: output data (training)
    - n: AR order
    - m: Input order
    - d: Delay
    Returns the trained model.
    """
    N = len(y_train)
    phi_train = []

    # Build the regressor matrix (ϕ)
    for k in range(max(n, d + m), N):
        row = build_arx_regressor_row(u_train, y_train, n, m, d, k)
        if row is not None:
            phi_train.append(row)
    
    phi_train = np.array(phi_train)
    
    if phi_train.shape[1] != (n + m):
        raise ValueError(f"Inconsistent number of features: Expected {n + m} features but got {phi_train.shape[1]}")
    
    y_target = y_train[max(n, d + m):]  # Target values
    
    # Fit the linear regression model
    model = LinearRegression()
    model.fit(phi_train, y_target)

    return model

def predict_arx(u_test, y_train_train, model, n, m, d):
    """
    Predict outputs for the test set using the ARX model.
    - u_test: input data (test set)
    - y_train_train: the past outputs from the training set (to initialize recursive prediction)
    - model: trained linear regression model
    - n: AR order
    - m: Input order
    - d: Delay
    """
    N_test = len(u_test)
    y_pred = []
    
    # Start with the past n outputs from y_train_train
    y_past = list(y_train_train[-n:])  # Use the last n elements of y_train_train for initialization

    # Predict recursively
    for k in range(max(n, d + m), N_test + max(n, d + m)):
        # Build regressor row
        row = build_arx_regressor_row(u_test, y_past, n, m, d, k)
        if row is None:
            break  # If we can't build a valid row, stop prediction
        row = np.array(row).reshape(1, -1)  # Reshape for prediction
        
        # Predict the next output
        y_next = model.predict(row)[0]  # Get the predicted value
        
        # Append the prediction
        y_pred.append(y_next)
        
        # Update the past outputs
        y_past.append(y_next)
        y_past = y_past[-n:]  # Keep only the last n values

    return np.array(y_pred)

def brute_force_arx(u_train, y_train, u_test, y_test, n_range, m_range, d_range):
    """
    Perform a brute-force search to find the best ARX parameters (n, m, d).
    - u_train: input data (training)
    - y_train: output data (training)
    - u_test: input data (testing)
    - y_test: output data (testing)
    - n_range: range for AR order
    - m_range: range for Input order
    - d_range: range for Delay
    """
    best_sse = float('inf')
    best_params = None

    for n in n_range:
        for m in m_range:
            for d in d_range:
                try:
                    # Fit the ARX model with current n, m, d
                    model = fit_arx(u_train, y_train, n, m, d)
                    
                    # Predict test outputs
                    y_pred = predict_arx(u_test, y_train_train, model, n, m, d)

                    # Slice the predictions to ensure they match the test set size
                    y_pred = y_pred[:len(y_test)]  # Adjust prediction size if necessary
                    
                    # Calculate SSE (Sum of Squared Errors)
                    sse = calculate_sse(y_test, y_pred)
                    
                    if sse < best_sse:
                        best_sse = sse
                        best_params = (n, m, d)
                    
                    print(f"n={n}, m={m}, d={d}: SSE={sse}")
                except ValueError as e:
                    print(f"Skipping combination n={n}, m={m}, d={d} due to error: {e}")

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
