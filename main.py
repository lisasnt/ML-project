import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from scipy import stats
from sklearn.model_selection import train_test_split

## DOMANDE Friday
# 1. correct to use only y_train_train after splitting y_train?
# 2. metodi in serie o parallelo?

# Sum of Squared Errors (SSE)
def calculate_sse(y_true, y_pred):
    sse = np.sum((y_true - y_pred) ** 2)
    return sse

# Coefficient of Determination (R^2)
def calculate_r2(y_true, y_pred):
    sse = calculate_sse(y_true, y_pred)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (sse / sst)
    return r2

def mad_median_outlier_detection(data):
    median = np.median(data)
    mad = np.median(np.abs(data - median)) #Median Absolute Deviation (MAD)
    madn = mad / 0.6745
    #Outlier detection 
    outliers = []
    non_outlier = []
    for i, x in enumerate(data):
        if np.abs(x - median) / madn > 2.24:
            outliers.append(i)  # Record the index of the outlier
        else: 
            non_outlier.append(i)
    return non_outlier, outliers, median, mad, madn

# Load data
X_train = np.load('./input/X_train.npy')    # Air temperature, Water temperature, Wind speed, Wind direction, Illumination
y_train = np.load('./input/y_train.npy')    # Concentration of toxic Algae
X_test = np.load('./input/X_test.npy')      # Input  of the test set       

# TODO boxplot, x_i vs y to explain lasso stuff

# Create test and training set
X_train_train, X_train_test, \
    y_train_train, y_train_test = train_test_split(X_train, y_train, test_size=0.2, random_state=14)

## 1. OUTLAIER REMOVAL
# 1.1 Z-score method
z_scores = np.abs(stats.zscore(y_train_train))
threshold = 3
non_outlier_indices = np.where(z_scores <= threshold)[0]

print("Cleaned set shape after Z-score:",non_outlier_indices.shape[0]) 
#Z-score isn't helpful! It is not suitable when data aren't well distributed    

# 1.2 Interquartile Range (IQR) method 
Q1 = np.percentile(y_train_train, 25)
Q3 = np.percentile(y_train_train, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
non_outlier_indices = np.where((y_train_train >= lower_bound) & (y_train_train <= upper_bound))[0]
outlier_indices = np.where((y_train_train < lower_bound) | (y_train_train > upper_bound))

print("Cleaned set shape after IRQ:",non_outlier_indices.shape[0])

# 1.3 MAD-median rule method
non_outlier, outliers, median, mad, madn = mad_median_outlier_detection(y_train_train)

X_train_train_clean = X_train_train[non_outlier]
y_train_train_clean = y_train_train[non_outlier]

print(f"Outleir removal with MAD method:----------------------------")
print("Cleaned set shape after MAD:", len(non_outlier))
print("Median:", median)
print("MAD:", mad)
print("MADN:", madn)

non_outlier, outliers, median, mad, madn = mad_median_outlier_detection(y_train_test)
X_train_test_clean = X_train_test[non_outlier]
y_train_test_clean = y_train_test[non_outlier]

plt.scatter(np.arange(y_train_train_clean.shape[0]), y_train_train_clean, color='blue', label='Non-Outliers')
plt.scatter(np.arange(y_train_train.shape[0]), y_train_train, color='red', label='Outliers')
plt.title('Visualization of Outliers in y_train')
plt.xlabel('Sample index')
plt.ylabel('Concentration of toxic algae')
plt.show()

## 2. LINEAR REGRESSION
regr = linear_model.LinearRegression()
regr.fit(X_train_train_clean, y_train_train_clean)
y_pred_train_test = regr.predict(X_train_test_clean) 

sse = calculate_sse(y_train_test_clean, y_pred_train_test)
r2 = calculate_r2(y_train_test_clean, y_pred_train_test)
print(f"Linear regression:------------------------------------------")
print(f"Coefficients:", regr.coef_)
print(f"Intercept:", regr.intercept_)
print(f"SSE: {sse}")
print(f"R^2: {r2}")

feature_names = ['Air Temp (x1)', 'Water Temp (x2)', 'Wind Speed (x3)', 'Wind Dir (x4)', 'Illumination (x5)']
for i in range(X_train_test_clean.shape[0]):
    plt.scatter(X_train_test_clean[:, i], y_pred_train_test, color='orange', label='Prediction')
    plt.scatter(X_train_test_clean[:, i], y_train_test_clean, color='blue', label='Test set')
    plt.xlabel(feature_names[i])
    plt.ylabel('Toxic Algae Concentration (y)')
    plt.title(f'{feature_names[i]} vs Toxic Algae Concentration')
    plt.legend()
    plt.show()

'''
## 3. RE    
# Ridge Model
ridge = Ridge()
alpha_range = np.logspace(-4, 4, 50)
ridge_cv = GridSearchCV(ridge, param_grid={'alpha': alpha_range}, cv=5)
ridge_cv.fit(X_train_clean, y_train_clean)
print(f"Best alpha for Ridge: {ridge_cv.best_params_['alpha']}")
y_pred_ridge = ridge_cv.predict(X_train_clean)

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_clean, y_train_clean)
y_pred_lasso = lasso.predict(X_train_clean)


alphas = np.logspace(-4, 4, 10) # values to try
mean_scores = [] # mean cross-validation scores for each alpha

# Loop over each alpha, apply cross-validation, and store the result
for alpha in alphas:
    # Initialize the Lasso model with the current alpha
    lasso = Lasso(alpha=alpha, max_iter=10000)
    
    # Perform 5-fold cross-validation
    # Using cross_val_score to evaluate the model's performance
    # scoring='neg_mean_squared_error' because Lasso minimizes squared error
    scores = cross_val_score(lasso, X_train_clean, y_train_clean, cv=5, scoring='neg_mean_squared_error')
    
    # Compute the mean of the cross-validated scores
    mean_score = np.mean(scores)
    mean_scores.append(mean_score)

# Convert mean_scores list to numpy array
mean_scores = np.array(mean_scores)

# Find the best alpha (the one with the highest mean cross-validation score)
best_alpha = alphas[np.argmax(mean_scores)]

# Output the best alpha and its corresponding score
print(f"Best alpha: {best_alpha}")
#print(f"Best score: {mean_scores[np.argmax(mean_scores)]}")
'''