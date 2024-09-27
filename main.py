import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, train_test_split
from scipy import stats

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
    inlier = []
    for i, x in enumerate(data):
        if np.abs(x - median) / madn > 2.24:
            outliers.append(i)  # Record the index of the outlier
        else: 
            inlier.append(i)
    return inlier, outliers, median, mad, madn

# Load data
X_train = np.load('./input/X_train.npy')    # Air temperature, Water temperature, Wind speed, Wind direction, Illumination
y_train = np.load('./input/y_train.npy')    # Concentration of toxic Algae
X_test = np.load('./input/X_test.npy')      # Input  of the test set       

# TODO boxplot, x_i vs y to explain lasso stuff

'''
# Normalization
X_train_train = X_train_train /np.linalg.norm(X_train_train)
y_train_train = y_train_train /np.linalg.norm(y_train_train)
X_train_test = X_train_test /np.linalg.norm(X_train_test)
y_train_test = y_train_test /np.linalg.norm(y_train_test)
'''

## 1. OUTLAIER REMOVAL
# 1.1 Z-score method
z_scores = np.abs(stats.zscore(y_train))
threshold = 3
inlier_indices = np.where(z_scores <= threshold)[0]

print("Number of inliers after Z-score:",inlier_indices.shape[0]) 
#Z-score isn't helpful! It is not suitable when data aren't well distributed    

# 1.2 Interquartile Range (IQR) method 
traindata = pd.DataFrame(X_train, columns=['X1', 'X2', 'X3', 'X4', 'X5'])
traindata['y'] = y_train
size0 = traindata.shape[0]
def iqr_filter(data_column):
    Q1 = np.percentile(data_column, 25)
    Q3 = np.percentile(data_column, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data_column >= lower_bound) & (data_column <= upper_bound)
filtered_data = traindata.copy()
for column in filtered_data.columns:
    filtered_data = filtered_data[iqr_filter(filtered_data[column])]
print("Number of inliers after IRQ: ",filtered_data.shape[0])
'''
Q1 = np.percentile(y_train, 25)
Q3 = np.percentile(y_train, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
inlier_indices = np.where((y_train >= lower_bound) & (y_train <= upper_bound))[0]
outlier_indices = np.where((y_train < lower_bound) | (y_train > upper_bound))
print("Number of inliers after IRQ:",inlier_indices.shape[0])
'''
#Not that helpful  

'''
# 1.3 MAD-median rule method
inlier, outliers, median, mad, madn = mad_median_outlier_detection(y_train)
print("Number of inliers after MAD:", len(inlier))
'''

# 1.4 RANSAC regression and then outlier removal
ransac = linear_model.RANSACRegressor(random_state=190)
ransac.fit(X_train, y_train)
y_pred_ransac = ransac.predict(X_train)
inlier_mask = ransac.inlier_mask_
X_train_clean = X_train[inlier_mask]
y_train_clean = y_train[inlier_mask]
print(f"Number of inliers after RANSAC: {np.sum(inlier_mask)}")

# Create test and training set
X_train_clean_train, X_train_clean_test, \
    y_train_clean_train, y_train_clean_test = train_test_split(X_train_clean, y_train_clean, test_size=0.2, random_state=14)

inlier, outliers, median, mad, madn = mad_median_outlier_detection(y_train_clean)

## 2. LINEAR REGRESSION
regr = linear_model.LinearRegression()
regr.fit(X_train_clean_train, y_train_clean_train)
y_pred_train_test = regr.predict(X_train_clean_test)

sse = calculate_sse(y_train_clean_test, y_pred_train_test)
r2 = calculate_r2(y_train_clean_test, y_pred_train_test)
#print(f"Coefficients:", regr.coef_)
#print(f"Intercept:", regr.intercept_)
print(f"Linear Regression SSE: {sse}")
print(f"Linear Regression R^2: {r2}")

feature_names = ['Air Temp (x1)', 'Water Temp (x2)', 'Wind Speed (x3)', 'Wind Dir (x4)', 'Illumination (x5)']
for i in range(len(feature_names)):
    plt.scatter(X_train_clean_test[:, i], y_pred_train_test, color='orange', label='Prediction')
    plt.scatter(X_train_clean_test[:, i], y_train_clean_test, color='blue', label='Test set')
    plt.xlabel(feature_names[i])
    plt.ylabel('Toxic Algae Concentration (y)')
    plt.title(f'{feature_names[i]} vs Toxic Algae Concentration')
    plt.legend()
 #   plt.show()
'''
v1 = np.linspace(0, 0.3, 1000)
v2 = np.linspace(0, 0.3, 1000)
v3 = np.linspace(0, 0.3, 1000)
v4 = np.linspace(0, 0.3, 1000)
v5 = np.linspace(0, 0.3, 1000)
input = np.zeros(shape=[1000,5])
input[:,0]=v1
input[:,1]=v2
input[:,2]=v3
input[:,3]=v4
input[:,4]=v5

y_pred_train_test = regr.predict(input)
feature_names = ['Air Temp (x1)', 'Water Temp (x2)', 'Wind Speed (x3)', 'Wind Dir (x4)', 'Illumination (x5)']
for i in range(len(feature_names)):
    plt.scatter(input[:, i], y_pred_train_test, color='orange', label='Prediction')
    plt.scatter(X_train_clean_test[:, i], y_train_clean_test, color='blue', label='Test set')
    plt.xlabel(feature_names[i])
    plt.ylabel('Toxic Algae Concentration (y)')
    plt.title(f'{feature_names[i]} vs Toxic Algae Concentration')
    plt.legend()
    plt.show()
'''

## 3. REGULARIZATION
# Ridge Model
ridge = linear_model.Ridge()
alpha_range = np.logspace(-4, 4, 50)
ridge_cv = GridSearchCV(ridge, param_grid={'alpha': alpha_range}, cv=5)
ridge_cv.fit(X_train_clean_train, y_train_clean_train)
print(f"Best alpha for Ridge: {ridge_cv.best_params_['alpha']}")
y_pred_ridge = ridge_cv.predict(X_train_clean_test)
sse = calculate_sse(y_train_clean_test,y_pred_ridge)
print("Ridge SSE: ", sse)

# Lasso Regression
lasso = linear_model.LassoCV(cv=15, random_state=190)
lasso.fit(X_train_clean_train, y_train_clean_train)
alpha_lasso = lasso.alpha_
beta_lasso = lasso.coef_
print(f"Best alpha for Lasso: {lasso.alpha_}")
y_pred_lassoCV = lasso.predict(X_train_clean_test)
sse = calculate_sse(y_train_clean_test,y_pred_lassoCV)
print("Lasso SSE: ", sse)
    
# Lasso Regression with grid search cv
lasso = linear_model.Lasso()
param_grid = {'alpha': [1, 10, 100]}
grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=15)
grid_search.fit(X_train_clean_train, y_train_clean_train)
alpha_lasso = grid_search.best_params_['alpha']
print(f"Best alpha from GridSearchCV: {alpha_lasso}")
best_lasso = grid_search.best_estimator_
y_pred_lasso = best_lasso.predict(X_train_clean_test)
sse = calculate_sse(y_train_clean_test, y_pred_lasso)
print("Lasso SSE with GridSearchCV: ", sse)

