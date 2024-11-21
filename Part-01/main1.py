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

feature_names = ['Air Temp (x1)', 'Water Temp (x2)', 'Wind Speed (x3)', 'Wind Dir (x4)', 'Illumination (x5)']

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
print("Number of inliers after IRQ:",filtered_data.shape[0])  

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
outlier_mask = ~inlier_mask
X_train_clean = X_train[inlier_mask]
y_train_clean = y_train[inlier_mask]
print(f"Number of inliers after RANSAC: {np.sum(inlier_mask)}")
'''
# Plot outliers in red
fig, axs = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Features vs Toxic Algae Concentration (Inliers and Outliers)', fontsize=16)
for i, ax in enumerate(axs.flat):
    if i < len(feature_names):
        ax.scatter(X_train[inlier_mask, i], y_train[inlier_mask], 
                   color='blue', label='Inliers', alpha=0.5)
        ax.scatter(X_train[outlier_mask, i], y_train[outlier_mask], 
                   color='red', label='Outliers', alpha=0.5)
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel('Toxic Algae Concentration (y)')
        ax.set_title(f'{feature_names[i]} vs Toxic Algae Concentration')
        ax.legend()
    else:
        fig.delaxes(ax) 
plt.tight_layout()
plt.show()
'''
# Create test and training set
X_train_clean_train, X_train_clean_test, \
    y_train_clean_train, y_train_clean_test = train_test_split(X_train_clean, y_train_clean, test_size=0.2, random_state=14)

## 2. LINEAR REGRESSION
regr = linear_model.LinearRegression()
regr.fit(X_train_clean_train, y_train_clean_train)
y_pred_train_test = regr.predict(X_train_clean_test)

sse = calculate_sse(y_train_clean_test, y_pred_train_test)
r2 = calculate_r2(y_train_clean_test, y_pred_train_test)
print(f"Linear Regression Coefficients:", regr.coef_)
print(f"Linear Regression Intercept:", regr.intercept_)
print(f"Linear Regression SSE:\t{sse}")
print(f"Linear Regression R^2:\t{r2}")

## 3. REGULARIZATION
# Ridge Model
ridge = linear_model.Ridge()
alpha_range = np.logspace(-4, 4, 50)
ridge_cv = GridSearchCV(ridge, param_grid={'alpha': alpha_range}, cv=5)
ridge_cv.fit(X_train_clean_train, y_train_clean_train)
#print(f"Best alpha for Ridge: {ridge_cv.best_params_['alpha']}")
y_pred_ridge = ridge_cv.predict(X_train_clean_test)
sse = calculate_sse(y_train_clean_test,y_pred_ridge)
r2 = calculate_r2(y_train_clean_test,y_pred_ridge)
print("Ridge Best alpha:\t", ridge_cv.best_params_['alpha'])
print("Ridge Coefficients:\t", ridge_cv.best_estimator_.coef_)
print("Ridge Intercept:\t", ridge_cv.best_estimator_.intercept_)
print("Ridge SSE:\t\t", sse)
print("Ridge R^2:\t\t", r2)

# Lasso Regression
lasso = linear_model.LassoCV(cv=15, random_state=190)
lasso.fit(X_train_clean_train, y_train_clean_train)
alpha_lasso = lasso.alpha_
beta_lasso = lasso.coef_
y_pred_lassoCV = lasso.predict(X_train_clean_test)
sse = calculate_sse(y_train_clean_test,y_pred_lassoCV)
r2 = calculate_r2(y_train_clean_test,y_pred_lassoCV)
print("Lasso Best alpha:\t", lasso.alpha_)
print("Lasso Coefficients:\t", lasso.coef_)
print("Lasso Intercept:\t", lasso.intercept_)
print("Lasso SSE:\t\t", sse)
print("Lasso R^2:\t\t", r2)

'''
# Lasso Regression with grid search cv
lasso = linear_model.Lasso()
param_grid = {'alpha': [1, 10, 100]}
grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=15)
grid_search.fit(X_train_clean_train, y_train_clean_train)
alpha_lasso = grid_search.best_params_['alpha']
#print(f"Best alpha from GridSearchCV: {alpha_lasso}")
best_lasso = grid_search.best_estimator_
y_pred_lasso = best_lasso.predict(X_train_clean_test)
sse = calculate_sse(y_train_clean_test, y_pred_lasso)
r2 = calculate_r2(y_train_clean_test, y_pred_lasso)
print("Lasso SSE with GridSearchCV: ", sse)
print(f"Linear Regression R^2: {r2}")
'''

#Lasso is the best one
y_test = lasso.predict(X_test)
#np.save('y_test.npy', y_test)  
'''
# Plot Predictions vs test set
fig, axs = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Features vs Toxic Algae Concentration', fontsize=16)
for i, ax in enumerate(axs.flat):
    if i < len(feature_names):
        ax.scatter(X_train_clean_test[:, i], y_pred_train_test, color='red', label='Prediction', alpha=0.5)
        ax.scatter(X_train_clean_test[:, i], y_train_clean_test, color='blue', label='Test set', alpha=0.5)
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel('Toxic Algae Concentration (y)')
        ax.set_title(f'{feature_names[i]} vs Toxic Algae Concentration')
        ax.legend()
    else:
        fig.delaxes(ax) 
plt.tight_layout()
plt.show()
'''