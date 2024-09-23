import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import pandas 

from sklearn import linear_model
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from scipy import stats

# Load data
X_train = np.load('./X_train.npy') # Air temperature, Water temperature, Wind speed, Wind direction, Illumination
y_train = np.load('./y_train.npy') # Concentration of toxic Algae
X_test = np.load('./X_test.npy')

## 1. LINEAR REGRESSION
# Create linear regression object, train the model and make predictions
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
raw_y_pred = regr.predict(X_test)
np.save('raw_y_pred.npy', raw_y_pred)

print("Coefficients: \n", regr.coef_)
print("Intercept: \n", regr.intercept_)

# Plot outputs with raw set
# TODO boxplot, x_i vs y to explayin lasso stuff

#plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
#plt.plot(X_test, raw_y_pred, color="blue", linewidth=1)
#plt.xticks(())
#plt.yticks(())

plt.scatter(np.arange(200), y_train)
plt.title('Scatter plot of y (with potential outliers)')
plt.xlabel('Sample index')
plt.ylabel('Concentration of toxic algae')
#plt.show()

## 2. OUTLAIER REMOVAL
# 2.1 Z-score method
z_scores = np.abs(stats.zscore(y_train))
threshold = 3
non_outlier_indices = np.where(z_scores <= threshold)[0]

print("Cleaned set after Z-score (<200):",non_outlier_indices.shape) 
#Z-score isn't helpful! It is not suitable when data aren't well distributed    

# 2.2 Interquartile Range (IQR) method 
Q1 = np.percentile(y_train, 25)
Q3 = np.percentile(y_train, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
non_outlier_indices = np.where((y_train >= lower_bound) & (y_train <= upper_bound))[0]
outlier_indices = np.where((y_train < lower_bound) | (y_train > upper_bound))

print("Cleaned set after IRQ (<200):",non_outlier_indices.shape)

X_train_clean = X_train[non_outlier_indices]
y_train_clean = y_train[non_outlier_indices]

plt.scatter(np.arange(y_train_clean.shape[0]), y_train_clean, color='blue', label='Non-Outliers')
plt.scatter(np.arange(y_train.shape[0]), y_train, color='red', label='Outliers')
plt.title('Visualization of Outliers in y_train')
plt.xlabel('Sample index')
plt.ylabel('Concentration of toxic algae')
#plt.show()

# 2.3 MAD-median rule method
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

non_outlier, outliers, median, mad, madn = mad_median_outlier_detection(y_train)

print("Detected outlier indices:", outliers)
print("Median:", median)
print("MAD:", mad)
print("MADN:", madn)
print(len(outliers))

X_train_clean = X_train[non_outlier]
y_train_clean = y_train[non_outlier]

## 2. REGULARIZATION
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
print(y_pred_lasso)
print(len(y_pred_lasso))

lasso = Lasso()
lasso_cv = GridSearchCV(lasso, param_grid={'alpha': alpha_range}, cv=5c)
#lasso_cv.fit(X_train_clean, y_train_clean)
#print(f"Best alpha for Lasso: {lasso_cv.best_params_['alpha']}")
#
#ridge_best = ridge_cv.best_estimator_
#y_pred_ridge = ridge_best.predict(X_test)
#
#lasso_best = lasso_cv.best_estimator_
#y_pred_lasso = lasso_best.predict(X_test)
#
#np.save('ridge_predictions.npy', y_pred_ridge)
#np.save('lasso_predictions.npy', y_pred_lasso)

print("Ridge predictions:", y_pred_ridge[:157])  # Print first 5 predictions
print("Lasso predictions:", y_pred_lasso[:157])