import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam as adam
from tensorflow.keras.utils import to_categorical

# Load data
X_train1 = np.load('./input/Xtrain1.npy')
Y_train1 = np.load('./input/Ytrain1.npy')

X_train1_extra = np.load('./input/Xtrain1_extra.npy')  

X_test1 = np.load('./input/Xtest1.npy')

# Normalize the data
X_train1 = X_train1.astype('float32') / 255.0
X_test1 = X_test1.astype('float32') / 255.0

# Reshape data for CNN
X_train1 = X_train1.reshape(-1, 48, 48, 1)

# Train-test split for validation
X_train_train, X_train_test, Y_train_train, Y_train_test = train_test_split(X_train1, Y_train1, test_size=0.2, random_state=42)

# Convert labels to categorical (for CNN)
Y_train_train_cat = to_categorical(Y_train_train, 2)
y_train_test_cat = to_categorical(Y_train_test, 2)

'''
CNN model
'''

# Initialize CNN model
model_CNN = Sequential()
model_CNN.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model_CNN.add(MaxPooling2D(pool_size=(2, 2)))
model_CNN.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model_CNN.add(MaxPooling2D(pool_size=(2, 2)))
model_CNN.add(Flatten())
model_CNN.add(Dense(64, activation='relu'))
model_CNN.add(Dense(2, activation='softmax'))

# Compile the model
model_CNN.compile(optimizer=adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
history_cnn = model_CNN.fit(X_train_train, Y_train_train_cat, epochs=40, batch_size=50, validation_data=(X_train_test, y_train_test_cat))

# Plot the loss curve
plt.plot(history_cnn.history['loss'], label='Train Loss')
plt.plot(history_cnn.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Predict on validation data
y_pred_cnn = np.argmax(model_CNN.predict(X_train_test), axis=1)

# Evaluate
f1_cnn = f1_score(Y_train_test, y_pred_cnn)
conf_matrix_cnn = confusion_matrix(Y_train_test, y_pred_cnn)

print(f'CNN F1 Score: {f1_cnn}')
print(f'CNN Confusion Matrix:\n {conf_matrix_cnn}')

'''
KNN model


# Flatten the data for KNN
X_train_flat = X_train_train.reshape(-1, 48*48)
X_val_flat = X_train_test.reshape(-1, 48*48)

# Initialize KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model
knn.fit(X_train_flat, Y_train_train)

# Make predictions
y_pred_knn = knn.predict(X_val_flat)

# Evaluate
f1_knn = f1_score(Y_train_test, y_pred_knn)
conf_matrix_knn = confusion_matrix(Y_train_test, y_pred_knn)

print(f'KNN F1 Score: {f1_knn}')
print(f'KNN Confusion Matrix:\n {conf_matrix_knn}')
'''