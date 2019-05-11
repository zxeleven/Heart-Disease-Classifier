import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

filename = "./heart.csv"

# Define seed and early stopping monitor
seed = 4
np.random.seed(seed)
early_stopping = EarlyStopping(patience=3)

# Read in dataset from heart.csv
dataset_pd = pd.read_csv(filename)

# Set variables for different sets of columns
category_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
true_value_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# One-hot encode categorical columns
for category in category_columns:
    one_hot = pd.get_dummies(dataset_pd[category], prefix=category)
    dataset_pd = pd.concat([dataset_pd, one_hot], axis=1)
    dataset_pd = dataset_pd.drop([category], axis=1)

# Normalize true_value columns
for true_value in true_value_columns:
    max_value = dataset_pd[true_value].max()
    min_value = dataset_pd[true_value].min()
    dataset_pd[true_value] = (dataset_pd[true_value] - min_value) / (max_value - min_value)

# Convert dataframe to numpy array
dataset = dataset_pd.values
num_cols = dataset.shape[1]

X = dataset[:,0:num_cols-2]
Y = dataset[:,num_cols-2:]

# Split the data into 70% for training and 30% for testing (15% for validation)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.15, random_state=seed)

# Create the model
model = Sequential()
model.add(Dense(30, input_dim=30, kernel_initializer='uniform', activation='relu'))
model.add(Dense(30, kernel_initializer='uniform', activation='relu'))
model.add(Dense(2, kernel_initializer='uniform', activation='softmax'))

# Compile and fit model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_split = 0.15, epochs=100, batch_size=200, verbose=2, callbacks = [early_stopping])

# Evaluate the model
scores = model.evaluate(X_test, Y_test)
print("Accuracy of the model after testing: %.2f%%" % (scores[1]*100))
