import pandas as pd
import numpy as np
import pickle
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Define file paths
train_path = r"https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
test_path = r" https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"

# Check if files exist
print("Checking if training file exists:", os.path.exists(train_path))
print("Checking if test file exists:", os.path.exists(test_path))

# Load the training dataset
df_train = pd.read_csv(train_path, parse_dates=['Timestamp'], index_col='Timestamp')

# Check for missing values
df_train.dropna(inplace=True)

# Select dependent variable (number of trips per hour)
y_train = df_train['trips']

# Define the Exponential Smoothing model
model = ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=24)  # 24-hour seasonality

# Fit the model
modelFit = model.fit()

# Save the model to disk for future predictions
with open("model.pkl", "wb") as f:
    pickle.dump(modelFit, f)

# Load the test dataset
df_test = pd.read_csv(test_path, parse_dates=['Timestamp'], index_col='Timestamp')

# Ensure test data is in the right format
df_test.dropna(inplace=True)  # Drop missing values if any

# Make predictions for the next 744 hours (January of the following year)
pred = modelFit.forecast(steps=744)

# Convert predictions to a NumPy array (to match assignment format)
pred = np.array(pred)

# Save predictions
pd.Series(pred, index=pd.date_range(start=df_test.index[-1], periods=744, freq="H")).to_csv("predictions.csv")

print("Forecasting complete. Predictions saved to 'predictions.csv'.")
