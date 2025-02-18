import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.statespace.varmax import VARMAX
import pickle

# Load training data
train_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
train_data = pd.read_csv(train_url)

# Load test data
test_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"
test_data = pd.read_csv(test_url)

# Ensure correct timestamp column name
if 'Timestamp' in train_data.columns:
    train_data.rename(columns={'Timestamp': 'timestamp'}, inplace=True)

# Convert timestamp column to datetime format and set as index
train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])
train_data.set_index('timestamp', inplace=True)

# Ensure dataset follows an hourly frequency
train_data = train_data.asfreq('h')

# Select the dependent variable (number of taxi trips)
y_train = train_data['trips']

# === OPTION 1: Exponential Smoothing Model === #
model = ExponentialSmoothing(y_train, seasonal='add', seasonal_periods=24)
modelFit = model.fit()

# Save the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(modelFit, f)


# Ensure correct timestamp column name in test data
if 'Timestamp' in test_data.columns:
    test_data.rename(columns={'Timestamp': 'timestamp'}, inplace=True)

# Convert test timestamp column to datetime format and set as index
test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
test_data.set_index('timestamp', inplace=True)
test_data = test_data.asfreq('h')

# Forecast for 744 hours (January of next year)
pred = modelFit.forecast(steps=744)

# Save predictions
pred.to_csv("predictions.csv")

print("Model training and prediction completed successfully!")

import matplotlib.pyplot as plt

# Load predictions
pred = pd.read_csv("predictions.csv", index_col=0)
pred.index = pd.to_datetime(pred.index)