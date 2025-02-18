import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
import pickle
import matplotlib.pyplot as plt

# URLs for training and testing datasets
train_data_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
test_data_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"

# Load and preprocess training data
train_df = pd.read_csv(train_data_url)
if 'Timestamp' in train_df.columns:
    train_df.rename(columns={'Timestamp': 'timestamp'}, inplace=True)

train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
train_df.set_index('timestamp', inplace=True)
train_df = train_df.asfreq('h')

# Define target variable
target_series = train_df['trips']

# === Exponential Smoothing Model === #
es_model = ExponentialSmoothing(target_series, seasonal='add', seasonal_periods=24)
es_fitted = es_model.fit()

# Save trained model to file
with open("trained_model.pkl", "wb") as model_file:
    pickle.dump(es_fitted, model_file)

# Load and preprocess testing data
test_df = pd.read_csv(test_data_url)
if 'Timestamp' in test_df.columns:
    test_df.rename(columns={'Timestamp': 'timestamp'}, inplace=True)

test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
test_df.set_index('timestamp', inplace=True)
test_df = test_df.asfreq('h')

# Generate forecast for 744 time steps (January of the next year)
predictions = es_fitted.forecast(steps=744)

# Save predictions
predictions.to_csv("forecasted_trips.csv")
print("Model training and forecasting completed successfully!")

# Load and visualize predictions
forecasted_df = pd.read_csv("forecasted_trips.csv", index_col=0)
forecasted_df.index = pd.to_datetime(forecasted_df.index)

plt.figure(figsize=(12, 6))
plt.plot(forecasted_df, label="Predicted Taxi Trips", color='blue')
plt.xlabel("Time")
plt.ylabel("Number of Trips")
plt.title("Taxi Trip Forecast for January")
plt.legend()
plt.show()
