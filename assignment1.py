import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle
import matplotlib.pyplot as plt

# Define dataset URLs
train_data_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
test_data_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"

# Load training dataset
train_df = pd.read_csv(train_data_url)
train_df.rename(columns={'Timestamp': 'timestamp'}, inplace=True)
train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
train_df.set_index('timestamp', inplace=True)
train_df = train_df.asfreq('h')

# Define the target variable
target_series = train_df['trips']

# Train an Exponential Smoothing Model
model = ExponentialSmoothing(target_series, seasonal='add', seasonal_periods=24).fit()

# Save the trained model
with open("taxi_trip_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Load test dataset
test_df = pd.read_csv(test_data_url)
test_df.rename(columns={'Timestamp': 'timestamp'}, inplace=True)
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
test_df.set_index('timestamp', inplace=True)
test_df = test_df.asfreq('h')

# Generate predictions for the next 744 hours (one month)
predictions = model.forecast(steps=744)

# Save predictions to a CSV file
predictions.to_csv("predicted_taxi_trips.csv")
print("Model training and forecasting completed successfully!")

# Load and visualize predictions
forecasted_df = pd.read_csv("predicted_taxi_trips.csv", index_col=0)
forecasted_df.index = pd.to_datetime(forecasted_df.index)

plt.figure(figsize=(12, 6))
plt.plot(forecasted_df, label="Predicted Taxi Trips", color='blue')
plt.xlabel("Time")
plt.ylabel("Number of Trips")
plt.title("Predicted Taxi Trips for January")
plt.legend()
plt.show()
