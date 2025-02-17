import pandas as pd
import numpy as np
import pickle
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Define file paths
train_path = r" https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
test_path = r"https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"


# Load the training dataset
df_train = pd.read_csv(train_path, parse_dates=['Timestamp'], index_col='Timestamp')

# Ensure the index has a consistent frequency
df_train = df_train.asfreq('h')

# Convert trips column to float64 to prevent dtype conflicts
df_train['trips'] = df_train['trips'].astype(float)

# Check for missing values and forward fill
df_train = df_train.ffill()

# Fit an Exponential Smoothing model
model = ExponentialSmoothing(df_train['trips'], trend='add', seasonal='add', seasonal_periods=24)
modelFit = model.fit()

# Save the model to disk for future predictions
with open("model.pkl", "wb") as f:
    pickle.dump(modelFit, f)

# Load the test dataset
df_test = pd.read_csv(test_path, parse_dates=['Timestamp'], index_col='Timestamp')

# Ensure test data frequency matches
df_test = df_test.asfreq('h')

# Make predictions for the next 744 hours (January of the following year)
pred = modelFit.forecast(steps=744)

# Convert predictions to a DataFrame and save them
pred = pd.DataFrame(pred, index=pd.date_range(start=df_test.index[-1] + pd.Timedelta(hours=1), periods=744, freq="h"), columns=['trips'])
pred.to_csv("predictions.csv")

print("Forecasting complete. Predictions saved to 'predictions.csv'.")
