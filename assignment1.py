import pandas as pd
import numpy as np
import pickle
import os
from pygam import LinearGAM, s

# Define file paths
train_path = r"https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
test_path = r"https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"

'''# Check if files exist
if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError("Training or test dataset is missing!")'''

# Load the training dataset
df_train = pd.read_csv(train_path, parse_dates=['Timestamp'], index_col='Timestamp')

# Convert trips column to float64
df_train['trips'] = df_train['trips'].astype(float)

# Handle missing values
df_train = df_train.ffill()

# Normalize time index (improves GAM performance)
df_train['time_index'] = (df_train.index - df_train.index.min()).total_seconds() / 3600

# Convert time_index to 2D array
X_train = df_train[['time_index']].values  # Ensure correct shape

# Fit a Generalized Additive Model (GAM)
model = LinearGAM(s(0)).fit(X_train, df_train['trips'])

# Save the model to disk for future predictions
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load the test dataset
df_test = pd.read_csv(test_path, parse_dates=['Timestamp'], index_col='Timestamp')

# Generate future time indexes
future_time_index = np.arange(df_train['time_index'].max() + 1, df_train['time_index'].max() + 745).reshape(-1, 1)

# Make predictions
pred = model.predict(future_time_index)

# Convert predictions to a DataFrame and save them
pred = pd.DataFrame(pred, index=pd.date_range(start=df_test.index[-1] + pd.Timedelta(hours=1), periods=744, freq="h"), columns=['trips'])
pred.to_csv("predictions.csv", index=False)

print("Forecasting complete. Predictions saved to 'predictions.csv'.")
