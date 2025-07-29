import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Path to the normalized data file
data_path = '/Users/home/Documents/github/cnn_classification/data/acc_vehicle_data_dof_5_5pc.csv'

# Load the data
data = pd.read_csv(data_path)

# Check data shape
print(f"Data shape: {data.shape}")

# Select 5 signals to plot
num_signals = 1
selected_signals = data.iloc[:num_signals, :]

# Plot each signal
plt.figure(figsize=(12, 6))
for i in range(num_signals):
    signal = selected_signals.iloc[i].values
    plt.plot(signal, label=f'Signal {i+1}')

# Labeling and legend
plt.show()