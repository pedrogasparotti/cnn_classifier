import numpy as np

# Path to the saved .npy file
file_path = "/Users/home/Documents/github/cnn_classification/data/processed/acc_vehicle_data_dof_5.npy"

# Load the saved matrices
accel_matrices = np.load(file_path)

# Check the shape of the loaded data
print(f"Loaded matrix shape: {accel_matrices.shape}")

# Sanity check: Print the first few elements of the matrix to inspect data
print("Sample data from the first row of the first matrix slice:")
print(accel_matrices[0, 0, :10])  # First row, first slice, first 10 elements

# If padding was applied, check the last row of the last slice
print("Sample data from the last row of the last matrix slice (to check padding):")
print(accel_matrices[-1, -1, -10:])  # Last row, last slice, last 10 elements