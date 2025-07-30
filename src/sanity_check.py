# Save this as check_data.py and run it
import numpy as np
import os

PROCESSED_DATA_DIR = os.path.join('data', 'processed')

def check_for_bad_values(file_path):
    """Checks a .npy file for NaN or Inf values."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    data = np.load(file_path)
    nan_count = np.isnan(data).sum()
    inf_count = np.isinf(data).sum()
    
    if nan_count > 0 or inf_count > 0:
        print(f"CORRUPT DATA FOUND IN: {os.path.basename(file_path)}")
        print(f"   - NaN values: {nan_count}")
        print(f"   - Infinity values: {inf_count}")
    else:
        print(f"Data is clean in: {os.path.basename(file_path)}")

print("--- Checking Preprocessed Data Files ---")
check_for_bad_values(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
check_for_bad_values(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
check_for_bad_values(os.path.join(PROCESSED_DATA_DIR, 'X_holdout.npy'))