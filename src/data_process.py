import numpy as np
import pandas as pd
import os

def load_csv_data(file_path):
    """
    Loads data from a CSV file.
    Parameters:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    try:
        dados = pd.read_csv(file_path)
        dados_cut = dados.iloc[:, 550:]  # Keep all rows, cut columns from 550
        shape = dados_cut.shape
        print(f"Successfully loaded data from {file_path}, shape of data is {shape}")
        return dados_cut
    except FileNotFoundError as e:
        print(f"Error loading CSV file: {e}")
        return None

def normalize_matrices(matrices):
    """
    Normalizes the matrix data to the range [0, 1].
    Parameters:
        matrices (np.ndarray): 3D or 4D matrix data to normalize.
    Returns:
        np.ndarray: Normalized matrices within the range [0, 1].
    """
    min_val = matrices.min()
    max_val = matrices.max()
    normalized_matrices = (matrices - min_val) / (max_val - min_val)
    return normalized_matrices

def transform_to_matrices(dados, target_shape, normalize=True):
    """
    Transforms the data from a DataFrame to a 3D matrix format and normalizes if specified.
    Parameters:
        dados (pd.DataFrame): DataFrame containing the data.
        target_shape (tuple): Target shape for the square matrices (height, width).
        normalize (bool): Whether to normalize the data to the range [0, 1].
    Returns:
        np.ndarray: 3D matrix of reshaped and optionally normalized data.
    """

    accel_data = dados.to_numpy()
    num_rows = accel_data.shape[0] 
    height, width = target_shape
    
    elements_per_matrix = height * width
    
    if accel_data.shape[1] < elements_per_matrix:
        print(f"Warning: Not enough columns ({accel_data.shape[1]}) to create {height}x{width} matrices. Padding with zeros.")
        padding_needed = elements_per_matrix - accel_data.shape[1]
        accel_data = np.pad(accel_data, ((0, 0), (0, padding_needed)), mode='constant')
    elif accel_data.shape[1] > elements_per_matrix:
        print(f"Warning: Too many columns ({accel_data.shape[1]}). Trimming excess columns.")
        accel_data = accel_data[:, :elements_per_matrix]
    
    accel_matrices = accel_data.reshape(num_rows, height, width)
    
    if normalize:
        accel_matrices = normalize_matrices(accel_matrices)
    
    print(f"Final matrix shape: {accel_matrices.shape}")
    return accel_matrices

def save_matrices(matrices, output_path, file_name="acc_vehicle_data_dof_5.npy"):
    """
    Saves the 3D matrix data to a .npy file.
    Parameters:
        matrices (np.ndarray): 3D matrix data to save.
        output_path (str): Directory to save the .npy file.
        file_name (str): Name of the .npy file.
    """
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, file_name)
    np.save(file_path, matrices)
    print(f"Matrix data saved to {file_path}")

file_path = "/Users/home/Documents/github/cnn_classification/data/acc_vehicle_data_dof_5.csv"
output_path = "/Users/home/Documents/github/cnn_classification/data/processed"

dados = load_csv_data(file_path)
if dados is not None:

    target_shape = (50, 50) 
    accel_matrices = transform_to_matrices(dados, target_shape)
    save_matrices(accel_matrices, output_path)