import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import json

def load_and_label_datasets():
    """
    Load three datasets and assign labels.
    
    Returns:
        combined_df: DataFrame with all signals and process_combined_signals
    """
    
    # Dataset paths and labels into tuples
    datasets = [
        ('/Users/home/Documents/github/cnn_classification/data/acc_vehicle_data_dof_5.csv', 0, 'y0'), #healthy
        ('/Users/home/Documents/github/cnn_classification/data/acc_vehicle_data_dof_5_5pc.csv', 1, 'y1'), #5pc damage
        ('/Users/home/Documents/github/cnn_classification/data/acc_vehicle_data_dof_5_10pc.csv', 2, 'y2') #10pc damage
    ]
    
    # store all data into an array
    combined_data = []
    
    for filepath, label, label_name in datasets:
        print(f"Loading {label_name}: {filepath}")
        
        df = pd.read_csv(filepath)

        # Add label columns
        df['label'] = label
        df['label_name'] = label_name
        
        combined_data.append(df)
        print(f"  Loaded {len(df)} samples")
    
    # Combine all datasets
    combined_df = pd.concat(combined_data, ignore_index=True).fillna(0)
    
    print(f"\nCombined dataset: {len(combined_df)} total samples")
    print("Label distribution:")
    print(combined_df['label_name'].value_counts())
    
    # combined dataset UNION ALL data into a single dataset
    return combined_df

def process_combined_signals(combined_df, target_length=2000, remove_first_n=500):
    """
    Process combined signal dataset for CNN classification.
    This version correctly scales each signal individually and is robust to errors.
    """
    labels = combined_df['label'].values
    label_names = combined_df['label_name'].unique().tolist()
    
    # Convert to NumPy once, drop labels. This is much faster than iterrows().
    signal_data = combined_df.drop(columns=['label', 'label_name']).to_numpy(dtype=np.float32)

    # --- Vectorized Processing ---
    # 1. Remove initial samples from all signals at once
    processed_signals = signal_data[:, remove_first_n:]
    
    # 2. Truncate or pad all signals
    current_length = processed_signals.shape[1]
    if current_length > target_length:
        processed_signals = processed_signals[:, :target_length]
    elif current_length < target_length:
        padding_width = target_length - current_length
        # Pad on the right side of each signal
        processed_signals = np.pad(processed_signals, ((0, 0), (0, padding_width)), mode='constant', constant_values=0)

    # --- Per-Signal Scaling ---
    # Iterate through each signal (row) and scale it individually.
    for i in range(processed_signals.shape[0]):
        signal = processed_signals[i]
        std_dev = np.std(signal)
        # Only scale if the signal has variance. Otherwise, it's a flat line.
        if std_dev > 1e-6: # Use a small epsilon for floating point safety
            mean = np.mean(signal)
            processed_signals[i] = (signal - mean) / std_dev
        else:
            # If standard deviation is zero, the signal is flat. Just center it.
            processed_signals[i] = signal - np.mean(signal)
            
    # Final check to guarantee no bad values are produced
    if np.isnan(processed_signals).any() or np.isinf(processed_signals).any():
        raise ValueError("Data contains NaN/Inf values AFTER processing. Check raw data.")

    # Add channel dimension for CNN
    processed_signals = processed_signals.reshape(processed_signals.shape[0], processed_signals.shape[1], 1)
    
    # NOTE: The original `scaler` object is removed because scaling is now custom.
    # The function no longer returns a scaler.
    return processed_signals, labels, label_names

def save_processed_dataset(signals, labels, label_names, output_dir='/Users/home/Documents/github/cnn_classification/data/processed', holdout_samples_per_class=100, samples_per_class_subsample = 100, seed=42):
    """
    Correctly isolates a holdout dataset and saves distinct train/test/holdout splits.
    This version fixes critical file naming bugs and clarifies the data splitting logic.
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    # --- 1. Isolate the Holdout Set ---
    # This set is completely separated and not used for training or standard validation.
    holdout_indices = []
    for class_label in unique_labels:
        # Find all indices for the current class
        class_indices = np.where(labels == class_label)[0]
        
        # Check if there are enough samples to draw from
        if len(class_indices) < holdout_samples_per_class:
            raise ValueError(
                f"Cannot sample {holdout_samples_per_class} for holdout set from class {class_label}. "
                f"Only {len(class_indices)} samples available."
            )
            
        # Randomly choose indices for the holdout set without replacement
        sampled_indices = np.random.choice(class_indices, size=holdout_samples_per_class, replace=False)
        holdout_indices.extend(sampled_indices)

    holdout_indices = np.array(holdout_indices)
    
    # Create the holdout dataset
    X_holdout = signals[holdout_indices]
    y_holdout_int = labels[holdout_indices]

    # --- 2. Create the Training/Testing Pool ---
    # The remaining data after the holdout set has been removed.
    X_pool = np.delete(signals, holdout_indices, axis=0)
    y_pool_int = np.delete(labels, holdout_indices, axis=0)

    # Get the unique class labels from the pool.
    unique_labels_pool = np.unique(y_pool_int)

    # This list will store the indices we select for our subsample.
    subsample_indices = []

    # Loop through each class to perform stratified sampling.
    for class_label in unique_labels_pool:
        # Find all indices in the pool that belong to the current class.
        class_indices_pool = np.where(y_pool_int == class_label)[0]
        
        # Check if there are enough samples to draw from.
        if len(class_indices_pool) < samples_per_class_subsample:
            raise ValueError(
                f"Cannot sample {samples_per_class_subsample} for subsample from class {class_label}. "
                f"Only {len(class_indices_pool)} samples available in the pool."
            )
            
        # Randomly choose 100 indices for this class without replacement.
        sampled_indices = np.random.choice(
            class_indices_pool, 
            size=samples_per_class_subsample, 
            replace=False
        )
        subsample_indices.extend(sampled_indices)

    # Convert the list of indices to a NumPy array for efficient slicing.
    subsample_indices = np.array(subsample_indices)
    np.random.shuffle(subsample_indices) # Shuffle to mix the order of classes

    # Create the final subsampled dataset.
    X_subsample = X_pool[subsample_indices]
    y_subsample_int = y_pool_int[subsample_indices]

    # --- 3. Split the Pool into Training and Testing Sets ---
    # Stratify to ensure class distribution is similar in train and test sets.
    X_train, X_test, y_train_int, y_test_int = train_test_split(
        X_subsample, y_subsample_int, test_size=0.2, random_state=seed, stratify=y_subsample_int
    )

    # --- 4. One-Hot Encode All Label Sets ---
    y_train = to_categorical(y_train_int, num_classes=num_classes)
    y_test = to_categorical(y_test_int, num_classes=num_classes)
    y_holdout = to_categorical(y_holdout_int, num_classes=num_classes)

    # --- 5. Save All Datasets with Clear, Correct Names ---
    # Training set
    np.save(os.path.join(output_dir, 'X_train_sub.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train_sub.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_train_int_sub.npy'), y_train_int)

    # Testing set
    np.save(os.path.join(output_dir, 'X_test_sub.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test_sub.npy'), y_test)
    np.save(os.path.join(output_dir, 'y_test_int_sub.npy'), y_test_int)

    # Holdout set (for double blind validation)
    np.save(os.path.join(output_dir, 'X_holdout.npy'), X_holdout)
    np.save(os.path.join(output_dir, 'y_holdout.npy'), y_holdout)
    np.save(os.path.join(output_dir, 'y_holdout_int.npy'), y_holdout_int)
    
    # --- 6. Save Accurate Metadata ---
    metadata = {
        'num_classes': num_classes,
        'class_names': list(np.unique(label_names)),
        'signal_length': signals.shape[1],
        'train_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'holdout_samples': X_holdout.shape[0],
        'total_samples_in_sets': X_train.shape[0] + X_test.shape[0] + X_holdout.shape[0],
        'label_encoding': 'one-hot and integer labels saved',
        'seed': seed,
        'obs': "healthy-> y0; 5pc_damage -> y1; 10pc_damage -> y2"
    }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    print("Datasets train/test/holdout splits saved.")
    return metadata


if __name__ == "__main__":
    print("=== Loading and combining datasets ===")
    combined_df = load_and_label_datasets()
    
    print("\n=== Processing signals ===")
    signals, labels, label_names= process_combined_signals(
        combined_df, target_length=2000, remove_first_n=500
    )
    
    print(f"Final processed shape: {signals.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Signal statistics - Mean: {signals.mean():.3f}, Std: {signals.std():.3f}")
    
    print("\n=== Saving processed dataset ===")
    metadata = save_processed_dataset(signals, labels, label_names)
    
    print(f"Dataset saved to: processed_data/")
    print(f"Classes: {metadata['num_classes']}")
    print(f"Total samples: {metadata['total_samples_in_sets']}")
    print(f"Train/Test split: {metadata['train_samples']}/{metadata['test_samples']}")
    print(f"Label encoding: {metadata['label_encoding']}")