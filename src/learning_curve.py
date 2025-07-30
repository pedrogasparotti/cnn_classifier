import numpy as np
import json
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

from cnn_model import build_model

# --- Configuration ---
PROCESSED_DATA_DIR = os.path.join('data', 'processed')
MODEL_OUTPUT_DIR = 'models'

# 1. Define the number of samples PER CLASS to test
SAMPLE_SIZES_PER_CLASS = [5, 10, 20, 40, 60, 80, 100]

# 2. Define how many random trials to run for each sample size
NUM_TRIALS_PER_SIZE = 5

# 3. Training parameters
EPOCHS = 30
BATCH_SIZE = 16

def load_data_pools(data_dir):
    """Loads the full training pool and the fixed test set."""
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train_int = np.load(os.path.join(data_dir, 'y_train_int.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test_int = np.load(os.path.join(data_dir, 'y_test_int.npy'))
    
    # For this experiment, we only need the training pool and the fixed test set
    return X_train, y_train_int, X_test, y_test_int

def build_model(input_shape, num_classes):
    """Builds the same CNN model. Must be called to get a fresh model."""
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def run_learning_curve_experiment():
    """
    Trains models on progressively larger subsets of data and plots the learning curve.
    """
    X_train_full, y_train_full, X_test, y_test = load_data_pools(PROCESSED_DATA_DIR)
    
    num_classes = len(np.unique(y_train_full))
    input_shape = (X_train_full.shape[1], X_train_full.shape[2])
    
    # Store results: {sample_size: [acc_trial_1, acc_trial_2, ...]}
    results = {size: [] for size in SAMPLE_SIZES_PER_CLASS}

    for size in SAMPLE_SIZES_PER_CLASS:
        print(f"\n--- Testing with {size} samples per class ---")
        
        for trial in range(NUM_TRIALS_PER_SIZE):
            print(f"  Trial {trial + 1}/{NUM_TRIALS_PER_SIZE}...")
            
            # Create a balanced subsample of the training data
            indices_to_sample = []
            for i in range(num_classes):
                class_indices = np.where(y_train_full == i)[0]
                # Ensure we don't try to sample more than available
                num_to_sample = min(size, len(class_indices))
                sampled_indices = np.random.choice(class_indices, num_to_sample, replace=False)
                indices_to_sample.extend(sampled_indices)
            
            X_sub, y_sub = X_train_full[indices_to_sample], y_train_full[indices_to_sample]

            # Build a new model for each trial
            model = build_model(input_shape, num_classes)
            
            model.fit(X_sub, y_sub, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            
            # Evaluate on the FULL, UNCHANGING test set
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            results[size].append(accuracy)

    # --- Process and Plot Results ---
    mean_accuracies = [np.mean(results[size]) for size in SAMPLE_SIZES_PER_CLASS]
    std_accuracies = [np.std(results[size]) for size in SAMPLE_SIZES_PER_CLASS]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x_axis = np.array(SAMPLE_SIZES_PER_CLASS) * num_classes
    mean_accuracies = np.array(mean_accuracies)
    std_accuracies = np.array(std_accuracies)

    ax.plot(x_axis, mean_accuracies, 'o-', color='b', label='Mean Accuracy')
    ax.fill_between(x_axis, mean_accuracies - std_accuracies, mean_accuracies + std_accuracies,
                    color='b', alpha=0.2, label='Standard Deviation')

    ax.set_title('Model Learning Curve', fontsize=18)
    ax.set_xlabel('Total Number of Training Samples', fontsize=14)
    ax.set_ylabel('Accuracy on Full Test Set', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    
    # Save the plot
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(MODEL_OUTPUT_DIR, 'learning_curve.png')
    plt.savefig(plot_path)
    print(f"\nLearning curve plot saved to: {plot_path}")
    plt.show()

if __name__ == '__main__':
    run_learning_curve_experiment()
    print("\nLearning curve experiment complete.")