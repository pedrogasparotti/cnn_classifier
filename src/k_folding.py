import json
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# --- Configuration ---
# Directory where the preprocessed .npy files are located.
PROCESSED_DATA_DIR = os.path.join('data', 'processed')
# Directory where all output figures and models will be saved.
RESULTS_DIR = 'cross_val_results'
NUM_FOLDS = 5
EPOCHS = 45
BATCH_SIZE = 32

def plot_and_save_history(history, fold_num, output_dir):
    """
    Plots the training/validation accuracy and loss curves and saves them as PNG files.

    Args:
        history: The History object returned by model.fit().
        fold_num (int): The current fold number.
        output_dir (str): The directory to save the plots in.
    """
    print(f"Generating training history plots for Fold {fold_num}...")
    
    # --- Plot Accuracy ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Fold {fold_num}: Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # --- Plot Loss ---
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Fold {fold_num}: Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    # Save the combined figure
    plot_path = os.path.join(output_dir, f'fold_{fold_num}_history.png')
    plt.savefig(plot_path)
    plt.close() # Close the figure to free up memory
    print(f"Saved history plot to {plot_path}")


def plot_and_save_confusion_matrix(y_true, y_pred_classes, fold_num, class_names, output_dir):
    """
    Computes, plots, and saves the confusion matrix for a given fold.

    Args:
        y_true: The true integer labels.
        y_pred_classes: The predicted integer labels.
        fold_num (int): The current fold number.
        class_names (list): A list of strings representing the class names.
        output_dir (str): The directory to save the plot in.
    """
    print(f"Generating confusion matrix for Fold {fold_num}...")
    cm = confusion_matrix(y_true, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f'Fold {fold_num} - Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    
    # Save the figure
    cm_path = os.path.join(output_dir, f'fold_{fold_num}_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")


def plot_and_save_class_distribution(y_train, y_val, fold_num, class_names, output_dir):
    """
    Plots and saves the class distribution for the training and validation sets of a fold.

    Args:
        y_train: The training labels for the fold.
        y_val: The validation labels for the fold.
        fold_num (int): The current fold number.
        class_names (list): A list of strings representing the class names.
        output_dir (str): The directory to save the plot in.
    """
    print(f"Generating class distribution plot for Fold {fold_num}...")
    train_counts = Counter(y_train)
    val_counts = Counter(y_val)
    
    # Ensure all classes are represented, even if count is 0
    for i in range(len(class_names)):
        if i not in train_counts: train_counts[i] = 0
        if i not in val_counts: val_counts[i] = 0

    df_train = pd.DataFrame.from_dict(train_counts, orient='index').sort_index()
    df_val = pd.DataFrame.from_dict(val_counts, orient='index').sort_index()

    x = np.arange(len(class_names))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, df_train[0], width, label='Train')
    rects2 = ax.bar(x + width/2, df_val[0], width, label='Validation')

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Number of Samples')
    ax.set_title(f'Fold {fold_num}: Class Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    
    plot_path = os.path.join(output_dir, f'fold_{fold_num}_distribution.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved class distribution plot to {plot_path}")


def load_full_dataset(data_dir):
    """
    Loads and combines the train/test splits back into a single pool.
    The holdout set is NOT touched.
    """
    print(f"Loading and combining data from: {data_dir}")
    # This assumes your data files exist. If not, you should handle exceptions.
    try:
        X_train = np.load(os.path.join(data_dir, 'X_train_sub.npy'))
        y_train_int = np.load(os.path.join(data_dir, 'y_train_int_sub.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test_sub.npy'))
        y_test_int = np.load(os.path.join(data_dir, 'y_test_int_sub.npy'))
    except FileNotFoundError as e:
        print(f"Error: Data file not found. Make sure your processed data is in '{data_dir}'")
        print(e)
        # Exit if data is not available
        exit()

    # Combine into a single dataset for k-folding
    X_pool = np.concatenate((X_train, X_test), axis=0)
    y_pool_int = np.concatenate((y_train_int, y_test_int), axis=0)

    metadata_path = os.path.join(data_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    print(f"Data pool created. Total samples for K-Fold: {len(X_pool)}")
    return X_pool, y_pool_int, metadata


def build_model(input_shape, num_classes):
    """
    Builds the same robust 1D CNN model as before.
    Must be called inside the loop to get a fresh model each time.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=5, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def run_cross_validation():
    """
    Performs K-Fold Cross-Validation, saves plots and the best model.
    """
    # --- 1. Setup Directories ---
    # Create the main results directory and subdirectories for outputs
    plots_dir = os.path.join(RESULTS_DIR, 'plots')
    cm_dir = os.path.join(RESULTS_DIR, 'confusion_matrices')
    dist_dir = os.path.join(RESULTS_DIR, 'class_distributions')
    models_dir = os.path.join(RESULTS_DIR, 'models')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)
    os.makedirs(dist_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # --- 2. Load Data ---
    X, y, metadata = load_full_dataset(PROCESSED_DATA_DIR)
    
    input_shape = (X.shape[1], X.shape[2])
    num_classes = metadata['num_classes']
    # Get class names for plotting, fall back to integer strings if not available
    class_names = metadata.get('class_names', [str(i) for i in range(num_classes)])

    # --- 3. Initialize Cross-Validation ---
    kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_losses = []
    best_accuracy = 0.0
    
    print(f"\n--- Starting {NUM_FOLDS}-Fold Cross-Validation ---")

    for fold, (train_indices, val_indices) in enumerate(kfold.split(X, y), 1):
        print(f"\n--- Fold {fold}/{NUM_FOLDS} ---")
        
        # a. Split data for the current fold
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        # b. Print and plot class distribution for this fold
        print("Class distribution in this fold:")
        print(f"  Training set:   {sorted(Counter(y_train).items())}")
        print(f"  Validation set: {sorted(Counter(y_val).items())}")
        plot_and_save_class_distribution(y_train, y_val, fold, class_names, dist_dir)

        # c. Build a fresh, untainted model
        print("Building new model for this fold...")
        model = build_model(input_shape, num_classes)
        
        # d. Define callbacks for this fold
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )

        # e. Train the model
        print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples...")
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # f. Evaluate and store results
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"Fold {fold} - Validation Accuracy: {accuracy * 100:.2f}%")
        print(f"Fold {fold} - Validation Loss: {loss:.4f}")
        fold_accuracies.append(accuracy)
        fold_losses.append(loss)
        
        # g. Save training history plots
        plot_and_save_history(history, fold, plots_dir)
        
        # h. Generate predictions and save confusion matrix
        y_pred_probs = model.predict(X_val)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        plot_and_save_confusion_matrix(y_val, y_pred_classes, fold, class_names, cm_dir)
        
        # i. Check if this is the best model so far and save it
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"New best model found! Saving model from Fold {fold}...")
            model.save(os.path.join(models_dir, 'best_model.keras'))

    # --- 4. Report Final Summary ---
    print("\n--- Cross-Validation Summary ---")
    print(f"Scores across {NUM_FOLDS} folds:")
    for i, acc in enumerate(fold_accuracies):
        print(f"  - Fold {i+1}: Accuracy={acc*100:.2f}%, Loss={fold_losses[i]:.4f}")
        
    mean_accuracy = np.mean(fold_accuracies) * 100
    std_accuracy = np.std(fold_accuracies) * 100
    
    print(f"\nAverage Validation Accuracy: {mean_accuracy:.2f}% (+/- {std_accuracy:.2f}%)")
    print("-----------------------------------")
    print(f"All plots and matrices saved in '{RESULTS_DIR}' directory.")
    print(f"The best performing model was saved to: {os.path.join(models_dir, 'best_model.keras')}")
    print("-----------------------------------")


if __name__ == '__main__':
    # Ensure the script is being run from a directory where 'data/processed' is accessible
    # or provide an absolute path.
    if not os.path.exists(PROCESSED_DATA_DIR):
        print(f"ERROR: The data directory '{PROCESSED_DATA_DIR}' does not exist.")
        print("Please ensure your preprocessed data is in the correct location or update the PROCESSED_DATA_DIR variable.")
    else:
        # We need pandas for the new plotting function
        try:
            import pandas as pd
        except ImportError:
            print("ERROR: pandas is required for plotting class distributions.")
            print("Please install it using: pip install pandas")
            exit()
        run_cross_validation()