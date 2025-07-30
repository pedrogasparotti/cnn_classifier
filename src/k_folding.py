import numpy as np
import json
import os
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# --- Configuration ---
PROCESSED_DATA_DIR = os.path.join('data', 'processed')
NUM_FOLDS = 5
EPOCHS = 50
BATCH_SIZE = 32

def load_full_dataset(data_dir):
    """
    Loads and combines the train/test splits back into a single pool.
    The holdout set is NOT touched.
    """
    print(f"Loading and combining data from: {data_dir}")
    X_train = np.load(os.path.join(data_dir, 'X_train_sub.npy'))
    y_train_int = np.load(os.path.join(data_dir, 'y_train_int_sub.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test_sub.npy'))
    y_test_int = np.load(os.path.join(data_dir, 'y_test_int_sub.npy'))

    # Combine into a single dataset for k-folding
    X_pool = np.concatenate((X_train, X_test), axis=0)
    y_pool_int = np.concatenate((y_train_int, y_test_int), axis=0)

    with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
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
    Performs K-Fold Cross-Validation.
    """
    X, y, metadata = load_full_dataset(PROCESSED_DATA_DIR)
    
    input_shape = (X.shape[1], X.shape[2])
    num_classes = metadata['num_classes']
    
    # Use StratifiedKFold for classification to preserve class distribution
    kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_losses = []
    
    print(f"\n--- Starting {NUM_FOLDS}-Fold Cross-Validation ---")

    for fold, (train_indices, val_indices) in enumerate(kfold.split(X, y), 1):
        print(f"\n--- Fold {fold}/{NUM_FOLDS} ---")
        
        # 1. Split data for the current fold
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        # 2. Build a fresh, untainted model
        print("Building new model for this fold...")
        model = build_model(input_shape, num_classes)
        
        # Define EarlyStopping for this fold
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )

        # 3. Train the model
        print("Training model on this fold's data...")
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # 4. Evaluate and store results
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"Fold {fold} - Validation Accuracy: {accuracy * 100:.2f}%")
        print(f"Fold {fold} - Validation Loss: {loss:.4f}")
        fold_accuracies.append(accuracy)
        fold_losses.append(loss)
        
    # 5. Report final summary
    print("\n--- Cross-Validation Summary ---")
    print(f"Scores across {NUM_FOLDS} folds:")
    for i, acc in enumerate(fold_accuracies):
        print(f"  - Fold {i+1}: {acc*100:.2f}%")
        
    mean_accuracy = np.mean(fold_accuracies) * 100
    std_accuracy = np.std(fold_accuracies) * 100
    
    print(f"\nAverage Validation Accuracy: {mean_accuracy:.2f}%")
    print(f"Standard Deviation: {std_accuracy:.2f}%")
    print("-----------------------------------")

if __name__ == '__main__':
    run_cross_validation()