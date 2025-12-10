import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    Dropout,
    Dense,
    GlobalAveragePooling1D,
    Input,
    Activation,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import keras_tuner as kt
import numpy as np
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration ---
PROCESSED_DATA_DIR = os.path.join('data', 'processed')
MODEL_OUTPUT_DIR = 'models'
TUNER_DIR = 'keras_tuner_dir_bayesian' # New directory for the new tuner
PROJECT_NAME = '1d_cnn_bayesian_tuning'
BEST_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, 'best_bayesian_tuned_cnn_model.keras')

# --- Data Loading ---
def load_data(data_dir):
    """
    Loads the preprocessed training and testing datasets.
    """
    print(f"Loading data from: {data_dir}")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}. Run a preprocessing script first.")

    X_train = np.load(os.path.join(data_dir, 'X_train_sub.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train_sub.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test_sub.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test_sub.npy'))
    
    with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
        
    print("Data loaded successfully.")
    return X_train, y_train, X_test, y_test, metadata

# --- HyperModel Definition ---
class CNNHyperModel(kt.HyperModel):
    """
    KerasTuner HyperModel class for our 1D CNN.
    This structure is compatible with any KerasTuner search algorithm.
    """
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        """
        Builds a 1D CNN model for hyperparameter tuning.
        """
        inputs = Input(shape=self.input_shape)
        x = inputs

        # Tune the number of feature extraction blocks
        for i in range(hp.Int("num_blocks", min_value=1, max_value=3, step=1)):
            x = Conv1D(
                filters=hp.Int(f"filters_{i}", min_value=16, max_value=64, step=32),
                kernel_size=5,
                padding="same",
            )(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = MaxPooling1D(pool_size=2)(x)
            # FIX: Changed max_value from 1.0 to 0.9 to be in the valid range [0, 1)
            x = Dropout(hp.Float(f"dropout_conv_{i}", min_value=0.4, max_value=0.9, step=0.1))(x)

        x = GlobalAveragePooling1D()(x)

        # --- Classifier Head ---
        x = Dense(
            units=hp.Int("dense_units", min_value=64, max_value=512, step=64),
            activation="relu",
        )(x)
        # FIX: Changed max_value from 1.0 to 0.9 to be in the valid range [0, 1)
        x = Dropout(hp.Float("dropout_dense", min_value=0.6, max_value=0.9, step=0.1))(x)

        outputs = Dense(self.num_classes, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outputs)

        # --- Compilation ---
        learning_rate = hp.Float("lr", min_value=1e-6, max_value=1e-4, sampling="log")
        optimizer = Adam(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

# --- Model Evaluation ---
def evaluate_model(model_path, X_test, y_test, metadata):
    """
    Evaluates the BEST saved model on the unseen test set.
    """
    print("\n--- Evaluating Best Tuned Model on Test Set ---")
    best_model = load_model(model_path)
    loss, accuracy = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Set Loss: {loss:.4f}")
    
    y_pred_probs = best_model.predict(X_test)
    y_pred_int = np.argmax(y_pred_probs, axis=1)
    y_test_int = np.argmax(y_test, axis=1)
    class_names = metadata['class_names']
    
    print("\nClassification Report:")
    print(classification_report(y_test_int, y_pred_int, target_names=class_names))

    cm = confusion_matrix(y_test_int, y_pred_int)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix on Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plot_path = os.path.join(MODEL_OUTPUT_DIR, 'bayesian_tuned_confusion_matrix.png')
    plt.savefig(plot_path)
    print(f"\nConfusion matrix plot saved to {plot_path}")
    plt.show()

# --- Main Execution ---
if __name__ == '__main__':
    # 1. Load Data
    # Note: Ensure your data loading paths are correct or modify as needed.
    # This example assumes the script is run from the root of your project.
    try:
        X_train, y_train, X_test, y_test, metadata = load_data(PROCESSED_DATA_DIR)
    except FileNotFoundError as e:
        print(e)
        print("Please ensure your preprocessed data exists or update the PROCESSED_DATA_DIR path.")
        exit() # Exit if data is not found

    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = y_train.shape[1]

    # 2. Set up HyperModel and Bayesian Optimization Tuner
    hypermodel = CNNHyperModel(input_shape=input_shape, num_classes=num_classes)
    
    tuner = kt.BayesianOptimization(
        hypermodel,
        objective="val_accuracy",
        max_trials=50,
        directory=TUNER_DIR,
        project_name=PROJECT_NAME,
        overwrite=True # Set to True to start a new search
    )

    stop_early = EarlyStopping(monitor="val_loss", patience=5)

    # 3. Start the Hyperparameter Search
    print("\n--- Starting Bayesian Hyperparameter Search ---")
    tuner.search(
        X_train,
        y_train,
        epochs=50, # Max epochs for each trial
        validation_data=(X_test, y_test),
        callbacks=[stop_early]
    )
    print("\n--- Hyperparameter Search Finished ---")

    # 4. Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\n--- Optimal Hyperparameters Found ---")
    for hp_name, hp_value in best_hps.values.items():
        print(f"{hp_name}: {hp_value}")
    
    # 5. Build and Train the Final Model with Best Hyperparameters
    print("\n--- Training Final Model with Best Hyperparameters ---")
    final_model = tuner.hypermodel.build(best_hps)
    
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    final_checkpoint = ModelCheckpoint(BEST_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    final_early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    history = final_model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32, 
        validation_data=(X_test, y_test),
        callbacks=[final_checkpoint, final_early_stopping]
    )
    print("--- Final Model Training Finished ---")
    
    # 6. Evaluate the Final Model
    evaluate_model(BEST_MODEL_PATH, X_test, y_test, metadata)
    
    print("\n--- Workflow Complete ---")
    print(f"The best tuned model is saved at: {BEST_MODEL_PATH}")