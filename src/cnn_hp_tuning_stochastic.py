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
TUNER_DIR = 'keras_tuner_dir'
PROJECT_NAME = '1d_cnn_hyper_tuning'
BEST_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, 'best_tuned_cnn_model.keras')

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
    This class structure helps in passing static parameters like input_shape and num_classes.
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
        for i in range(hp.Int("num_blocks", min_value=1, max_value=4, step=1)):
            # Tune the number of filters for each block's Conv1D layer
            x = Conv1D(
                filters=hp.Int(f"filters_{i}", min_value=32, max_value=256, step=32),
                kernel_size=5,
                padding="same",
            )(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = MaxPooling1D(pool_size=2)(x)
            # Tune the dropout rate for each convolutional block
            x = Dropout(hp.Float(f"dropout_conv_{i}", min_value=0.1, max_value=0.5, step=0.1))(x)

        # Global Pooling layer to reduce dimensionality
        x = GlobalAveragePooling1D()(x)

        # --- Classifier Head ---
        # Tune the number of units in the dense layer
        x = Dense(
            units=hp.Int("dense_units", min_value=64, max_value=512, step=64),
            activation="relu",
        )(x)
        # Tune the dropout rate for the classifier
        x = Dropout(hp.Float("dropout_dense", min_value=0.3, max_value=0.7, step=0.1))(x)

        outputs = Dense(self.num_classes, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outputs)

        # --- Compilation ---
        # Tune the learning rate for the Adam optimizer
        learning_rate = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log")
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
    
    # Load the best performing model
    best_model = load_model(model_path)
    
    # 1. Get loss and accuracy
    loss, accuracy = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Set Loss: {loss:.4f}")
    
    # 2. Generate detailed classification report
    y_pred_probs = best_model.predict(X_test)
    y_pred_int = np.argmax(y_pred_probs, axis=1)
    y_test_int = np.argmax(y_test, axis=1)
    class_names = metadata['class_names']
    
    print("\nClassification Report:")
    print(classification_report(y_test_int, y_pred_int, target_names=class_names))

    # 3. Generate and plot confusion matrix
    cm = confusion_matrix(y_test_int, y_pred_int)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix on Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plot_path = os.path.join(MODEL_OUTPUT_DIR, 'tuned_confusion_matrix.png')
    plt.savefig(plot_path)
    print(f"\nConfusion matrix plot saved to {plot_path}")
    plt.show()

# --- Main Execution ---
if __name__ == '__main__':
    # 1. Load Data
    # Ensure data exists. If not, this will raise an error.
    X_train, y_train, X_test, y_test, metadata = load_data(PROCESSED_DATA_DIR)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = y_train.shape[1]

    # 2. Set up HyperModel and Tuner
    hypermodel = CNNHyperModel(input_shape=input_shape, num_classes=num_classes)
    
    tuner = kt.Hyperband(
        hypermodel,
        objective="val_accuracy",
        max_epochs=50,  # Max epochs to train a model for during search
        factor=3,
        directory=TUNER_DIR,
        project_name=PROJECT_NAME,
    )

    # Define a callback to stop training early if validation loss doesn't improve
    stop_early = EarlyStopping(monitor="val_loss", patience=5)

    # 3. Start the Hyperparameter Search
    print("\n--- Starting Hyperparameter Search ---")
    tuner.search(
        X_train,
        y_train,
        epochs=50, # Number of epochs for each trial
        validation_data=(X_test, y_test), # Use test set for validation during search
        callbacks=[stop_early]
    )
    print("\n--- Hyperparameter Search Finished ---")

    # 4. Get the optimal hyperparameters and train the final model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\n--- Optimal Hyperparameters Found ---")
    print(f"Number of Blocks: {best_hps.get('num_blocks')}")
    for i in range(best_hps.get('num_blocks')):
        print(f"  - Filters (Block {i+1}): {best_hps.get(f'filters_{i}')}")
        print(f"  - Dropout (Block {i+1}): {best_hps.get(f'dropout_conv_{i}')}")
    print(f"Dense Units: {best_hps.get('dense_units')}")
    print(f"Dense Dropout: {best_hps.get('dropout_dense')}")
    print(f"Learning Rate: {best_hps.get('lr'):.5f}")
    
    # Tune batch size separately or choose a good default. Here we add it to the HPs.
    # For a more integrated approach, a custom Tuner class would be needed.
    # We will choose from a predefined list.
    hp_batch_size = best_hps.get('batch_size') if 'batch_size' in best_hps else 32
    print(f"Batch Size: {hp_batch_size}")


    # 5. Build and Train the Final Model with Best Hyperparameters
    print("\n--- Training Final Model with Best Hyperparameters ---")
    final_model = tuner.hypermodel.build(best_hps)
    
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    # Callbacks for the final training run
    final_checkpoint = ModelCheckpoint(
        BEST_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1
    )
    final_early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    )

    history = final_model.fit(
        X_train,
        y_train,
        epochs=100, # Train for more epochs on the final model
        batch_size=hp_batch_size,
        validation_data=(X_test, y_test),
        callbacks=[final_checkpoint, final_early_stopping]
    )
    print("--- Final Model Training Finished ---")
    final_model.summary()
    
    # 6. Evaluate the Final Model
    evaluate_model(BEST_MODEL_PATH, X_test, y_test, metadata)
    
    print("\n--- Workflow Complete ---")
    print(f"The best tuned model is saved at: {BEST_MODEL_PATH}")