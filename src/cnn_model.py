import numpy as np
import json
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

PROCESSED_DATA_DIR = os.path.join('data', 'processed')
MODEL_OUTPUT_DIR = 'models'
BEST_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, 'best_cnn_model.keras')

# Training Hyperparameters
EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

def load_data(data_dir):
    """
    Loads the preprocessed training and testing datasets.
    Your data must be split correctly BEFORE you even think about modeling.
    """
    print(f"Loading data from: {data_dir}")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}. Run the preprocessing script first.")

    X_train = np.load(os.path.join(data_dir, 'X_train_sub.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train_sub.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test_sub.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test_sub.npy'))
    
    with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
        
    print("Data loaded successfully.")
    return X_train, y_train, X_test, y_test, metadata

def build_model(input_shape, num_classes):
    """
    Builds a 1D CNN model using GlobalAveragePooling1D.
    """
    # Use the modern Keras functional API starting with an Input layer
    inputs = tf.keras.Input(shape=input_shape)

    # --- Feature Extraction Block 1 ---
    x = Conv1D(filters=32, kernel_size=5, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    # --- Feature Extraction Block 2 ---
    x = Conv1D(filters=64, kernel_size=5, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # --- Feature Extraction Block 3 ---
    x = Conv1D(filters=128, kernel_size=5, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # --- Classifier Head ---
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Model built and compiled.")
    model.summary()
    return model
    
def train_model(model, X_train, y_train):
    """
    Trains the model using callbacks
    """
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    # Save the best model based on validation loss, not the last one.
    checkpoint = ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Stop training if the model isn't improving. This saves time and prevents overfitting.
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10, # Stop after 10 epochs of no improvement
        restore_best_weights=True,
        verbose=1
    )
    
    print("\n--- Starting Model Training ---")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[checkpoint, early_stopping]
    )
    print("--- Model Training Finished ---\n")
    return history

def evaluate_model(X_test, y_test, metadata):
    """
    Evaluates the BEST saved model on the unseen test set.
    """
    print("--- Evaluating Best Model on Test Set ---")
    
    # Load the best performing model
    best_model = load_model(BEST_MODEL_PATH)
    
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
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_int, y_pred_int)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix on Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot
    plot_path = os.path.join(MODEL_OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(plot_path)
    print(f"Confusion matrix plot saved to {plot_path}")
    plt.show()

if __name__ == '__main__':
    # 1. Load Data
    X_train, y_train, X_test, y_test, metadata = load_data(PROCESSED_DATA_DIR)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = y_train.shape[1]

    # 2. Build Model
    model = build_model(input_shape, num_classes)
    
    # 3. Train Model
    history = train_model(model, X_train, y_train)
    
    # 4. Evaluate Model

    evaluate_model(X_test, y_test, metadata)
    
    print("\n--- Workflow Complete ---")
    print(f"The best trained model is saved at: {BEST_MODEL_PATH}")
    print("Use the 'X_holdout.npy' dataset for final 'double blind' validation on this saved model.")