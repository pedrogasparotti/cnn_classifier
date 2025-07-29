import numpy as np
import json
import os
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
# These paths must match the output of your previous scripts
PROCESSED_DATA_DIR = os.path.join('data', 'processed')
MODEL_OUTPUT_DIR = 'models'
BEST_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, 'best_cnn_model.keras')

def evaluate_on_holdout():
    """
    Loads the final model and evaluates it on the unseen holdout dataset.
    This provides the most unbiased measure of model performance.
    """
    print("--- Starting Final Evaluation on Holdout Set ---")

    # --- 1. Load Assets ---
    # Check if all necessary files exist before starting
    required_files = [
        BEST_MODEL_PATH,
        os.path.join(PROCESSED_DATA_DIR, 'X_holdout.npy'),
        os.path.join(PROCESSED_DATA_DIR, 'y_holdout_int.npy'),
        os.path.join(PROCESSED_DATA_DIR, 'metadata.json')
    ]
    for f in required_files:
        if not os.path.exists(f):
            print(f"ERROR: Required file not found: {f}")
            return

    # Load the trained model
    print(f"Loading model from: {BEST_MODEL_PATH}")
    model = tf.keras.models.load_model(BEST_MODEL_PATH)

    # Load the holdout data and metadata
    print("Loading holdout data...")
    X_holdout = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_holdout.npy'))
    y_holdout_int = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_holdout_int.npy'))
    with open(os.path.join(PROCESSED_DATA_DIR, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    class_names = metadata['class_names']
    print("Assets loaded successfully.")

    # --- 2. Make Predictions ---
    print("\nGenerating predictions on holdout data...")
    y_pred_probs = model.predict(X_holdout)
    y_pred_int = np.argmax(y_pred_probs, axis=1)

    # --- 3. Generate and Display Report & Matrix ---
    print("\nClassification Report (Holdout Set):")
    print(classification_report(y_holdout_int, y_pred_int, target_names=class_names))

    print("Generating confusion matrix...")
    cm = confusion_matrix(y_holdout_int, y_pred_int)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 16}) # Increase font size for readability
    plt.title('Confusion Matrix - Holdout Set Performance', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    # Save the plot
    plot_path = os.path.join(MODEL_OUTPUT_DIR, 'holdout_confusion_matrix.png')
    plt.savefig(plot_path)
    print(f"\nConfusion matrix plot saved to: {plot_path}")
    
    plt.show()


if __name__ == '__main__':
    evaluate_on_holdout()
    print("\nHoldout evaluation complete.")