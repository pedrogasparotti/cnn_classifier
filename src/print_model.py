import tensorflow as tf
import os

# --- 1. SET THE PATH TO YOUR MODEL FILE HERE ---
MODEL_PATH = '/Users/home/Documents/github/cnn_classification/cross_val_results/models/best_model.keras'

OUTPUT_IMAGE_PATH = None

def view_and_save_architecture(model_path, output_path=None):
    """
    Loads a Keras model, prints its summary, and saves its architecture as a PNG.
    """
    print(f"Loading model from: {model_path}")

    # --- Check if the file exists ---
    if not os.path.exists(model_path):
        print("\nERROR: Model file not found!")
        print(f"   Please make sure the path is correct: '{model_path}'")
        return

    # --- Load the model ---
    try:
        model = tf.keras.models.load_model(model_path)
        print("\n✅ Model Loaded Successfully!")
    except Exception as e:
        print(f"\n🛑 ERROR: Failed to load the model. It might be corrupted or not a valid Keras file.")
        print(f"   Details: {e}")
        return

    # --- Print the text summary ---
    print("\n--- Model Architecture Summary ---")
    model.summary()
    print("--------------------------------\n")

    # --- Save the .png image ---
    if output_path is None:
        base_name = os.path.basename(model_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_path = os.path.join(os.path.dirname(model_path), f"{name_without_ext}_architecture.png")

    try:
        print(f"Saving architecture plot to: {output_path}")
        tf.keras.utils.plot_model(model, to_file=output_path, show_shapes=True, show_layer_activations=True)
        print(f"Successfully saved architecture plot!")
    except ImportError:
        print("\n⚠️ WARNING: Could not create architecture plot.")
        print("   `pydot` and `graphviz` are required for this.")
        print("   Install them with: pip install pydot")
        print("   You may also need to install graphviz on your system.")
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred while plotting the model: {e}")


# --- Run the function with the path defined above ---
view_and_save_architecture(MODEL_PATH, OUTPUT_IMAGE_PATH)