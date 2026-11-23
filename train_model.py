import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import argparse
import os
import argparse

# Path to the data file you just created
DATA_PATH = "data.json"

# Configure GPU settings for NVIDIA CUDA
def configure_gpu():
    """
    Configures TensorFlow to use GPU with optimal settings.
    Enables memory growth to avoid allocating all GPU memory at once.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for each GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set GPU as visible device
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"✓ GPU ENABLED: {len(gpus)} Physical GPU(s), {len(logical_gpus)} Logical GPU(s)")
            print(f"  GPU Device(s): {[gpu.name for gpu in gpus]}")
            
            # Enable mixed precision for faster training on modern GPUs
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("✓ Mixed precision training enabled (FP16) for faster computation")
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("⚠ WARNING: No GPU detected. Running on CPU (will be slower)")
        print("  Make sure you have:")
        print("  1. NVIDIA GPU with CUDA support")
        print("  2. CUDA Toolkit installed")
        print("  3. cuDNN installed")
        print("  4. TensorFlow GPU version: pip install tensorflow[and-cuda]")
    
    return len(gpus) > 0

def load_data(data_path):
    """
    Loads training data from a JSON file.
    
    :param data_path (str): Path to the JSON file.
    :return:
        X (ndarray): Inputs (Mel-spectrograms).
        y (ndarray): Target labels (integers).
        mapping (list): List of genre names.
    """
    print("Loading data...")
    with open(data_path, "r") as fp:
        data = json.load(fp)

    # Convert lists to numpy arrays
    X = np.array(data["mel_spectrograms"])
    y = np.array(data["labels"])
    mapping = data["mapping"]
    
    print("Data loaded!")
    return X, y, mapping

def prepare_datasets(X, y, test_size=0.25, validation_size=0.2):
    """
    Splits the data into training, validation, and test sets.
    
    :param X (ndarray): Inputs.
    :param y (ndarray): Targets.
    :param test_size (float): Proportion of data to use for the test set.
    :param validation_size (float): Proportion of training data to use for the validation set.
    :return:
        X_train, y_train, X_validation, y_validation, X_test, y_test
    """
    
    # Create the main train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Create a validation split from the training data
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=validation_size, random_state=42
    )
    
    # --- IMPORTANT ---
    # A CNN expects a 4D array for its input: (num_samples, height, width, channels)
    # Our Mel-spectrograms are 2D (height, width), so we add a 'channels' dimension of 1.
    # This is like telling the CNN it's a grayscale image.
    
    # Add the channel dimension to the arrays
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    return X_train, y_train, X_validation, y_validation, X_test, y_test

def build_model(input_shape, num_genres):
    """
    Builds and compiles the CNN model.
    
    :param input_shape (tuple): Shape of the input data (height, width, channels).
    :param num_genres (int): Number of unique genres (output neurons).
    :return:
        model (tf.keras.Model): The compiled CNN model.
    """
    
    # Create a Sequential model
    model = tf.keras.Sequential()

    # --- 1st Convolutional Block ---
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.25))

    # --- 2nd Convolutional Block ---
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.25))

    # --- 3rd Convolutional Block ---
    model.add(tf.keras.layers.Conv2D(128, (2, 2), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.25))

    # --- Flatten layer ---
    # This converts the 2D feature maps into a 1D vector for the Dense layers
    model.add(tf.keras.layers.Flatten())

    # --- Dense (Fully Connected) Layer ---
    model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Dropout(0.5))

    # --- Output Layer ---
    # CRITICAL:
    # - num_genres (10) outputs
    # - 'softmax' activation: This is what gives you the probabilities for each genre!
    model.add(tf.keras.layers.Dense(num_genres, activation='softmax'))

    # --- Compile the Model ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the GTZAN genre classifier CNN")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--no-mixed", action="store_true", help="Disable mixed precision even if GPU present")
    parser.add_argument("--data", type=str, default=DATA_PATH, help="Path to data.json")
    parser.add_argument("--gpu-index", type=int, help="Which CUDA GPU index to use (0-based). If omitted, all visible GPUs may be used.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU only (ignore GPUs)")
    args = parser.parse_args()

    print("=" * 60)
    print("INITIALIZING GPU CONFIGURATION")
    print("=" * 60)
    # Honor explicit device selection before TensorFlow initializes GPUs
    if args.cpu:
        os.environ["FORCE_CPU"] = "1"
    elif args.gpu_index is not None:
        os.environ["GPU_INDEX"] = str(args.gpu_index)
    has_gpu = configure_gpu()
    if args.no_mixed and has_gpu:
        # Reset to float32 if user disables mixed precision
        tf.keras.mixed_precision.set_global_policy('float32')
        print("Mixed precision disabled by flag --no-mixed")
    print("=" * 60)
    print()

    # Load the data
    X, y, mapping = load_data(args.data)

    # Split the data
    X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_datasets(X, y)

    # Get the input shape from the training data
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    num_genres = len(mapping)

    print("\nBuilding model on GPU..." if has_gpu else "\nBuilding model on CPU...")
    model = build_model(input_shape, num_genres)

    print("\n" + "=" * 60)
    print("TRAINING ON GPU" if has_gpu else "TRAINING ON CPU")
    print("=" * 60)
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size}")
    print("This may take some time depending on your GPU...\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_validation, y_validation),
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    print("\nEvaluating on the test set:")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"\nTest Loss: {test_loss:.3f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")

    model_save_path = "genre_classifier.keras"
    model.save(model_save_path)
    print(f"\nModel saved successfully to {model_save_path}")

