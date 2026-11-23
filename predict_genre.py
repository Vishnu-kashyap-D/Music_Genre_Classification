
import json
import importlib
import numpy as np
import tensorflow as tf
import librosa
import sys  # Used to read command-line arguments
import os   # For file name handling in prints and GPU selection

librosa_utils = importlib.import_module("librosa.util.utils")

SILENCE_RMS_THRESHOLD = 1e-4
SILENCE_PEAK_THRESHOLD = 1e-3

# Configure GPU for inference
def configure_gpu():
    """Configure TensorFlow to use a specific GPU (if requested) and enable memory growth."""
    # Allow forcing CPU only via env var
    if os.environ.get("FORCE_CPU") == "1":
        try:
            tf.config.set_visible_devices([], 'GPU')
            print("⚙ Forcing CPU: all GPUs hidden via FORCE_CPU=1")
        except Exception as e:
            print(f"GPU visibility configuration error (FORCE_CPU): {e}. Continuing on CPU.")
        return

    # Optional GPU index selection via env var (must occur before memory growth settings)
    gpu_index_env = os.environ.get("GPU_INDEX")
    try:
        gpus_all = tf.config.list_physical_devices('GPU')
        if gpus_all and gpu_index_env is not None:
            try:
                idx = int(gpu_index_env)
                if 0 <= idx < len(gpus_all):
                    tf.config.set_visible_devices(gpus_all[idx], 'GPU')
                    print(f"⚙ Selecting GPU index {idx} via GPU_INDEX={gpu_index_env}")
                else:
                    print(f"GPU_INDEX={gpu_index_env} out of range (0..{len(gpus_all)-1}). Using all visible GPUs.")
            except ValueError:
                print(f"Invalid GPU_INDEX value: {gpu_index_env}. Using all visible GPUs.")
    except Exception as e:
        print(f"GPU visibility configuration error: {e}")

    # After visibility is set, list current visible GPUs and configure memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU Enabled for prediction: {len(gpus)} GPU(s) visible")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("⚠ No GPU detected - using CPU for prediction")

# Configure GPU before any model operations
configure_gpu()

# --- Constants ---
MODEL_PATH = "genre_classifier.keras"
DATA_JSON_PATH = "data.json"
SAMPLE_RATE = 22050
DURATION_SECONDS = 30 # Duration of original tracks
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION_SECONDS
NUM_SEGMENTS = 10
SAMPLES_PER_SEGMENT = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
EXPECTED_VECTORS_PER_SEGMENT = 130 # This was calculated in preprocess_data.py

class GenrePredictor:
    """
    This class encapsulates all the logic needed to load the model
    and make predictions on new audio files.
    """
    
    def __init__(self, model_path, data_json_path):
        print(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        
        print(f"Loading genre mapping from {data_json_path}...")
        with open(data_json_path, "r") as fp:
            data = json.load(fp)
            self.mapping = data["mapping"]
        
        print("Model and mapping loaded successfully.")

    def preprocess_audio(self, file_path):
        """
        Loads and preprocesses a single audio file, converting it
        into the Mel-spectrogram format the model expects.
        
        :param file_path (str): Path to the audio file.
        :return: (ndarray): The processed Mel-spectrogram, ready for the model.
        """
        
        try:
            # 1. Load the audio file
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            
            # 2. Check if it's long enough, if not, pad it
            if len(signal) < SAMPLES_PER_TRACK:
                print(f"Warning: Audio file is shorter than {DURATION_SECONDS}s. Padding with silence.")
                signal = librosa_utils.fix_length(signal, size=SAMPLES_PER_TRACK)
            
            # 3. Take only the first 30 seconds (if it's longer)
            signal = signal[:SAMPLES_PER_TRACK]
            
            # 4. Detect silence before heavy processing
            rms = float(np.mean(librosa.feature.rms(y=signal)))
            peak = float(np.max(np.abs(signal)))
            if rms < SILENCE_RMS_THRESHOLD and peak < SILENCE_PEAK_THRESHOLD:
                return None

            # 5. Extract Mel-spectrogram for each segment
            all_segment_spectrograms = []
            
            for s in range(NUM_SEGMENTS):
                start_sample = SAMPLES_PER_SEGMENT * s
                finish_sample = start_sample + SAMPLES_PER_SEGMENT
                
                mel_spectrogram = librosa.feature.melspectrogram(
                    y=signal[start_sample:finish_sample],
                    sr=sr,
                    n_mels=128,
                    n_fft=2048,
                    hop_length=512
                )
                
                # IMPORTANT: match preprocessing scale (preprocess_data.py uses ref=np.max)
                mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
                
                # 6. Ensure the spectrogram has the expected shape
                if mel_spectrogram_db.shape[1] == EXPECTED_VECTORS_PER_SEGMENT:
                    all_segment_spectrograms.append(mel_spectrogram_db.T)
                elif mel_spectrogram_db.shape[1] < EXPECTED_VECTORS_PER_SEGMENT:
                    # Pad if shorter
                    pad_width = EXPECTED_VECTORS_PER_SEGMENT - mel_spectrogram_db.shape[1]
                    mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, pad_width)), mode='constant')
                    all_segment_spectrograms.append(mel_spectrogram_db.T)
                else:
                    # Truncate if longer
                    mel_spectrogram_db = mel_spectrogram_db[:, :EXPECTED_VECTORS_PER_SEGMENT]
                    all_segment_spectrograms.append(mel_spectrogram_db.T)

            # 7. Convert to numpy array and add the channel + batch dimensions
            # Final shape should be (num_segments, height, width, 1)
            X = np.array(all_segment_spectrograms, dtype=np.float32)[..., np.newaxis]
            return X

        except Exception as e:
            print(f"Error processing audio file {file_path}: {e}")
            return None

    def predict(self, file_path):
        """
        Makes a genre prediction on a new audio file.
        
        :param file_path (str): Path to the audio file.
        """
        
        # 1. Preprocess the audio file
        processed_input = self.preprocess_audio(file_path)
        
        if processed_input is None:
            print(f"\n--- Prediction for: {os.path.basename(file_path)} ---")
            print("Silence detected; returning zero probabilities for all genres.")
            for genre in self.mapping:
                print(f"{genre.capitalize()}: 0.00%")
            print("------------------------------------------")
            return

        # 2. Get predictions for all 10 segments
        all_predictions = self.model.predict(processed_input)
        
        # 3. Average the predictions across all segments
        # This gives a more robust, overall prediction for the entire song
        average_prediction = np.mean(all_predictions, axis=0)
        
        # 4. Get the top 3 predicted genres
        # argsort gives the indices from lowest to highest, so we reverse it [::-1]
        top_3_indices = np.argsort(average_prediction)[::-1][:3]
        
        # 5. Print the results
        print(f"\n--- Prediction for: {os.path.basename(file_path)} ---")
        
        for i in top_3_indices:
            genre = self.mapping[i]
            probability = average_prediction[i] * 100
            print(f"{genre.capitalize()}: {probability:.2f}%")
            
        print("------------------------------------------")


if __name__ == "__main__":
    # Ensure the user has provided an audio file path
    if len(sys.argv) < 2:
        print(f"Usage: py {sys.argv[0]} <path_to_audio_file>")
        print("\nExample:")
        print(f"py {sys.argv[0]} \"archive/Data/genres_original/jazz/jazz.00005.wav\"")
    else:
        # Create the predictor
        predictor = GenrePredictor(MODEL_PATH, DATA_JSON_PATH)
        
        # Get the file path from the command line
        file_to_predict = sys.argv[1]
        
        # Make the prediction
        predictor.predict(file_to_predict)
