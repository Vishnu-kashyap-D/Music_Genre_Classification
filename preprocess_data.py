import os
import json
import librosa
import math
import numpy as np

# <<< CRITICAL CHANGE: Updated this path to match your extracted folder.
DATASET_PATH = "archive/Data/genres_original" 
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mel_spectrograms(dataset_path, json_path, n_mels=128, n_fft=2048, hop_length=512, num_segments=10):
    """
    Extracts Mel-spectrograms from audio files and saves them to a JSON file.

    :param dataset_path (str): Path to the directory containing genre subfolders.
    :param json_path (str): Path to the output JSON file.
    :param n_mels (int): Number of Mel bands to generate.
    :param n_fft (int): Length of the Fast Fourier Transform window.
    :param hop_length (int): Number of samples between successive frames.
    :param num_segments (int): Number of segments to split each audio track into.
    """

    # Dictionary to store data
    data = {
        "mapping": [],  # List of genre names, e.g., ["blues", "classical", ...]
        "labels": [],   # Target labels (0 for blues, 1 for classical, etc.)
        "mel_spectrograms": []  # The extracted features (the "images")
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    
    # We need to ensure all segments have the same number of time steps (frames)
    # This is calculated from the segment length and hop_length
    expected_num_vectors_per_segment = int(np.ceil(samples_per_segment / hop_length))

    print(f"Starting audio preprocessing from: {dataset_path}")

    # Loop through all genre sub-folders
    # We start 'i' from 1 so we can skip the root directory easily
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # Ensure we're not at the root level (which is i=0)
        if i > 0:
            
            # Save the semantic label (genre name)
            semantic_label = os.path.basename(dirpath)
            data["mapping"].append(semantic_label)
            print(f"\nProcessing: {semantic_label}")

            # Process all audio files in the genre sub-directory
            for f in filenames:
                
                # Load audio file
                file_path = os.path.join(dirpath, f)
                
                try:
                    # Load the audio file
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                    
                    # Ensure the track is at least 30 seconds long
                    if len(signal) < SAMPLES_PER_TRACK:
                        print(f"\nSkipping {f}: too short")
                        continue

                    # Process all segments of the audio file
                    for s in range(num_segments):
                        start_sample = samples_per_segment * s
                        finish_sample = start_sample + samples_per_segment
                        
                        # Extract Mel-spectrogram
                        mel_spectrogram = librosa.feature.melspectrogram(
                            y=signal[start_sample:finish_sample],
                            sr=sr,
                            n_mels=n_mels,
                            n_fft=n_fft,
                            hop_length=hop_length
                        )
                        
                        # Convert to decibels (dB)
                        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
                        
                        # Ensure the Mel-spectrogram has the expected number of time steps
                        if mel_spectrogram_db.shape[1] == expected_num_vectors_per_segment:
                            # Transpose so shape is (time_steps, n_mels)
                            data["mel_spectrograms"].append(mel_spectrogram_db.T.tolist())
                            data["labels"].append(i - 1) # Use the index (i-1) as the label
                            print(f".", end="")
                        else:
                            # Pad if the segment is slightly shorter (common issue)
                            if mel_spectrogram_db.shape[1] < expected_num_vectors_per_segment:
                                pad_width = expected_num_vectors_per_segment - mel_spectrogram_db.shape[1]
                                mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, pad_width)), mode='constant')
                                data["mel_spectrograms"].append(mel_spectrogram_db.T.tolist())
                                data["labels"].append(i - 1)
                                print(f"p", end="") # 'p' for padded
                            else:
                                # Truncate if slightly longer
                                mel_spectrogram_db = mel_spectrogram_db[:, :expected_num_vectors_per_segment]
                                data["mel_spectrograms"].append(mel_spectrogram_db.T.tolist())
                                data["labels"].append(i - 1)
                                print(f"t", end="") # 't' for truncated

                        
                except Exception as e:
                    print(f"\nError processing file {file_path}: {e}")

    # Save all extracted data to a single JSON file
    print(f"\n\nSaving processed data to {json_path}...")
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
    print("Preprocessing complete!")

if __name__ == "__main__":
    save_mel_spectrograms(DATASET_PATH, JSON_PATH)

