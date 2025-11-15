import json
from pathlib import Path

import numpy as np
import torch

from predict_genre_torch import apply_pop_hiphop_resolver
from train_model_torch import CNNGenreClassifier

MODEL_PATH = Path("torch_models/genre_classifier_torch.pt")
DATA_PATH = Path("data.json")
SEGMENTS_PER_TRACK = 10

with DATA_PATH.open(encoding="utf-8") as fh:
    payload = json.load(fh)

mapping = payload["mapping"]
pop_idx = mapping.index("pop")
hiphop_idx = mapping.index("hiphop")

mel_segments = np.array(payload["mel_spectrograms"], dtype=np.float32)
labels = np.array(payload["labels"], dtype=np.int64)

model_state = torch.load(MODEL_PATH, map_location="cpu")
model = CNNGenreClassifier(num_classes=len(mapping))
model.load_state_dict(model_state["state_dict"])
model.eval()

softmax = torch.nn.Softmax(dim=1)

track_predictions_base = []
track_predictions_adjusted = []
track_labels = []

num_tracks = len(mel_segments) // SEGMENTS_PER_TRACK
for i in range(num_tracks):
    start = i * SEGMENTS_PER_TRACK
    end = start + SEGMENTS_PER_TRACK
    track_mels = mel_segments[start:end]
    track_label = labels[start]

    if track_label not in (pop_idx, hiphop_idx):
        continue

    batch = torch.from_numpy(track_mels[:, None, :, :])
    with torch.inference_mode():
        logits = model(batch)
        probs = softmax(logits).numpy()

    averaged = probs.mean(axis=0)
    base_pred = np.argmax(averaged)
    adjusted = apply_pop_hiphop_resolver(averaged.copy(), track_mels[:, None, :, :], mapping)
    adjusted_pred = np.argmax(adjusted)

    track_predictions_base.append(base_pred)
    track_predictions_adjusted.append(adjusted_pred)
    track_labels.append(track_label)

track_labels = np.array(track_labels)
track_predictions_base = np.array(track_predictions_base)
track_predictions_adjusted = np.array(track_predictions_adjusted)

base_acc = np.mean(track_predictions_base == track_labels)
adjusted_acc = np.mean(track_predictions_adjusted == track_labels)

print(f"Tracks evaluated: {len(track_labels)}")
print(f"Base accuracy: {base_acc:.4f}")
print(f"Adjusted accuracy: {adjusted_acc:.4f}")
