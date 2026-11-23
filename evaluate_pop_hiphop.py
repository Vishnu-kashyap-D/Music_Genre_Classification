import json
from pathlib import Path

import numpy as np
import torch

from predict_genre_torch import apply_pop_hiphop_resolver, load_pop_hiphop_aux
from train_model_torch import CNNGenreClassifier

MODEL_PATH = Path("torch_models/genre_classifier_torch.pt")
DATA_PATH = Path("data.json")


with DATA_PATH.open(encoding="utf-8") as fh:
    payload = json.load(fh)

mapping = payload["mapping"]
pop_idx = mapping.index("pop")
hiphop_idx = mapping.index("hiphop")

mel_segments = np.array(payload["mel_spectrograms"], dtype=np.float32)
labels = np.array(payload["labels"], dtype=np.int64)

mask = np.isin(labels, [pop_idx, hiphop_idx])
mel_segments = mel_segments[mask][:, None, :, :]
labels = labels[mask]

model_payload = torch.load(MODEL_PATH, map_location="cpu")
model = CNNGenreClassifier(num_classes=len(mapping))
model.load_state_dict(model_payload["state_dict"])
model.eval()

softmax = torch.nn.Softmax(dim=1)

correct_base = 0
correct_adjusted = 0

for segment, label in zip(mel_segments, labels):
    tensor = torch.from_numpy(segment).unsqueeze(0)
    with torch.inference_mode():
        logits = model(tensor)
        probs = softmax(logits)[0].numpy()

    base_pred = np.argmax(probs)
    if base_pred == label:
        correct_base += 1

    adjusted = apply_pop_hiphop_resolver(probs.copy(), segment[np.newaxis, ...], mapping)
    adjusted_pred = np.argmax(adjusted)
    if adjusted_pred == label:
        correct_adjusted += 1

print(f"Segments evaluated: {len(labels)}")
print(f"Base accuracy: {correct_base / len(labels):.4f}")
print(f"Adjusted accuracy: {correct_adjusted / len(labels):.4f}")
