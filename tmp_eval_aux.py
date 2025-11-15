import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
targets = (labels[mask] == hiphop_idx).astype(np.int64)


def to_features(sample: np.ndarray) -> np.ndarray:
    mel = sample.squeeze(0)
    mean = mel.mean(axis=1)
    std = mel.std(axis=1)
    return np.concatenate([mean, std]).astype(np.float32)


feature_matrix = np.stack([to_features(sample) for sample in mel_segments])
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix,
    targets,
    test_size=0.25,
    random_state=42,
    stratify=targets,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "svm": SVC(kernel="rbf", class_weight="balanced", gamma="scale"),
    "rf": RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    ),
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    print(name)
    print(
        classification_report(
            y_test,
            preds,
            target_names=["pop", "hiphop"],
            zero_division=0,
            digits=4,
        )
    )
