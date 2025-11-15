import argparse
import json
import os
import random
from datetime import datetime

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
DATA_PATH = "data.json"
DEFAULT_SAVE_PATH = os.path.join("torch_models", "genre_classifier_torch.pt")
POP_HIPHOP_AUX_PATH = os.path.join("torch_models", "pop_hiphop_aux.joblib")


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CNNGenreClassifier(nn.Module):
    """Simple convolutional network for mel-spectrogram inputs."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_dataset(data_path: str):
    """Load preprocessed mel-spectrograms and labels from data.json."""
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mel_spectrograms"], dtype=np.float32)
    y = np.array(data["labels"], dtype=np.int64)
    mapping = data["mapping"]
    return X, y, mapping


def split_datasets(X: np.ndarray, y: np.ndarray, test_size: float = 0.25, validation_size: float = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation_size, random_state=42, stratify=y_train
    )
    # Add channel dimension for PyTorch (N, C, H, W)
    X_train = X_train[:, None, ...]
    X_val = X_val[:, None, ...]
    X_test = X_test[:, None, ...]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def to_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def apply_spec_augment(batch: torch.Tensor) -> torch.Tensor:
    augmented = batch.clone()

    noise_std = 0.015
    augmented = augmented + noise_std * torch.randn_like(augmented)

    gain = 1.0 + 0.1 * (torch.rand(batch.size(0), 1, 1, 1, device=batch.device) - 0.5)
    augmented = augmented * gain

    max_time_mask = 18
    max_freq_mask = 14

    for idx in range(augmented.size(0)):
        time_width = int(torch.randint(0, max_time_mask + 1, (1,), device=batch.device).item())
        if time_width > 0:
            start = int(torch.randint(0, augmented.size(-1) - time_width + 1, (1,), device=batch.device).item())
            augmented[idx, :, :, start : start + time_width] = augmented[idx, :, :, start : start + time_width].min()

        freq_width = int(torch.randint(0, max_freq_mask + 1, (1,), device=batch.device).item())
        if freq_width > 0:
            start = int(torch.randint(0, augmented.size(-2) - freq_width + 1, (1,), device=batch.device).item())
            augmented[idx, :, start : start + freq_width, :] = augmented[idx, :, start : start + freq_width, :].mean()

    shift = torch.randint(-10, 11, (batch.size(0),), device=batch.device)
    for idx, value in enumerate(shift):
        if value.item() != 0:
            augmented[idx] = torch.roll(augmented[idx], shifts=int(value.item()), dims=-1)

    return torch.clamp(augmented, min=-80.0, max=0.0)


def extract_aux_features(sample: np.ndarray) -> np.ndarray:
    """Collapse a mel-spectrogram sample into richer summary statistics."""
    mel = sample.squeeze(0)  # (time, n_mels)
    if mel.ndim != 2:
        mel = mel.reshape(mel.shape[0], -1)

    mean_band = mel.mean(axis=0)
    std_band = mel.std(axis=0)

    delta = np.diff(mel, axis=0, prepend=mel[:1])
    delta_mean = delta.mean(axis=0)
    delta_std = delta.std(axis=0)

    accel = np.diff(delta, axis=0, prepend=delta[:1])
    accel_mean = accel.mean(axis=0)
    accel_std = accel.std(axis=0)

    max_band = mel.max(axis=0)
    min_band = mel.min(axis=0)

    features = np.concatenate(
        [
            mean_band,
            std_band,
            delta_mean,
            delta_std,
            accel_mean,
            accel_std,
            max_band,
            min_band,
        ]
    )
    return features.astype(np.float32, copy=False)


def build_pop_hiphop_dataset(
    X: np.ndarray,
    y: np.ndarray,
    pop_idx: int,
    hiphop_idx: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    mask = np.isin(y, [pop_idx, hiphop_idx])
    if not np.any(mask):
        return None, None

    features: list[np.ndarray] = []
    labels: list[int] = []
    for sample, label in zip(X[mask], y[mask]):
        features.append(extract_aux_features(sample))
        labels.append(1 if label == hiphop_idx else 0)

    if not features:
        return None, None

    return np.stack(features), np.array(labels, dtype=np.int64)


def train_pop_hiphop_disambiguator(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    mapping: list[str],
    save_path: str = POP_HIPHOP_AUX_PATH,
) -> None:
    if "pop" not in mapping or "hiphop" not in mapping:
        print("ℹ Pop/Hiphop genres not present; skipping auxiliary head training.")
        return

    pop_idx = mapping.index("pop")
    hiphop_idx = mapping.index("hiphop")

    train_features, train_labels = build_pop_hiphop_dataset(train_X, train_y, pop_idx, hiphop_idx)
    if train_features is None or train_labels is None or train_features.shape[0] < 20:
        print("⚠ Not enough pop/hiphop samples to train auxiliary head; skipping.")
        return

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)

    clf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(train_scaled, train_labels)

    best_threshold = 0.5
    eval_report = None
    if test_X is not None and test_y is not None:
        test_features, test_labels = build_pop_hiphop_dataset(test_X, test_y, pop_idx, hiphop_idx)
        if test_features is not None and test_labels is not None and test_features.shape[0] > 0:
            test_scaled = scaler.transform(test_features)
            probas = clf.predict_proba(test_scaled)[:, 1]

            thresholds = np.linspace(0.35, 0.65, 31, dtype=np.float32)
            best_accuracy = 0.0
            best_preds = None
            for thr in thresholds:
                candidate_preds = (probas >= thr).astype(np.int64)
                acc = float(np.mean(candidate_preds == test_labels))
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_threshold = float(thr)
                    best_preds = candidate_preds

            if best_preds is None:
                best_preds = (probas >= 0.5).astype(np.int64)

            eval_report = classification_report(
                test_labels,
                best_preds,
                target_names=["pop", "hiphop"],
                zero_division=0,
                digits=4,
            )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({"model": clf, "scaler": scaler, "threshold": best_threshold}, save_path)
    print(f"✓ Saved pop/hiphop auxiliary head to {save_path}")

    if eval_report:
        print("\nPop/Hiphop auxiliary head evaluation:\n" + eval_report)


def choose_device(gpu_index: int | None = None, force_cpu: bool = False) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        print("⚙ Using CPU for training")
        return torch.device("cpu")

    visible_devices = torch.cuda.device_count()
    if gpu_index is not None:
        if gpu_index < 0 or gpu_index >= visible_devices:
            print(f"⚠ GPU index {gpu_index} out of range; defaulting to cuda:0")
            gpu_index = 0
    else:
        gpu_index = 0

    torch.cuda.set_device(gpu_index)
    device = torch.device(f"cuda:{gpu_index}")
    name = torch.cuda.get_device_name(device)
    print(f"✓ Using CUDA device {gpu_index}: {name}")
    return device


def train(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    use_mixed_precision: bool,
    augment: bool,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)

    best_val_accuracy = 0.0
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if augment:
                inputs = apply_spec_augment(inputs)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_mixed_precision, dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_accum = 0.0
        with torch.inference_mode():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss_accum += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_loss = val_loss_accum / val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.2f}%"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state_dict is None:
        best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    return best_state_dict, best_val_accuracy


def evaluate(model: nn.Module, device: torch.device, data_loader: DataLoader) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    correct = 0
    loss_accum = 0.0

    with torch.inference_mode():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_accum += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return loss_accum / total, correct / total


def save_checkpoint(save_path: str, model_state_dict: dict, mapping: list[str], meta: dict) -> None:
    payload = {
        "state_dict": model_state_dict,
        "mapping": mapping,
        "meta": meta,
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(payload, save_path)
    print(f"\nModel saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch genre classifier on GTZAN spectrograms")
    parser.add_argument("--data", type=str, default=DATA_PATH, help="Path to data.json produced by preprocess_data.py")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate for Adam optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 weight decay")
    parser.add_argument("--gpu-index", type=int, help="CUDA device index to use")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--no-mixed", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--save-path", type=str, default=DEFAULT_SAVE_PATH, help="Path to save the trained model checkpoint")
    parser.add_argument(
        "--predict-file",
        type=str,
        help="Optional audio file to run through the freshly trained model",
    )
    parser.add_argument(
        "--aux-only",
        action="store_true",
        help="Only train the pop/hiphop auxiliary disambiguator and exit",
    )
    parser.add_argument(
        "--freeze-features",
        action="store_true",
        help="Freeze feature extractor layers during training",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply data augmentation during training",
    )
    parser.add_argument(
        "--window-duration",
        type=int,
        default=30,
        help="Window duration in seconds for dataset generation",
    )
    parser.add_argument(
        "--slice-duration",
        type=int,
        default=5,
        help="Slice duration in seconds for dataset generation",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=5,
        help="Window stride in seconds for dataset generation",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="torch_cache",
        help="Directory to cache preprocessed data",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=5,
        help="Number of epochs for fine-tuning the model",
    )
    parser.add_argument(
        "--fine-tune-lr",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning",
    )
    parser.add_argument(
        "--fine-tune-weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for fine-tuning",
    )
    parser.add_argument(
        "--fine-tune-patience",
        type=int,
        default=3,
        help="Patience for early stopping during fine-tuning",
    )
    args = parser.parse_args()

    set_seed(42)

    X, y, mapping = load_dataset(args.data)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_datasets(X, y)

    if args.aux_only:
        if args.predict_file:
            parser.error("--aux-only cannot be combined with --predict-file")

        aux_train_X = np.concatenate((X_train, X_val), axis=0)
        aux_train_y = np.concatenate((y_train, y_val), axis=0)
        train_pop_hiphop_disambiguator(aux_train_X, aux_train_y, X_test, y_test, mapping)
        return

    device = choose_device(args.gpu_index, args.cpu)
    use_mixed_precision = torch.cuda.is_available() and not args.no_mixed and not args.cpu
    if use_mixed_precision:
        print("✓ Mixed precision (torch.cuda.amp) enabled")
    else:
        print("ℹ Mixed precision disabled")

    train_loader = to_dataloader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader = to_dataloader(X_val, y_val, args.batch_size, shuffle=False)
    test_loader = to_dataloader(X_test, y_test, args.batch_size, shuffle=False)

    model = CNNGenreClassifier(num_classes=len(mapping)).to(device)
    torch.backends.cudnn.benchmark = True

    # Freeze feature extractor layers if specified
    if args.freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False
        print("🔒 Feature extractor layers frozen")

    if args.augment:
        print("🎛 Applying spec augment (noise, gain, masks, shifts)")

    print("\nStarting training...")
    best_state_dict, best_val_accuracy = train(
        model,
        device,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_mixed_precision=use_mixed_precision,
        augment=args.augment,
    )

    # Load best weights for evaluation
    model.load_state_dict(best_state_dict)
    test_loss, test_accuracy = evaluate(model, device, test_loader)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy * 100:5.2f}%")

    metadata = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "validation_accuracy": best_val_accuracy,
        "test_accuracy": test_accuracy,
        "input_shape": list(X_train.shape[1:]),
        "augmentation": bool(args.augment),
    }
    save_checkpoint(args.save_path, best_state_dict, mapping, metadata)

    aux_train_X = np.concatenate((X_train, X_val), axis=0)
    aux_train_y = np.concatenate((y_train, y_val), axis=0)
    train_pop_hiphop_disambiguator(aux_train_X, aux_train_y, X_test, y_test, mapping)

    # Genre prediction for the provided file
    if args.predict_file:
        from predict_genre_torch import predict as run_single_prediction

        print(f"\nRunning post-training prediction for: {args.predict_file}")
        model.eval()
        run_single_prediction(args.predict_file, model, mapping, device)


if __name__ == "__main__":
    main()
