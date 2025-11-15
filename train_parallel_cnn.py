import argparse
import os
import hashlib
import importlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from train_model_torch import choose_device, set_seed

librosa_utils = importlib.import_module("librosa.util.utils")

try:  # optional dependency for OpenL3 embeddings
    import openl3
except ImportError:  # pragma: no cover - handled at runtime when feature enabled
    openl3 = None

DEFAULT_DATASET_PATH = os.path.join("Data", "genres_original")
DEFAULT_SAVE_PATH = os.path.join("torch_models", "parallel_genre_classifier_torch.pt")
DEFAULT_CACHE_DIR = os.path.join("torch_cache", "parallel")


@dataclass
class DatasetConfig:
    sample_rate: int = 22050
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    window_duration: int = 15
    slice_duration: int = 5
    window_stride: int = 5

    def __post_init__(self) -> None:
        if self.window_duration % self.slice_duration != 0:
            raise ValueError("window_duration must be divisible by slice_duration")

    @property
    def slices_per_window(self) -> int:
        return self.window_duration // self.slice_duration

    @property
    def window_samples(self) -> int:
        return self.sample_rate * self.window_duration

    @property
    def slice_samples(self) -> int:
        return self.sample_rate * self.slice_duration

    @property
    def stride_samples(self) -> int:
        return self.sample_rate * self.window_stride

    @property
    def expected_frames(self) -> int:
        return int(np.ceil(self.slice_samples / self.hop_length))


@dataclass
class OpenL3Config:
    embedding_dim: int = 512
    content_type: str = "music"
    input_repr: str = "mel256"
    center: bool = False
    hop_size: float = 0.5
    batch_size: int = 32

    def cache_tag(self) -> str:
        return f"openl3_{self.embedding_dim}_{self.content_type}_{self.input_repr}".replace("/", "-")


def load_openl3_model(config: OpenL3Config):
    if openl3 is None:
        raise RuntimeError("openl3 package is not installed.")
    return openl3.models.load_audio_embedding_model(
        input_repr=config.input_repr,
        content_type=config.content_type,
        embedding_size=config.embedding_dim,
    )


def list_audio_files(dataset_path: str) -> Tuple[List[str], List[int], List[str]]:
    mapping: List[str] = []
    file_paths: List[str] = []
    labels: List[int] = []

    for genre_dir in sorted(Path(dataset_path).iterdir()):
        if not genre_dir.is_dir():
            continue
        mapping.append(genre_dir.name)
        label_idx = len(mapping) - 1
        for audio_file in sorted(genre_dir.glob("*.wav")):
            file_paths.append(str(audio_file))
            labels.append(label_idx)
    if not file_paths:
        raise FileNotFoundError(f"No audio files found under {dataset_path}")
    return file_paths, labels, mapping


def pad_or_truncate(mel_db: np.ndarray, expected_frames: int) -> np.ndarray:
    if mel_db.shape[1] == expected_frames:
        return mel_db
    if mel_db.shape[1] < expected_frames:
        pad_width = expected_frames - mel_db.shape[1]
        return np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant", constant_values=-80.0)
    return mel_db[:, :expected_frames]


def compute_mel_slices(signal: np.ndarray, cfg: DatasetConfig) -> List[np.ndarray]:
    slices: List[np.ndarray] = []
    for idx in range(cfg.slices_per_window):
        start = idx * cfg.slice_samples
        end = start + cfg.slice_samples
        seg = signal[start:end]
        mel = librosa.feature.melspectrogram(
            y=seg,
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max, top_db=None)
        mel_db = pad_or_truncate(mel_db, cfg.expected_frames)
        slices.append(mel_db.astype(np.float32, copy=False))
    return slices


def _iter_window_chunks(signal: np.ndarray, cfg: DatasetConfig) -> List[np.ndarray]:
    if signal.size < cfg.window_samples:
        fixed = librosa_utils.fix_length(signal, size=cfg.window_samples)
        return [fixed]

    max_start = signal.size - cfg.window_samples
    start_positions = list(range(0, max_start + 1, cfg.stride_samples))
    if not start_positions or start_positions[-1] != max_start:
        start_positions.append(max_start)

    chunks: List[np.ndarray] = []
    for start in start_positions:
        chunk = signal[start : start + cfg.window_samples]
        if chunk.size < cfg.window_samples:
            chunk = librosa_utils.fix_length(chunk, size=cfg.window_samples)
        chunks.append(chunk)
    return chunks


def _compute_openl3_slices(
    signal: np.ndarray,
    cfg: DatasetConfig,
    l3_config: OpenL3Config,
    l3_model,
) -> Optional[np.ndarray]:
    if openl3 is None:
        raise RuntimeError("openl3 package is required for OpenL3 feature extraction but is not installed")

    slices: List[np.ndarray] = []
    for idx in range(cfg.slices_per_window):
        start = idx * cfg.slice_samples
        end = start + cfg.slice_samples
        seg = signal[start:end]
        if seg.size < cfg.slice_samples:
            seg = librosa_utils.fix_length(seg, size=cfg.slice_samples)

        embeddings, _ = openl3.get_audio_embedding(
            seg,
            sr=cfg.sample_rate,
            hop_size=l3_config.hop_size,
            center=l3_config.center,
            batch_size=l3_config.batch_size,
            model=l3_model,
        )
        if embeddings.size == 0:
            return None
        slice_emb = embeddings.mean(axis=0).astype(np.float32, copy=False)
        slices.append(slice_emb)

    return np.stack(slices, axis=0)


def compute_windows_for_track(
    file_path: str,
    cfg: DatasetConfig,
    feature_type: str,
    l3_config: Optional[OpenL3Config] = None,
    l3_model=None,
    *,
    offset: Optional[float] = None,
    duration: Optional[float] = None,
    max_windows: Optional[int] = None,
) -> List[np.ndarray]:
    load_kwargs = {"sr": cfg.sample_rate, "mono": True}
    if offset is not None and offset > 0:
        load_kwargs["offset"] = max(offset, 0.0)
    if duration is not None and duration > 0:
        load_kwargs["duration"] = max(duration, 0.0)

    try:
        signal, _ = librosa.load(file_path, **load_kwargs)
    except Exception:
        return []

    windows: List[np.ndarray] = []
    chunks = _iter_window_chunks(signal, cfg)

    for chunk in chunks:
        if feature_type == "mel":
            slices = compute_mel_slices(chunk, cfg)
            windows.append(np.stack([slc[None, ...] for slc in slices], axis=0))
        elif feature_type == "openl3":
            if l3_config is None or l3_model is None:
                raise ValueError("OpenL3 configuration and model must be provided when feature_type='openl3'")
            l3_slices = _compute_openl3_slices(chunk, cfg, l3_config, l3_model)
            if l3_slices is None:
                continue
            windows.append(l3_slices)
        else:
            raise ValueError(f"Unsupported feature_type '{feature_type}'")

    if max_windows is not None and max_windows > 0:
        return windows[:max_windows]

    return windows


def _track_cache_name(track_path: str) -> str:
    digest = hashlib.sha1(track_path.encode("utf-8")).hexdigest()
    return f"{digest}.npz"


def build_split_features(
    tracks: Sequence[Tuple[str, int]],
    cfg: DatasetConfig,
    cache_dir: str | None,
    split_name: str,
    feature_type: str,
    l3_config: Optional[OpenL3Config],
    l3_model,
) -> Tuple[np.ndarray, np.ndarray]:
    aggregate_cache = None
    track_cache_dir = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_suffix = f"{feature_type}_{cfg.window_duration}s_{cfg.slice_duration}s_stride{cfg.window_stride}"
        if feature_type == "openl3" and l3_config is not None:
            cache_suffix += f"_{l3_config.cache_tag()}"
        aggregate_cache = os.path.join(
            cache_dir,
            f"parallel_{split_name}_{cache_suffix}.npz",
        )
        track_cache_dir = os.path.join(cache_dir, f"{split_name}_tracks_{cache_suffix}")
        os.makedirs(track_cache_dir, exist_ok=True)

    if aggregate_cache and os.path.exists(aggregate_cache):
        data = np.load(aggregate_cache)
        return data["X"], data["y"]

    features: List[np.ndarray] = []
    labels: List[int] = []
    processed = 0
    total_tracks = len(tracks)
    interrupted = False

    try:
        for file_path, label in tracks:
            cache_file = None
            if track_cache_dir:
                cache_file = os.path.join(track_cache_dir, _track_cache_name(file_path))
            if cache_file and os.path.exists(cache_file):
                cached = np.load(cache_file)
                cached_windows = cached["X"]
                windows = [np.array(w, dtype=np.float32, copy=False) for w in np.asarray(cached_windows)]
            else:
                windows = compute_windows_for_track(file_path, cfg, feature_type, l3_config, l3_model)
                if cache_file and windows:
                    np.savez_compressed(cache_file, X=np.asarray(windows, dtype=np.float32))

            if not windows:
                processed += 1
                continue

            features.extend(windows)
            labels.extend([label] * len(windows))
            processed += 1
            if processed % 10 == 0 or processed == total_tracks:
                print(f"Processed {processed}/{total_tracks} tracks for '{split_name}' split")
    except KeyboardInterrupt:
        interrupted = True

    if not features:
        raise RuntimeError(f"No windows were generated for split '{split_name}'")

    X = np.stack(features).astype(np.float32)
    y = np.array(labels, dtype=np.int64)

    if aggregate_cache and not interrupted:
        np.savez_compressed(aggregate_cache, X=X, y=y)

    if interrupted:
        print(
            "KeyboardInterrupt detected while processing feature cache. "
            "Partial per-track caches were saved; rerun to resume."
        )
        raise KeyboardInterrupt

    return X, y


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.sequential = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.sequential(x)
        return x * scale


class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels)
        self.activation = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.activation(out)
        return out


class MelSliceEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = ResidualSEBlock(32, 64, stride=2)
        self.layer2 = ResidualSEBlock(64, 128, stride=2)
        self.layer3 = ResidualSEBlock(128, 128, stride=2)

        self.dropout = nn.Dropout(0.3)
        self.project = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        avg_pool = F.adaptive_avg_pool2d(x, 1).flatten(1)
        max_pool = F.adaptive_max_pool2d(x, 1).flatten(1)
        combined = torch.cat([avg_pool, max_pool], dim=1)
        combined = self.dropout(combined)
        return self.project(combined)


class EmbeddingSliceEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__()
        hidden = max(embedding_dim * 2, 128)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ParallelCNN(nn.Module):
    def __init__(
        self,
        num_slices: int,
        num_classes: int,
        embedding_dim: int = 128,
        shared_backbone: bool = True,
        feature_type: str = "mel",
        input_feature_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_slices = num_slices
        self.shared_backbone = shared_backbone
        self.feature_type = feature_type

        if feature_type == "mel":
            if shared_backbone:
                self.backbone = MelSliceEncoder(embedding_dim)
            else:
                self.branches = nn.ModuleList([MelSliceEncoder(embedding_dim) for _ in range(num_slices)])
        elif feature_type == "openl3":
            if input_feature_dim is None:
                raise ValueError("input_feature_dim must be provided when feature_type='openl3'")
            if shared_backbone:
                self.backbone = EmbeddingSliceEncoder(input_feature_dim, embedding_dim)
            else:
                self.branches = nn.ModuleList(
                    [EmbeddingSliceEncoder(input_feature_dim, embedding_dim) for _ in range(num_slices)]
                )
        else:
            raise ValueError(f"Unsupported feature_type '{feature_type}'")
        heads = 8
        while embedding_dim % heads != 0 and heads > 1:
            heads //= 2
        self.mha_heads = max(heads, 1)
        self.slice_norm = nn.LayerNorm(embedding_dim)
        self.multihead = nn.MultiheadAttention(embedding_dim, num_heads=self.mha_heads, batch_first=True)

        attn_hidden = max(embedding_dim // 2, 1)
        self.pool_attention = nn.Sequential(
            nn.Linear(embedding_dim, attn_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(attn_hidden, 1),
        )

        fusion_input = embedding_dim * 3
        self.fusion_dropout = nn.Dropout(0.3)
        self.head = nn.Sequential(
            nn.Linear(fusion_input, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def encode_slice(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        if self.shared_backbone:
            return self.backbone(x)
        return self.branches[idx](x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = [self.encode_slice(x[:, idx], idx) for idx in range(self.num_slices)]
        stacked = torch.stack(embeddings, dim=1)

        normalized = self.slice_norm(stacked)
        attn_output, _ = self.multihead(normalized, normalized, normalized)
        attn_output = self.slice_norm(attn_output)

        attn_scores = self.pool_attention(attn_output)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_pooled = (attn_output * attn_weights).sum(dim=1)

        avg_pooled = attn_output.mean(dim=1)
        max_pooled = attn_output.max(dim=1).values

        fused = torch.cat([attn_pooled, avg_pooled, max_pooled], dim=1)
        fused = self.fusion_dropout(fused)
        return self.head(fused)


def apply_parallel_augment(batch: torch.Tensor) -> torch.Tensor:
    augmented = batch + 0.012 * torch.randn_like(batch)
    gain = 1.0 + 0.08 * (torch.rand(batch.size(0), batch.size(1), 1, 1, 1, device=batch.device) - 0.5)
    augmented = augmented * gain

    max_time_mask = 24
    max_freq_mask = 18
    for slice_idx in range(batch.size(1)):
        slice_tensor = augmented[:, slice_idx]
        time_width = torch.randint(0, max_time_mask + 1, (batch.size(0),), device=batch.device)
        freq_width = torch.randint(0, max_freq_mask + 1, (batch.size(0),), device=batch.device)
        for b in range(batch.size(0)):
            if time_width[b] > 0:
                start = torch.randint(0, slice_tensor.size(-1) - time_width[b] + 1, (1,), device=batch.device).item()
                slice_tensor[b, :, :, start : start + time_width[b]] = slice_tensor[b, :, :, start : start + time_width[b]].min()
            if freq_width[b] > 0:
                start = torch.randint(0, slice_tensor.size(-2) - freq_width[b] + 1, (1,), device=batch.device).item()
                block = slice_tensor[b, :, start : start + freq_width[b], :]
                slice_tensor[b, :, start : start + freq_width[b], :] = block.mean()
    shift = torch.randint(-12, 13, (batch.size(0),), device=batch.device)
    for b in range(batch.size(0)):
        if shift[b] != 0:
            augmented[b] = torch.roll(augmented[b], shifts=int(shift[b].item()), dims=-1)
    return torch.clamp(augmented, min=-80.0, max=0.0)


def train_epoch(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    use_mixed_precision: bool,
    augment: bool,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).float()
        if augment:
            inputs = apply_parallel_augment(inputs)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_mixed_precision, dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * inputs.size(0)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5)
        # consider a sample correct if any predicted label matches any true label
        match = (preds & (targets.bool())).any(dim=1).sum().item()
        correct += match
        total += targets.size(0)
    return running_loss / total, correct / total


def evaluate(model: nn.Module, device: torch.device, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    loss_accum = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_accum += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5)
            match = (preds & (targets.bool())).any(dim=1).sum().item()
            correct += match
            total += targets.size(0)
    return loss_accum / total, correct / total


def fine_tune_model(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    use_mixed_precision: bool,
    augment: bool,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)

    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_train_loss = float("inf")
    best_train_acc = 0.0
    best_epoch = 0
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            device,
            train_loader,
            criterion,
            optimizer,
            scaler,
            use_mixed_precision,
            augment,
        )
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)

        print(
            f"FineTune Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.2f}%"
        )

        improved = val_loss + 1e-4 < best_val_loss
        if improved:
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_train_loss = train_loss
            best_train_acc = train_acc
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Fine-tune early stopping triggered.")
                break

    model.load_state_dict(best_state)
    state = best_state
    stats = {
        "best_epoch": float(best_epoch),
        "train_loss": float(best_train_loss),
        "train_acc": float(best_train_acc),
        "val_loss": float(best_val_loss),
        "val_acc": float(best_val_acc),
    }
    return state, stats


def create_dataloaders(
    dataset_path: str,
    cfg: DatasetConfig,
    cache_dir: str | None,
    batch_size: int,
    seed: int,
    feature_type: str,
    l3_config: Optional[OpenL3Config],
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], List[str]]:
    file_paths, labels, mapping = list_audio_files(dataset_path)
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        file_paths,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=0.2,
        random_state=seed,
        stratify=train_labels,
    )
    train_tracks = list(zip(train_paths, train_labels))
    val_tracks = list(zip(val_paths, val_labels))
    test_tracks = list(zip(test_paths, test_labels))
    l3_model = None
    if feature_type == "openl3":
        if l3_config is None:
            raise ValueError("OpenL3 configuration must be provided when feature_type='openl3'")
        if openl3 is None:
            raise RuntimeError("openl3 package is required for feature_type 'openl3' but is not installed")
        l3_model = load_openl3_model(l3_config)
    X_train, y_train = build_split_features(train_tracks, cfg, cache_dir, "train", feature_type, l3_config, l3_model)
    X_val, y_val = build_split_features(val_tracks, cfg, cache_dir, "val", feature_type, l3_config, l3_model)
    X_test, y_test = build_split_features(test_tracks, cfg, cache_dir, "test", feature_type, l3_config, l3_model)

    num_classes = len(mapping)
    def to_onehot(y_arr: np.ndarray) -> np.ndarray:
        out = np.zeros((y_arr.shape[0], num_classes), dtype=np.float32)
        out[np.arange(y_arr.shape[0]), y_arr] = 1.0
        return out

    y_train_oh = to_onehot(y_train)
    y_val_oh = to_onehot(y_val)
    y_test_oh = to_onehot(y_test)

    return (X_train, y_train_oh), (X_val, y_val_oh), (X_test, y_test_oh), mapping


def to_dataloader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train parallel CNN genre classifier with slice windows")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH, help="Path to genre audio dataset")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=48, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay")
    parser.add_argument("--gpu-index", type=int, help="CUDA device index to use")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--no-mixed", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--save-path", type=str, default=DEFAULT_SAVE_PATH, help="Where to store the checkpoint")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR, help="Directory for feature cache files")
    parser.add_argument("--window-duration", type=int, default=15, help="Window size in seconds for each sample")
    parser.add_argument("--slice-duration", type=int, default=5, help="Duration in seconds per slice")
    parser.add_argument("--window-stride", type=int, default=5, help="Stride in seconds between windows")
    parser.add_argument("--embedding-dim", type=int, default=160, help="Embedding dimension per slice encoder")
    parser.add_argument("--no-shared-backbone", action="store_true", help="Use independent CNN per slice")
    parser.add_argument("--feature-type", choices=("mel", "openl3"), default="mel", help="Feature extractor to use")
    parser.add_argument("--openl3-embedding-dim", type=int, default=512, help="OpenL3 embedding size (ignored for mel)")
    parser.add_argument("--openl3-content-type", type=str, default="music", help="OpenL3 content type (music/environmental)")
    parser.add_argument("--openl3-input-repr", type=str, default="mel256", help="OpenL3 input representation")
    parser.add_argument("--openl3-hop-size", type=float, default=0.5, help="Hop size in seconds for OpenL3 embeddings")
    parser.add_argument("--openl3-center", action="store_true", help="Use centered frames for OpenL3 slicing")
    parser.add_argument("--openl3-batch-size", type=int, default=32, help="Batch size for OpenL3 embedding extraction")
    parser.add_argument("--augment", action="store_true", help="Enable SpecAugment-like transformations")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--fine-tune-epochs", type=int, default=5, help="Fine-tuning epochs after main training (0 disables)")
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4, help="Learning rate for fine-tuning stage")
    parser.add_argument("--fine-tune-weight-decay", type=float, default=1e-5, help="Weight decay during fine-tuning")
    parser.add_argument("--fine-tune-patience", type=int, default=3, help="Early stopping patience for fine-tuning")
    parser.add_argument("--fine-tune-augment", action="store_true", help="Apply augmentation while fine-tuning")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = DatasetConfig(
        window_duration=args.window_duration,
        slice_duration=args.slice_duration,
        window_stride=args.window_stride,
    )

    feature_type = args.feature_type
    l3_config: Optional[OpenL3Config] = None
    if feature_type == "openl3":
        if openl3 is None:
            raise RuntimeError(
                "openl3 package is not installed. Install it or switch --feature-type back to 'mel'."
            )
        l3_config = OpenL3Config(
            embedding_dim=args.openl3_embedding_dim,
            content_type=args.openl3_content_type,
            input_repr=args.openl3_input_repr,
            center=bool(args.openl3_center),
            hop_size=float(args.openl3_hop_size),
            batch_size=int(args.openl3_batch_size),
        )
        if args.augment:
            print("ℹ SpecAugment disabled for OpenL3 features.")
            args.augment = False
        if args.fine_tune_augment:
            print("ℹ Fine-tune augmentation disabled for OpenL3 features.")
            args.fine_tune_augment = False

    cache_dir = args.cache_dir if args.cache_dir else None
    (X_train, y_train), (X_val, y_val), (X_test, y_test), mapping = create_dataloaders(
        args.dataset_path,
        cfg,
        cache_dir,
        args.batch_size,
        args.seed,
        feature_type,
        l3_config,
    )

    print(
        f"Loaded dataset with {len(mapping)} genres | feature: {feature_type} | "
        f"train samples: {X_train.shape[0]}, val: {X_val.shape[0]}, test: {X_test.shape[0]}"
    )

    train_loader = to_dataloader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader = to_dataloader(X_val, y_val, args.batch_size, shuffle=False)
    test_loader = to_dataloader(X_test, y_test, args.batch_size, shuffle=False)

    input_feature_dim: Optional[int] = None
    if feature_type == "openl3":
        input_feature_dim = int(X_train.shape[-1])

    device = choose_device(args.gpu_index, args.cpu)
    use_mixed_precision = torch.cuda.is_available() and not args.no_mixed and not args.cpu
    if use_mixed_precision:
        print("✓ Mixed precision enabled")
    else:
        print("ℹ Mixed precision disabled")

    model = ParallelCNN(
        num_slices=cfg.slices_per_window,
        num_classes=len(mapping),
        embedding_dim=args.embedding_dim,
        shared_backbone=not args.no_shared_backbone,
        feature_type=feature_type,
        input_feature_dim=input_feature_dim,
    ).to(device)
    torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            device,
            train_loader,
            criterion,
            optimizer,
            scaler,
            use_mixed_precision,
            augment=args.augment,
        )
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.2f}%"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    fine_tune_stats: Dict[str, float] | None = None
    if args.fine_tune_epochs > 0:
        print("\nStarting fine-tuning stage...")
        fine_state, fine_tune_stats = fine_tune_model(
            model,
            device,
            train_loader,
            val_loader,
            criterion,
            use_mixed_precision,
            augment=args.fine_tune_augment,
            epochs=args.fine_tune_epochs,
            lr=args.fine_tune_lr,
            weight_decay=args.fine_tune_weight_decay,
            patience=max(1, args.fine_tune_patience),
        )
        best_state = fine_state
        model.load_state_dict(best_state)
        best_val_acc = max(best_val_acc, fine_tune_stats.get("val_acc", best_val_acc))

    test_loss, test_acc = evaluate(model, device, test_loader, criterion)
    print(f"\nFinal Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:5.2f}%")

    metadata = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "val_accuracy": float(best_val_acc),
        "test_accuracy": float(test_acc),
        "window_duration": cfg.window_duration,
        "slice_duration": cfg.slice_duration,
        "window_stride": cfg.window_stride,
        "embedding_dim": args.embedding_dim,
        "shared_backbone": not args.no_shared_backbone,
        "augment": bool(args.augment),
        "multihead_heads": int(getattr(model, "mha_heads", 1)),
        "feature_type": feature_type,
    }

    if fine_tune_stats is not None:
        metadata["fine_tune"] = {
            "epochs": int(args.fine_tune_epochs),
            "learning_rate": float(args.fine_tune_lr),
            "weight_decay": float(args.fine_tune_weight_decay),
            "patience": int(max(1, args.fine_tune_patience)),
            "augment": bool(args.fine_tune_augment),
            "best_epoch": float(fine_tune_stats.get("best_epoch", 0.0)),
            "train_loss": float(fine_tune_stats.get("train_loss", 0.0)),
            "train_accuracy": float(fine_tune_stats.get("train_acc", 0.0)),
            "val_loss": float(fine_tune_stats.get("val_loss", 0.0)),
            "val_accuracy": float(fine_tune_stats.get("val_acc", 0.0)),
        }

    if l3_config is not None:
        metadata["openl3"] = {
            "embedding_dim": int(l3_config.embedding_dim),
            "content_type": l3_config.content_type,
            "input_repr": l3_config.input_repr,
            "hop_size": float(l3_config.hop_size),
            "center": bool(l3_config.center),
            "batch_size": int(l3_config.batch_size),
        }

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "mapping": mapping,
            "meta": metadata,
        },
        args.save_path,
    )
    print(f"✓ Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
