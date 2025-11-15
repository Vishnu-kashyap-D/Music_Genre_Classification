import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import f1_score

from train_parallel_cnn import DatasetConfig, OpenL3Config, ParallelCNN, choose_device


def load_checkpoint(
    model_path: str,
    device: torch.device,
) -> tuple[ParallelCNN, list[str], DatasetConfig, str, Optional[OpenL3Config]]:
    payload = torch.load(model_path, map_location="cpu")
    mapping = payload["mapping"]
    meta = payload.get("meta", {})

    cfg = DatasetConfig(
        window_duration=int(meta.get("window_duration", 15)),
        slice_duration=int(meta.get("slice_duration", 5)),
        window_stride=int(meta.get("window_stride", 5)),
    )

    feature_type = str(meta.get("feature_type", "mel"))
    l3_config: Optional[OpenL3Config] = None
    input_feature_dim: Optional[int] = None
    if feature_type == "openl3":
        openl3_meta = meta.get("openl3", {})
        l3_config = OpenL3Config(
            embedding_dim=int(openl3_meta.get("embedding_dim", 512)),
            content_type=str(openl3_meta.get("content_type", "music")),
            input_repr=str(openl3_meta.get("input_repr", "mel256")),
            center=bool(openl3_meta.get("center", False)),
            hop_size=float(openl3_meta.get("hop_size", 0.5)),
            batch_size=int(openl3_meta.get("batch_size", 32)),
        )
        input_feature_dim = l3_config.embedding_dim

    model = ParallelCNN(
        num_slices=cfg.slices_per_window,
        num_classes=len(mapping),
        embedding_dim=int(meta.get("embedding_dim", 160)),
        shared_backbone=bool(meta.get("shared_backbone", True)),
        feature_type=feature_type,
        input_feature_dim=input_feature_dim,
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, mapping, cfg, feature_type, l3_config


def load_split(
    cache_dir: Path,
    split: str,
    cfg: DatasetConfig,
    feature_type: str,
    l3_config: Optional[OpenL3Config],
) -> tuple[np.ndarray, np.ndarray]:
    suffix = f"{feature_type}_{cfg.window_duration}s_{cfg.slice_duration}s_stride{cfg.window_stride}"
    if feature_type == "openl3" and l3_config is not None:
        suffix += f"_{l3_config.cache_tag()}"
    cache_path = cache_dir / f"parallel_{split}_{suffix}.npz"
    if not cache_path.exists():
        legacy = cache_dir / (
            f"parallel_{split}_{cfg.window_duration}s_{cfg.slice_duration}s_stride{cfg.window_stride}.npz"
        )
        if legacy.exists():
            cache_path = legacy
        else:
            raise FileNotFoundError(f"Cache file not found: {cache_path}")
    data = np.load(cache_path)
    return data["X"], data["y"]


def evaluate_split(model: ParallelCNN, device: torch.device, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    batch_size = 128
    total = X.shape[0]
    correct = 0
    preds: list[int] = []
    targets: list[int] = y.tolist()

    with torch.inference_mode():
        for start in range(0, total, batch_size):
            batch = torch.from_numpy(X[start : start + batch_size]).to(device)
            outputs = model(batch)
            probs = torch.sigmoid(outputs).cpu().numpy()
            predicted = probs.argmax(axis=1)
            preds.extend(predicted.tolist())
            correct += (predicted == y[start : start + batch_size]).sum()

    accuracy = correct / total
    f1 = f1_score(targets, preds, average="macro")
    return accuracy, f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute accuracy/F1 on cached parallel CNN splits")
    parser.add_argument("--cache-dir", type=str, default=os.path.join("torch_cache", "parallel_main"))
    parser.add_argument("--model", type=str, default=os.path.join("torch_models", "parallel_genre_classifier_torch.pt"))
    parser.add_argument("--gpu-index", type=int, help="CUDA device index")
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation")
    args = parser.parse_args()

    device = choose_device(args.gpu_index, args.cpu)
    model, _, cfg, feature_type, l3_config = load_checkpoint(args.model, device)

    cache_dir = Path(args.cache_dir)
    results = {}
    for split in ("train", "val", "test"):
        X, y = load_split(cache_dir, split, cfg, feature_type, l3_config)
        accuracy, f1 = evaluate_split(model, device, X, y)
        results[split] = (accuracy, f1)

    for split, (accuracy, f1) in results.items():
        print(f"{split.capitalize()} accuracy: {accuracy*100:.2f}% | F1 (macro): {f1:.4f}")


if __name__ == "__main__":
    main()
