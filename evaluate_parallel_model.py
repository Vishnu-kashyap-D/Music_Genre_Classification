import argparse
import os

import numpy as np
import torch
from typing import Optional

from train_parallel_cnn import (
    DatasetConfig,
    OpenL3Config,
    ParallelCNN,
    compute_windows_for_track,
    choose_device,
    load_openl3_model,
)


def load_checkpoint(
    model_path: str,
    device: torch.device,
) -> tuple[ParallelCNN, list[str], DatasetConfig, str, Optional[OpenL3Config], Optional[object]]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at {model_path}")

    payload = torch.load(model_path, map_location=device)
    mapping = payload["mapping"]
    meta = payload.get("meta", {})

    cfg = DatasetConfig(
        window_duration=int(meta.get("window_duration", 15)),
        slice_duration=int(meta.get("slice_duration", 5)),
        window_stride=int(meta.get("window_stride", 5)),
    )

    feature_type = str(meta.get("feature_type", "mel"))
    l3_config: Optional[OpenL3Config] = None
    l3_model = None
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
        l3_model = load_openl3_model(l3_config)

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
    return model, mapping, cfg, feature_type, l3_config, l3_model


def predict(
    file_path: str,
    model: ParallelCNN,
    mapping: list[str],
    cfg: DatasetConfig,
    device: torch.device,
    feature_type: str,
    l3_config: Optional[OpenL3Config],
    l3_model,
    *,
    clip_start: Optional[float] = None,
    clip_duration: Optional[float] = None,
    max_windows: Optional[int] = None,
) -> None:
    windows = compute_windows_for_track(
        file_path,
        cfg,
        feature_type,
        l3_config,
        l3_model,
        offset=clip_start,
        duration=clip_duration,
        max_windows=max_windows,
    )
    if not windows:
        print("No valid windows extracted; ensure the clip is long enough.")
        return

    tensor = torch.from_numpy(np.stack(windows)).to(device)
    with torch.inference_mode():
        outputs = model(tensor)
        probabilities = torch.sigmoid(outputs).cpu().numpy()

    averaged = probabilities.mean(axis=0)
    top_indices = np.argsort(averaged)[::-1]
    predicted_indices = np.where(averaged >= 0.5)[0]

    print(f"\n--- Parallel CNN Prediction for: {os.path.basename(file_path)} ---")
    if predicted_indices.size:
        print("Predicted labels (threshold 0.5):")
        for idx in predicted_indices:
            print(f"  {mapping[idx].capitalize()}: {averaged[idx]*100:.2f}%")
    else:
        print("No labels passed the 0.5 threshold.")

    print("Top-3 probabilities:")
    for idx in top_indices[:3]:
        print(f"  {mapping[idx].capitalize()}: {averaged[idx]*100:.2f}%")
    print("------------------------------------------")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with the parallel CNN model")
    parser.add_argument("audio_path", type=str, help="Path to audio file for evaluation")
    parser.add_argument("--model", type=str, default=os.path.join("torch_models", "parallel_genre_classifier_torch.pt"))
    parser.add_argument("--gpu-index", type=int, help="CUDA device index")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--clip-start", type=float, help="Start time (in seconds) to begin analyzing the audio")
    parser.add_argument("--clip-duration", type=float, help="Duration (in seconds) of audio to analyze")
    parser.add_argument(
        "--max-windows",
        type=int,
        help="Limit the number of model windows to evaluate (each window is cfg.window_duration seconds)",
    )
    args = parser.parse_args()

    device = choose_device(args.gpu_index, args.cpu)
    model, mapping, cfg, feature_type, l3_config, l3_model = load_checkpoint(args.model, device)
    predict(
        args.audio_path,
        model,
        mapping,
        cfg,
        device,
        feature_type,
        l3_config,
        l3_model,
        clip_start=args.clip_start,
        clip_duration=args.clip_duration,
        max_windows=args.max_windows,
    )


if __name__ == "__main__":
    main()
