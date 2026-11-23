import argparse
import csv
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from train_parallel_cnn import (
    DatasetConfig,
    OpenL3Config,
    ParallelCNN,
    choose_device,
    compute_windows_for_track,
    load_openl3_model,
)


def resolve_audio_path(
    raw_path: str,
    csv_path: Path,
    root_dir: Path | None,
) -> Path | None:
    candidate = Path(raw_path)
    search_roots = []

    if candidate.is_absolute() and candidate.exists():
        return candidate

    if candidate.is_absolute():
        search_roots.append(candidate.parent)
        if root_dir is not None:
            search_roots.append(root_dir)
    else:
        if root_dir is not None:
            search_roots.append(root_dir)
        search_roots.append(csv_path.parent)

    for base in search_roots:
        resolved = (base / candidate.name) if candidate.is_absolute() else (base / candidate)
        if resolved.exists():
            return resolved

    return None


def load_checkpoint(
    model_path: str,
    device: torch.device,
) -> tuple[ParallelCNN, list[str], DatasetConfig, str, Optional[OpenL3Config], Optional[object]]:
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


def evaluate_dataset(
    csv_path: Path,
    model: ParallelCNN,
    mapping: list[str],
    cfg: DatasetConfig,
    device: torch.device,
    root_dir: Path | None,
    feature_type: str,
    l3_config: Optional[OpenL3Config],
    l3_model,
) -> None:
    total = 0
    correct = 0
    per_genre_total: dict[str, int] = {genre: 0 for genre in mapping}
    per_genre_correct: dict[str, int] = {genre: 0 for genre in mapping}

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV must contain headers")

        field_map: dict[str, str] = {}
        for field in reader.fieldnames:
            cleaned = field.strip().lower().lstrip("\ufeff")
            if cleaned:
                field_map[cleaned] = field

        if "file" not in field_map or "label" not in field_map:
            raise ValueError("CSV must contain 'file' and 'label' columns")

        file_key = field_map["file"]
        label_key = field_map["label"]
        mapping_lc = {g.lower(): g for g in mapping}
        for row in reader:
            raw_path = row[file_key].strip().strip('"').strip("'")
            label = row[label_key].strip()
            if not raw_path:
                continue
            resolved = resolve_audio_path(raw_path, csv_path, root_dir)
            if resolved is None:
                print(f"Skipping {raw_path}: file not found; consider using --root to specify a base directory")
                continue

            windows = compute_windows_for_track(str(resolved), cfg, feature_type, l3_config, l3_model)
            if not windows:
                print(f"Skipping {resolved}: no valid windows extracted")
                continue

            tensor = torch.from_numpy(np.stack(windows)).to(device)
            with torch.inference_mode():
                outputs = model(tensor)
                probabilities = torch.sigmoid(outputs).mean(dim=0).cpu().numpy()

            # parse multi-label ground truth from CSV (split on '|') and map to known mapping
            tokens = [t.strip().lower() for t in label.split("|") if t.strip()]
            gt_indices = set()
            for t in tokens:
                if t in mapping_lc:
                    gt_indices.add(mapping.index(mapping_lc[t]))
            if not gt_indices:
                # nothing we can evaluate against in mapping
                print(f"Skipping {resolved}: label '{label}' contains no known genres")
                continue

            total += 1
            for idx in gt_indices:
                per_genre_total[mapping[idx]] = per_genre_total.get(mapping[idx], 0) + 1

            # predicted labels by threshold
            threshold = 0.5
            pred_indices = set(np.where(probabilities >= threshold)[0].tolist())
            # if no predictions above threshold, fallback to top-1
            if not pred_indices:
                pred_indices = {int(np.argmax(probabilities))}

            if pred_indices & gt_indices:
                correct += 1
                for idx in gt_indices:
                    per_genre_correct[mapping[idx]] = per_genre_correct.get(mapping[idx], 0) + 1

    if total == 0:
        print("No valid audio files evaluated.")
        return

    accuracy = correct / total
    print(f"Overall accuracy: {accuracy*100:.2f}% ({correct}/{total})")

    print("\nPer-genre accuracy:")
    for genre, count in sorted(per_genre_total.items()):
        correct_count = per_genre_correct.get(genre, 0)
        if count == 0:
            print(f"  {genre}: no samples")
        else:
            print(f"  {genre}: {correct_count/count*100:.2f}% ({correct_count}/{count})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate parallel CNN on a custom CSV dataset")
    parser.add_argument("csv", type=str, help="CSV file with 'file' and 'label' columns")
    parser.add_argument("--model", type=str, default=os.path.join("torch_models", "parallel_genre_classifier_torch.pt"))
    parser.add_argument("--root", type=str, help="Optional base directory to resolve audio paths")
    parser.add_argument("--gpu-index", type=int, help="CUDA device index")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    device = choose_device(args.gpu_index, args.cpu)
    model, mapping, cfg, feature_type, l3_config, l3_model = load_checkpoint(args.model, device)
    root_dir = Path(args.root).resolve() if args.root else None
    evaluate_dataset(Path(args.csv), model, mapping, cfg, device, root_dir, feature_type, l3_config, l3_model)


if __name__ == "__main__":
    main()
