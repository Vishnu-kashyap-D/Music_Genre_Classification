import argparse
import copy
import os
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from train_model_torch import (
    CNNGenreClassifier,
    choose_device,
    evaluate,
    load_dataset,
    set_seed,
    split_datasets,
    DATA_PATH,
    DEFAULT_SAVE_PATH,
    POP_HIPHOP_AUX_PATH,
)


class MelDataset(Dataset):
    """Wrap mel-spectrogram numpy arrays for optional augmentation."""

    def __init__(self, data: np.ndarray, labels: np.ndarray) -> None:
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.data.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def build_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = MelDataset(X_train, y_train)
    val_ds = MelDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def apply_light_augmentation(batch: torch.Tensor) -> torch.Tensor:
    """Apply gentle noise and gain jitter in-place for stability."""
    noise_scale = 0.01 * torch.rand(batch.size(0), 1, 1, 1, device=batch.device)
    batch = batch + noise_scale * torch.randn_like(batch)
    gain = 1.0 + 0.05 * (torch.rand(batch.size(0), 1, 1, 1, device=batch.device) - 0.5)
    batch = batch * gain
    return torch.clamp(batch, min=-80.0, max=0.0)


def finetune(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    use_mixed_precision: bool,
    augment: bool,
) -> Tuple[dict, dict]:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)

    best_val = float("inf")
    best_state: dict = {}
    best_stats: dict = {}
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if augment:
                inputs = apply_light_augmentation(inputs)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_mixed_precision, dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss_accum = 0.0
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss_accum += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(dim=1) == targets).sum().item()
                val_total += targets.size(0)

        val_loss = val_loss_accum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.2f}%"
        )

        if val_loss + 1e-4 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_stats = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Early stopping triggered.")
                break

    if not best_state:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return best_state, best_stats


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the PyTorch genre classifier with gentle updates")
    parser.add_argument("--data", type=str, default=DATA_PATH, help="Path to data.json produced by preprocess_data.py")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_SAVE_PATH, help="Existing checkpoint to fine-tune")
    parser.add_argument("--save-path", type=str, help="Where to write the updated checkpoint (defaults to --checkpoint)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Fine-tuning learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--patience", type=int, default=5, help="Epoch patience for early stopping")
    parser.add_argument("--freeze-features", action="store_true", help="Freeze convolutional feature extractor")
    parser.add_argument("--augment", action="store_true", help="Enable light mel-domain augmentation")
    parser.add_argument("--gpu-index", type=int, help="CUDA device index to use")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--skip-aux", action="store_true", help="Skip retraining the pop/hiphop auxiliary head")
    parser.add_argument("--predict-file", type=str, help="Optional audio clip to test after fine-tuning")
    args = parser.parse_args()

    set_seed(1337)

    save_path = args.save_path or args.checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    device = choose_device(args.gpu_index, args.cpu)
    use_mixed_precision = torch.cuda.is_available() and not args.cpu

    print("Loading dataset...")
    X, y, mapping = load_dataset(args.data)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_datasets(X, y)

    train_loader, val_loader = build_dataloaders(X_train, y_train, X_val, y_val, args.batch_size, shuffle=True)
    test_loader = DataLoader(MelDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False, num_workers=0)

    payload = torch.load(args.checkpoint, map_location=device)
    state_dict = payload["state_dict"]
    metadata = payload.get("meta", {})

    model = CNNGenreClassifier(num_classes=len(mapping)).to(device)
    model.load_state_dict(state_dict)

    if args.freeze_features:
        print("Freezing convolutional feature extractor layers.")
        for param in model.features.parameters():
            param.requires_grad = False

    print("Starting fine-tuning...")
    best_state, best_stats = finetune(
        model,
        device,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        use_mixed_precision=use_mixed_precision,
        augment=args.augment,
    )

    model.load_state_dict(best_state)
    test_loss, test_accuracy = evaluate(model, device, test_loader)
    print(f"\nPost fine-tune test loss: {test_loss:.4f} | accuracy: {test_accuracy*100:.2f}%")

    new_meta = metadata.copy()
    new_meta.update(
        {
            "fine_tuned_at": datetime.utcnow().isoformat() + "Z",
            "fine_tune_epochs": args.epochs,
            "fine_tune_lr": args.lr,
            "fine_tune_weight_decay": args.weight_decay,
            "fine_tune_patience": args.patience,
            "fine_tune_freeze_features": args.freeze_features,
            "fine_tune_augment": args.augment,
            "fine_tune_stats": best_stats,
            "test_accuracy_post": float(test_accuracy),
        }
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "mapping": mapping,
            "meta": new_meta,
        },
        save_path,
    )
    print(f"✓ Fine-tuned checkpoint saved to {save_path}")

    if not args.skip_aux:
        print("\nRetraining pop/hiphop auxiliary head against updated embeddings...")
        from train_model_torch import train_pop_hiphop_disambiguator

        aux_train_X = np.concatenate((X_train, X_val), axis=0)
        aux_train_y = np.concatenate((y_train, y_val), axis=0)
        train_pop_hiphop_disambiguator(aux_train_X, aux_train_y, X_test, y_test, mapping, save_path=POP_HIPHOP_AUX_PATH)

    if args.predict_file:
        from predict_genre_torch import choose_device as infer_choose_device, predict as run_inference

        infer_device = infer_choose_device(args.gpu_index, args.cpu)
        model.eval()
        run_inference(args.predict_file, model, mapping, infer_device)


if __name__ == "__main__":
    main()
