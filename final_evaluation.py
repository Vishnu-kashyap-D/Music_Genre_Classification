import argparse
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from torch.utils.data import DataLoader

# Import architecture and config from training script
from train_parallel_cnn import (
    ParallelCNN, 
    DatasetConfig, 
    OpenL3Config, 
    choose_device, 
    create_dataloaders, 
    to_dataloader
)

def evaluate_song_level(model, device, loader):
    model.eval()
    
    all_probs = []
    all_targets = []
    all_track_ids = []

    with torch.inference_mode():
        for batch in loader:
            if len(batch) == 3:
                inputs, targets, track_ids = batch
            else:
                inputs, targets = batch
                track_ids = torch.zeros(targets.size(0), dtype=torch.long)

            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu()
            targets = targets.cpu()
            
            all_probs.append(probs)
            all_targets.append(targets)
            all_track_ids.append(track_ids)

    full_probs = torch.cat(all_probs)
    full_targets = torch.cat(all_targets)
    full_track_ids = torch.cat(all_track_ids)
    
    unique_tracks = torch.unique(full_track_ids)
    
    y_true = []
    y_pred = []
    
    for t_id in unique_tracks:
        mask = (full_track_ids == t_id)
        if not mask.any(): continue
        
        # Aggregate logic
        track_probs = full_probs[mask]
        avg_prob = track_probs.mean(dim=0)
        
        # Ground truth (assuming one-hot, take argmax)
        track_target = full_targets[mask][0]
        
        pred_label = avg_prob.argmax().item()
        true_label = track_target.argmax().item()
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        
    return np.array(y_true), np.array(y_pred)

def main():
    parser = argparse.ArgumentParser(description="Generate Final Paper Artifacts for MIR Model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to frozen .pt checkpoint")
    parser.add_argument("--dataset-path", type=str, default="Data/genres_original")
    parser.add_argument("--output-dir", type=str, default="final_results")
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = choose_device(args.gpu_index, False)
    
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    meta = checkpoint["meta"]
    mapping = checkpoint["mapping"]
    
    # Reconstruct Configs
    cfg = DatasetConfig(
        window_duration=meta["window_duration"],
        slice_duration=meta["slice_duration"],
        window_stride=meta["window_stride"]
    )
    
    l3_config = None
    if "openl3" in meta:
        l3_conf = meta["openl3"]
        l3_config = OpenL3Config(
            embedding_dim=l3_conf["embedding_dim"],
            content_type=l3_conf["content_type"],
            input_repr=l3_conf["input_repr"],
            hop_size=l3_conf["hop_size"],
            center=l3_conf["center"],
            batch_size=l3_conf["batch_size"]
        )
    
    # Reconstruct Model
    # Important: Check if input_feature_dim stored in separate meta key or infer from weights
    input_feature_dim = meta.get("input_feature_dim")
    # If not explicitly in meta (might happen if trained with older script logic), infer from OpenL3 setup
    # If not explicitly in meta (might happen if trained with older script logic), infer from OpenL3 setup
    if meta["feature_type"] == "openl3" and input_feature_dim is None:
         # Standard OpenL3 Mel256 embedding dim is 512, but input to encoder depends on OpenL3 config.
         # Actually EmbeddingSliceEncoder takes `input_dim`.
         # Let's inspect the weights in state_dict.
         # The backbone is `backbone` in shared mode, or `branches.0` in independent.
         key = "backbone.net.1.weight" if meta["shared_backbone"] else "branches.0.net.1.weight"
         if key in state_dict:
             input_feature_dim = state_dict[key].shape[1]
         else:
             # Fallback if layer norm is first
             key_ln = "backbone.net.0.weight" if meta["shared_backbone"] else "branches.0.net.0.weight"
             if key_ln in state_dict:
                 input_feature_dim = state_dict[key_ln].shape[0]
             else:
                 print("Warning: Could not infer input_feature_dim from weights. Assuming 512 (default).")
                 input_feature_dim = 512

    model = ParallelCNN(
        num_slices=cfg.slices_per_window,
        num_classes=len(mapping),
        embedding_dim=meta["embedding_dim"],
        shared_backbone=meta["shared_backbone"],
        feature_type=meta["feature_type"],
        input_feature_dim=input_feature_dim
    ).to(device)
    
    model.load_state_dict(state_dict)
    
    print("Loading Test Data...")
    # Note: We reuse create_dataloaders but only need test set
    # Using a fixed seed ensures reproduction if the original split code is used
    (X_train, y_train, t_ids_train), (X_val, y_val, t_ids_val), (X_test, y_test, t_ids_test), _ = create_dataloaders(
        args.dataset_path, cfg, "torch_cache/parallel", 1, 1337, meta["feature_type"], l3_config
    )
    
    test_loader = to_dataloader(X_test, y_test, t_ids_test, batch_size=32, shuffle=False)
    
    print("Running Final Song-Level Evaluation...")
    y_true, y_pred = evaluate_song_level(model, device, test_loader)
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    
    print(f"\n=== FINAL RESULTS (Song-Level) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=mapping, columns=mapping)
    cm_file = os.path.join(args.output_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_file)
    print(f"\nConfusion Matrix saved to {cm_file}")
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=mapping, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_file = os.path.join(args.output_dir, "classification_report.csv")
    report_df.to_csv(report_file)
    print(f"Classification Report saved to {report_file}")
    
    # Markdown Summary
    md_summary = f"""
# Final Experimental Results

**Model**: {args.model_path}
**Evaluation Level**: Song-Level Aggregation (Mean Pooling)

## Overall Metrics
| Metric | Value |
| :--- | :--- |
| **Accuracy** | {acc:.4f} |
| **Macro F1** | {macro_f1:.4f} |
| **Micro F1** | {micro_f1:.4f} |

## Per-Class F1 (Comparison Targets)
| Genre | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| Pop | {report.get('pop', {}).get('precision', 0):.2f} | {report.get('pop', {}).get('recall', 0):.2f} | {report.get('pop', {}).get('f1-score', 0):.2f} |
| Hip-Hop | {report.get('hiphop', {}).get('precision', 0):.2f} | {report.get('hiphop', {}).get('recall', 0):.2f} | {report.get('hiphop', {}).get('f1-score', 0):.2f} |
| Rock | {report.get('rock', {}).get('precision', 0):.2f} | {report.get('rock', {}).get('recall', 0):.2f} | {report.get('rock', {}).get('f1-score', 0):.2f} |
| Metal | {report.get('metal', {}).get('precision', 0):.2f} | {report.get('metal', {}).get('recall', 0):.2f} | {report.get('metal', {}).get('f1-score', 0):.2f} |
| Disco | {report.get('disco', {}).get('precision', 0):.2f} | {report.get('disco', {}).get('recall', 0):.2f} | {report.get('disco', {}).get('f1-score', 0):.2f} |

## Confusion Matrix Highlights
*   **Pop/Hip-Hop Confusion**: {cm[mapping.index('pop'), mapping.index('hiphop')] if 'pop' in mapping and 'hiphop' in mapping else 'N/A'} Pop misclassified as Hip-Hop.
*   **Rock/Metal Confusion**: {cm[mapping.index('rock'), mapping.index('metal')] if 'rock' in mapping and 'metal' in mapping else 'N/A'} Rock misclassified as Metal.
"""
    
    md_file = os.path.join(args.output_dir, "final_summary.md")
    with open(md_file, "w") as f:
        f.write(md_summary)
    print(f"Markdown Summary saved to {md_file}")

if __name__ == "__main__":
    main()
