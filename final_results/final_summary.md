
# Final Experimental Results

**Model**: torch_models/parallel_genre_classifier_torch.pt
**Evaluation Level**: Song-Level Aggregation (Mean Pooling)

## Overall Metrics
| Metric | Value |
| :--- | :--- |
| **Accuracy** | 0.8191 |
| **Macro F1** | 0.8116 |
| **Micro F1** | 0.8191 |

## Per-Class F1 (Comparison Targets)
| Genre | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| Pop | 0.75 | 0.90 | 0.82 |
| Hip-Hop | 0.68 | 0.95 | 0.79 |
| Rock | 0.67 | 0.50 | 0.57 |
| Metal | 0.91 | 1.00 | 0.95 |
| Disco | 0.90 | 0.90 | 0.90 |

## Confusion Matrix Highlights
*   **Pop/Hip-Hop Confusion**: 2 Pop misclassified as Hip-Hop.
*   **Rock/Metal Confusion**: 2 Rock misclassified as Metal.
