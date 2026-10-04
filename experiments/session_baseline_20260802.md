# Session Baseline Experiment - 2026-08-02

## Objective

Measure whether the six-step classifier generalizes across the two newly
captured sessions without allowing adjacent images from the same session to
leak across training and validation.

## Dataset

| Session | Images | Role in experiment |
| --- | ---: | --- |
| `noon_session` | 207 | Train once, validate once |
| `night_session` | 206 | Train once, validate once |

Every session contains all six classes with 34-35 images per class. The
session manifests include SHA-256 hashes, and no exact duplicate was found
across the two splits.

## Training configuration

- Architecture: MobileNetV2 with a six-class classifier head
- Class order: `Chk1`, `Chk2`, `Chk3`, `Chk4`, `Chk5`, `Chk6`
- Initialization: origami feature weights from `origami_model_best.pth`
- Frozen backbone: yes; 329,478 of 2,553,350 parameters trainable
- Epochs: 12
- Batch size: 16
- Learning rate: 0.001
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau
- Seed: 42

## Results

| Direction | Best epoch | Accuracy | Macro F1 | Best checkpoint |
| --- | ---: | ---: | ---: | --- |
| Train noon, validate night | 7 | 43.69% | 42.62% | `Checkpoints/session_baselines/noon_train_night_val/origami_step6_best.pth` |
| Train night, validate noon | 4 | **49.76%** | **46.00%** | `Checkpoints/session_baselines/night_train_noon_val/origami_step6_best.pth` |

### Per-class recall

| Class | Noon to night | Night to noon |
| --- | ---: | ---: |
| Chk1 | 41.18% | **88.57%** |
| Chk2 | **44.12%** | 8.82% |
| Chk3 | 58.82% | **74.29%** |
| Chk4 | **73.53%** | 23.53% |
| Chk5 | 5.71% | **31.43%** |
| Chk6 | 40.00% | **70.59%** |

## Confusion matrices

Rows are actual classes and columns are predictions in `Chk1..Chk6` order.

### Train noon, validate night

```text
Chk1  14 13  1  4  2  0
Chk2   2 15  7  8  2  0
Chk3   2  6 20  5  0  1
Chk4   3  3  0 25  2  1
Chk5   0  3  1 26  2  3
Chk6   0  4  1 14  2 14
```

The largest failure is `Chk5 -> Chk4` (26 of 35 images). The model trained on
noon images does not reliably preserve the head-fold distinction under the
night session appearance.

### Train night, validate noon

```text
Chk1  31  0  1  3  0  0
Chk2   8  3 16  3  1  3
Chk3   4  2 26  1  2  0
Chk4   2  2  6  8 13  3
Chk5   4  3  5  7 11  5
Chk6   0  1  2  4  3 24
```

This direction is more balanced overall, but `Chk2 -> Chk3` (16 of 34) and
the two-way `Chk4/Chk5` confusion remain severe.

## Decision

For a temporary webcam smoke test, use:

```text
Checkpoints/session_baselines/night_train_noon_val/origami_step6_best.pth
```

This checkpoint is the stronger of the two cross-session baselines. Do not
replace the default web checkpoint permanently yet: 49.76% cross-session
accuracy is evidence that two sessions are insufficient for a reliable guided
flow.

## Next data priorities

1. Capture Session 3 in a new environment and keep it in the training pool.
2. Increase shape and camera-angle variety for Chk2, especially the boundary
   between Kite Folds and Bird Base.
3. Add clear oblique views for Chk4 and Chk5 so the straight neck tip versus
   folded head is visible.
4. After Session 3, train on S1+S2 and validate on S3. Do not use random image
   splitting.
5. Reserve later independent sessions for validation and final test according
   to the five-session plan.

## Artifacts

- `data/step6/session_baselines/*/session_manifest.csv`
- `data/step6/session_baselines/*/session_summary.json`
- `plots/session_baselines/*/step6_training_history.png`
- `plots/session_baselines/*/step6_val_report.json`
- `plots/session_baselines/*/step6_val_confusion_matrix.png`
