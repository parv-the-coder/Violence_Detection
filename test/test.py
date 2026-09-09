"""
PyTorch End-to-End Evaluation Script.

Evaluates the trained Temporal Transformer classifier combined with the
DINOv2 spatial extractor on the preprocessed test frames. Unlike
train/train.py, which scores pre-extracted features, this runs the full
spatial + temporal pipeline so the reported latency reflects deployment.
"""

import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_CHECKPOINT_PATH,
    NPY_DATA_DIR,
    SEQUENCE_LENGTH,
    TRAIN_OUTPUT_DIR,
)
from models.spatial_extractor import DINOv2SpatialExtractor
from models.temporal_extractor import TemporalTransformer


def resolve_checkpoint() -> str:
    """Return the first checkpoint that exists, mirroring the inference engine."""

    candidate_paths = [
        MODEL_CHECKPOINT_PATH,
        os.path.join(TRAIN_OUTPUT_DIR, "models", "best_model.pt"),
        os.path.join(TRAIN_OUTPUT_DIR, "best_model.pt"),
    ]
    for path in candidate_paths:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No checkpoint found. Checked: " + ", ".join(str(p) for p in candidate_paths)
    )


def crosses_video_boundary(start_idx: int, end_idx: int, new_idx_array: np.ndarray) -> bool:
    """True if the window [start_idx, end_idx) spans two different videos.

    Indices are global across the whole split, matching how
    conv_img_to_npy.py records video start offsets.
    """
    return bool(np.any((new_idx_array > start_idx) & (new_idx_array < end_idx)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading DINOv2 spatial extractor...")
    spatial_model = DINOv2SpatialExtractor().to(device).eval()

    # conv_img_to_npy.py already wrote float32 [0, 1] arrays, so only ImageNet
    # normalisation is applied here -- the PIL-oriented transform from the
    # extractor would try to re-run ToTensor on an existing tensor.
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    print("Loading Temporal Transformer...")
    temporal_model = TemporalTransformer().to(device)
    checkpoint_path = resolve_checkpoint()
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    temporal_model.load_state_dict(state_dict)
    temporal_model.eval()

    test_images_dir = os.path.join(NPY_DATA_DIR, "test_img")
    test_labels_dir = os.path.join(NPY_DATA_DIR, "test_lbl")
    test_new_idx = np.load(os.path.join(NPY_DATA_DIR, "test_images_new_idx.npy"), allow_pickle=True)

    chunk_files = sorted(
        (f for f in os.listdir(test_images_dir) if f.endswith(".npy")),
        key=lambda name: int(os.path.splitext(name)[0]),
    )
    if not chunk_files:
        print(f"Error: no .npy chunks found in {test_images_dir}")
        return

    all_outputs = []
    all_labels = []
    total_correct = 0
    total_samples = 0
    spatial_times = []
    temporal_times = []

    # Frames are numbered continuously across chunks, so track the running
    # offset to compare against the global video-boundary indices.
    global_offset = 0

    print("Starting evaluation...")
    with torch.no_grad():
        for chunk_file in chunk_files:
            images = np.load(os.path.join(test_images_dir, chunk_file))
            labels = np.load(os.path.join(test_labels_dir, chunk_file))

            for step in range(0, len(images) - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH):
                start_idx = step
                end_idx = step + SEQUENCE_LENGTH

                if crosses_video_boundary(global_offset + start_idx, global_offset + end_idx, test_new_idx):
                    continue

                window_images = images[start_idx:end_idx]
                target_label = int(labels[end_idx - 1])

                # 1. Spatial extraction over the whole window
                t0 = time.time()
                tensor_images = torch.from_numpy(window_images).permute(0, 3, 1, 2).float()
                tensor_images = torch.stack([normalize(img) for img in tensor_images]).to(device)
                spatial_features = spatial_model(tensor_images)  # (SEQUENCE_LENGTH, 768)
                spatial_times.append(time.time() - t0)

                # 2. Temporal classification
                t1 = time.time()
                output = temporal_model(spatial_features.unsqueeze(0))  # (1, 2)
                temporal_times.append(time.time() - t1)

                prob_violence = output[0, 0].item()
                predicted_label = 1 if prob_violence > 0.5 else 0

                all_outputs.append(prob_violence)
                all_labels.append(target_label)
                total_correct += int(predicted_label == target_label)
                total_samples += 1

                if total_samples % 100 == 0:
                    print(f"Processed {total_samples} windows. Current Acc: {total_correct / total_samples:.4f}")

            global_offset += len(images)

    if total_samples == 0:
        print("No samples processed.")
        return

    accuracy = total_correct / total_samples
    try:
        roc_auc = roc_auc_score(all_labels, all_outputs)
    except ValueError:
        roc_auc = 0.0
    cm = confusion_matrix(all_labels, [1 if x > 0.5 else 0 for x in all_outputs])

    print("\n" + "=" * 40)
    print("FINAL EVALUATION RESULTS")
    print("=" * 40)
    print(f"Total Samples: {total_samples}")
    print(f"Accuracy:      {accuracy:.4f}")
    print(f"ROC AUC:       {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("-" * 40)
    print(f"LATENCY (per {SEQUENCE_LENGTH}-frame window)")
    print(f"Spatial Extraction:  {np.mean(spatial_times) * 1000:.2f} ms")
    print(f"Temporal Classifier: {np.mean(temporal_times) * 1000:.2f} ms")
    print(f"Total Pipeline:      {(np.mean(spatial_times) + np.mean(temporal_times)) * 1000:.2f} ms")
    print("=" * 40)


if __name__ == "__main__":
    main()
