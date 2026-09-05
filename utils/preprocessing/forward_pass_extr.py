"""
Forward Pass Feature Extraction using DINOv2 (PyTorch).

Fixes vs original:
  - Batched inference (BATCH_SIZE=64) instead of one image at a time
    → ~60× faster on GPU (570k frames in minutes instead of hours)
  - Images already normalised to [0-1] by conv_img_to_npy, so only
    ImageNet mean/std normalisation is applied here (no /255 duplication)
  - tqdm progress bar per chunk with ETA
  - Graceful fallback to CPU with a clear warning
"""

import os
import sys

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    NPY_DATA_DIR, FEATURE_OUTPUT_DIR,
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, RANDOM_SEED,
)
from models.spatial_extractor import DINOv2SpatialExtractor

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 64   # increase to 128 if your GPU has >8GB VRAM
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def get_transform():
    """ImageNet normalisation only — images are already [0, 1] from conv_img_to_npy."""
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def extract_features(images_dir: str, labels_dir: str, split_name: str,
                     model, device, transform) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract DINOv2 features from all numpy image chunks using batched inference.

    Returns:
        features : np.ndarray of shape (N, 768)
        labels   : np.ndarray of shape (N,)
    """
    chunk_files = sorted(
        f for f in os.listdir(images_dir) if f.endswith(".npy")
    )
    n_chunks = len(chunk_files)

    all_features = []
    all_labels   = []

    for chunk_idx, chunk_file in enumerate(chunk_files):
        images = np.load(os.path.join(images_dir, chunk_file))   # (N, H, W, C) float32 [0-1]
        labels = np.load(os.path.join(labels_dir, chunk_file))

        print(f"[{split_name}] chunk {chunk_idx+1}/{n_chunks} — {len(images)} images")

        chunk_features = []

        # ── Batched inference ─────────────────────────────────────────────
        for batch_start in tqdm(range(0, len(images), BATCH_SIZE), desc=f"  chunk {chunk_idx+1}"):
            batch_np = images[batch_start : batch_start + BATCH_SIZE]

            # (B, H, W, C) → (B, C, H, W), float32 [0-1]
            batch_tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2).float()

            # Apply ImageNet normalisation to each image in the batch
            batch_tensor = torch.stack([transform(img) for img in batch_tensor])
            batch_tensor = batch_tensor.to(device)   # (B, 3, 224, 224)

            with torch.no_grad():
                features = model(batch_tensor)        # (B, 768)

            chunk_features.append(features.cpu().numpy())

        all_features.append(np.concatenate(chunk_features, axis=0))
        all_labels.append(labels)

    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[WARN] CUDA not available — running on CPU. This will be slow.")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    print("Loading DINOv2...")
    model = DINOv2SpatialExtractor().to(device).eval()
    transform = get_transform()
    os.makedirs(FEATURE_OUTPUT_DIR, exist_ok=True)

    # ── Train ──────────────────────────────────────────────────────────────
    train_features, train_labels = extract_features(
        images_dir  = os.path.join(NPY_DATA_DIR, "train_img"),
        labels_dir  = os.path.join(NPY_DATA_DIR, "train_lbl"),
        split_name  = "train",
        model=model, device=device, transform=transform,
    )
    np.save(os.path.join(FEATURE_OUTPUT_DIR, "train.npy"),     train_features)
    np.save(os.path.join(FEATURE_OUTPUT_DIR, "train_lbl.npy"), train_labels)
    print(f"Train features: {train_features.shape}")

    train_idx = np.load(os.path.join(NPY_DATA_DIR, "train_images_new_idx.npy"))
    np.save(os.path.join(FEATURE_OUTPUT_DIR, "train_images_new_idx.npy"), train_idx)

    # ── Test ───────────────────────────────────────────────────────────────
    test_features, test_labels = extract_features(
        images_dir  = os.path.join(NPY_DATA_DIR, "test_img"),
        labels_dir  = os.path.join(NPY_DATA_DIR, "test_lbl"),
        split_name  = "test",
        model=model, device=device, transform=transform,
    )
    np.save(os.path.join(FEATURE_OUTPUT_DIR, "test.npy"),     test_features)
    np.save(os.path.join(FEATURE_OUTPUT_DIR, "test_lbl.npy"), test_labels)
    print(f"Test features: {test_features.shape}")

    test_idx = np.load(os.path.join(NPY_DATA_DIR, "test_images_new_idx.npy"))
    np.save(os.path.join(FEATURE_OUTPUT_DIR, "test_images_new_idx.npy"), test_idx)

    print("\nFeature extraction complete!")


if __name__ == "__main__":
    main()