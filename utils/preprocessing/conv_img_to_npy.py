"""
Convert extracted frames into chunked NumPy arrays for training.

Frames are cropped square, resized to IMAGE_SIZE, normalised to float32
[0-1], and written out in NPY_CHUNK_SIZE-sized .npy chunks so the feature
extractor can stream them instead of holding the dataset in memory.
"""

import os
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    FRAME_OUTPUT_DIR,
    NPY_DATA_DIR, IMAGE_SIZE, NPY_CHUNK_SIZE,
    TEST_SPLIT_RATIO, RANDOM_SEED,
)

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load, center-crop, resize to IMAGE_SIZE × IMAGE_SIZE.
    """
    image = Image.open(image_path).convert("RGB")
    w, h = image.width, image.height

    # Center crop to square
    short_side = min(w, h)
    left   = (w - short_side) // 2
    top    = (h - short_side) // 2
    right  = left + short_side
    bottom = top  + short_side
    image  = image.crop((left, top, right, bottom))

    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.array(image, dtype=np.float32) / 255.0   # [0, 1]


def load_chunk(paths: list) -> np.ndarray:
    """Load a list of image paths one after another."""
    images = [preprocess_image(path) for path in paths]
    return np.array(images, dtype=np.float32)


def save_chunks(img_paths, labels, split_name):
    """Save image + label arrays in NPY_CHUNK_SIZE chunks."""
    img_out_dir = os.path.join(NPY_DATA_DIR, f"{split_name}_img")
    lbl_out_dir = os.path.join(NPY_DATA_DIR, f"{split_name}_lbl")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    n_chunks = (len(img_paths) + NPY_CHUNK_SIZE - 1) // NPY_CHUNK_SIZE
    for chunk_idx in range(n_chunks):
        start = chunk_idx * NPY_CHUNK_SIZE
        end   = start + NPY_CHUNK_SIZE
        chunk_paths  = img_paths[start:end]
        chunk_labels = np.array(labels[start:end], dtype=np.int64)

        print(f"  [{split_name}] chunk {chunk_idx+1}/{n_chunks} — loading {len(chunk_paths)} images...")
        chunk_images = load_chunk(chunk_paths)

        np.save(os.path.join(img_out_dir, f"{chunk_idx}.npy"), chunk_images)
        np.save(os.path.join(lbl_out_dir, f"{chunk_idx}.npy"), chunk_labels)
        print(f"    saved — images {chunk_images.shape}, labels {chunk_labels.shape}")


def main():
    random.seed(RANDOM_SEED)

    # 1. Discover every extracted frame and label it from its category
    all_img_paths = []
    all_labels = []
    print(f"Scanning extracted frames in {FRAME_OUTPUT_DIR}...")
    for category in sorted(os.listdir(FRAME_OUTPUT_DIR)):
        cat_path = os.path.join(FRAME_OUTPUT_DIR, category)
        if not os.path.isdir(cat_path):
            continue

        # Normal category = 0, Anomaly categories = 1
        label = 0 if "Normal" in category else 1

        for video_dir in sorted(os.listdir(cat_path)):
            video_path = os.path.join(cat_path, video_dir)
            if not os.path.isdir(video_path):
                continue
            image_files = sorted(
                [f for f in os.listdir(video_path) if f.endswith('.jpg')],
                key=lambda x: int(os.path.splitext(x)[0])
            )
            for image_file in image_files:
                all_img_paths.append(os.path.join(video_path, image_file))
                all_labels.append(label)

    print(f"Found {len(all_img_paths)} total frames.")
    if len(all_img_paths) == 0:
        print("Error: No processed frames found. Did you run step 1 (conv_video_to_img.py)?")
        return

    # 2. Shuffle and split the frames 80/20
    train_img_paths, test_img_paths, train_labels, test_labels = train_test_split(
        all_img_paths, all_labels, test_size=TEST_SPLIT_RATIO,
        random_state=RANDOM_SEED, shuffle=True
    )

    print(f"\nTrain samples (frames): {len(train_img_paths)} | Test samples (frames): {len(test_img_paths)}")

    os.makedirs(NPY_DATA_DIR, exist_ok=True)

    # 4. Process and save chunks
    if train_img_paths:
        save_chunks(train_img_paths, train_labels, "train")
        
    if test_img_paths:
        save_chunks(test_img_paths, test_labels, "test")
        
    print("\nDone!")


if __name__ == "__main__":
    main()