"""
Convert images to NumPy arrays for training.

Fixes vs original:
  - Center-crop instead of top-left crop (original cropped wrong region)
  - Parallel image loading with ThreadPoolExecutor (much faster for 100k+ frames)
  - Normalises to float32 [0-1] here so feature extractor doesn't need to
  - Cleaner chunk progress reporting
"""

import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

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

NUM_WORKERS = 8   # parallel image loading threads

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


def load_chunk_parallel(paths: list) -> np.ndarray:
    """Load a list of image paths in parallel using threads."""
    images = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(preprocess_image, p): i for i, p in enumerate(paths)}
        for future in as_completed(futures):
            idx = futures[future]
            images[idx] = future.result()
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
        chunk_images = load_chunk_parallel(chunk_paths)

        np.save(os.path.join(img_out_dir, f"{chunk_idx}.npy"), chunk_images)
        np.save(os.path.join(lbl_out_dir, f"{chunk_idx}.npy"), chunk_labels)
        print(f"    saved — images {chunk_images.shape}, labels {chunk_labels.shape}")


def main():
    random.seed(RANDOM_SEED)

    # 1. Discover all video directories and assign video-level labels
    all_videos = []
    print(f"Scanning extracted frames in {FRAME_OUTPUT_DIR}...")
    for category in sorted(os.listdir(FRAME_OUTPUT_DIR)):
        cat_path = os.path.join(FRAME_OUTPUT_DIR, category)
        if not os.path.isdir(cat_path):
            continue
            
        # Normal category = 0, Anomaly categories = 1
        is_anomaly = "Normal" not in category
        label = 1 if is_anomaly else 0
        
        for video_dir in sorted(os.listdir(cat_path)):
            video_path = os.path.join(cat_path, video_dir)
            if os.path.isdir(video_path):
                # Store (full_path, video_dir_name, label)
                all_videos.append((video_path, video_dir, label))

    print(f"Found {len(all_videos)} total videos.")
    if len(all_videos) == 0:
        print("Error: No processed video directories found. Did you run step 1 (conv_video_to_img.py)?")
        return

    # 2. Randomly split videos at the video level (80/20)
    all_videos = sorted(all_videos)  # Sort for deterministic shuffle
    random.shuffle(all_videos)
    
    train_videos, test_videos = train_test_split(
        all_videos, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_SEED
    )
    print(f"Split: {len(train_videos)} train videos | {len(test_videos)} test videos")

    # Helper function to load frame paths
    def collect_frames_for_videos(video_list):
        img_paths = []
        labels = []
        new_idx = []
        for video_path, _, label in video_list:
            image_files = sorted(
                [f for f in os.listdir(video_path) if f.endswith('.jpg')],
                key=lambda x: int(os.path.splitext(x)[0])
            )
            if not image_files:
                continue
                
            new_idx.append(len(img_paths))
            for image_file in image_files:
                img_paths.append(os.path.join(video_path, image_file))
                labels.append(label)
        return img_paths, labels, new_idx

    # 3. Build train/test index lists
    print("Building train dataset index...")
    train_img_paths, train_labels, train_new_idx = collect_frames_for_videos(train_videos)
    print("Building test dataset index...")
    test_img_paths, test_labels, test_new_idx = collect_frames_for_videos(test_videos)

    print(f"\nTrain samples (frames): {len(train_img_paths)} | Test samples (frames): {len(test_img_paths)}")

    os.makedirs(NPY_DATA_DIR, exist_ok=True)

    # 4. Process and save chunks
    if train_img_paths:
        save_chunks(train_img_paths, train_labels, "train")
        np.save(os.path.join(NPY_DATA_DIR, "train_images_new_idx.npy"), np.array(train_new_idx))
        
    if test_img_paths:
        save_chunks(test_img_paths, test_labels, "test")
        np.save(os.path.join(NPY_DATA_DIR, "test_images_new_idx.npy"), np.array(test_new_idx))
        
    print("\nDone!")


if __name__ == "__main__":
    main()