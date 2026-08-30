"""
Downsample surveillance videos to a fixed FPS and save the kept frames as JPEGs.

Walks RAW_VIDEO_DIR recursively, samples frames at TARGET_FPS, drops frames that
are too blurry or too similar to the previous frame, and writes the
survivors to

    FRAME_OUTPUT_DIR/<category>/<video_stem>/<frame_number>.jpg
"""

import os

import cv2
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import RAW_VIDEO_DIR as TRAIN_VIDEO_FOLDER
from config import FRAME_OUTPUT_DIR as TRAIN_FRAME_FOLDER

# --- Sampling ---
TARGET_FPS   = 5            # frames kept per second of source video

# --- Blur filter ---
BLUR_THRESH  = 80.0         # Laplacian variance below this -> frame skipped

# --- Duplicate removal ---
PHASH_DIFF_THRESH = 2       # hamming distance at or below this -> duplicate

# --- Output ---
JPEG_QUALITY = 95


def phash(gray, hash_size: int = 8):
    """64-bit perceptual hash of a grayscale frame.

    Resize to a thumbnail, DCT it, keep the low-frequency corner, and
    threshold at the median. Unlike a plain pixel diff this survives the
    brightness drift and compression noise you get on CCTV footage.
    """
    resized = cv2.resize(gray, (hash_size * 4, hash_size * 4),
                         interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    dct_top = dct[:hash_size, :hash_size]
    return (dct_top > np.median(dct_top)).flatten()


def phash_distance(hash_a, hash_b) -> int:
    """Hamming distance between two pHash fingerprints."""
    return int(np.sum(hash_a != hash_b))


def laplacian_blur_score(gray):
    """Higher = sharper. Below BLUR_THRESH the frame is too blurry to keep."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def process_video(video_path, frame_dir):
    """Sample one video down to TARGET_FPS and save the sharp frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return 0, 0, 0

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        # A few clips report 0 fps in their metadata; assume a sane default
        source_fps = 25.0
    frame_interval = max(1, int(source_fps / TARGET_FPS))

    saved = 0
    skipped_blur = 0
    skipped_dup = 0
    frame_count = 0
    prev_hash = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval != 0:
            frame_count += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if laplacian_blur_score(gray) < BLUR_THRESH:
            skipped_blur += 1
            frame_count += 1
            continue

        current_hash = phash(gray)
        if prev_hash is not None and phash_distance(current_hash, prev_hash) <= PHASH_DIFF_THRESH:
            skipped_dup += 1
            frame_count += 1
            continue
        prev_hash = current_hash

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        pil_frame.save(os.path.join(frame_dir, f"{frame_count}.jpg"), quality=JPEG_QUALITY)
        saved += 1
        frame_count += 1

    cap.release()
    return saved, skipped_blur, skipped_dup


def main():
    import pathlib

    root_path = pathlib.Path(TRAIN_VIDEO_FOLDER)
    total_saved = 0
    total_blur_skip = 0
    total_dup_skip = 0

    for video_path_obj in root_path.rglob("*.mp4"):
        category = video_path_obj.parent.name
        frame_dir = os.path.join(TRAIN_FRAME_FOLDER, category, video_path_obj.stem)

        # Resume-friendly: skip videos that already have frames on disk
        if os.path.exists(frame_dir) and len(os.listdir(frame_dir)) > 0:
            print(f"  [SKIP] Already done: {category}/{video_path_obj.name}")
            continue

        os.makedirs(frame_dir, exist_ok=True)
        saved, skipped_blur, skipped_dup = process_video(str(video_path_obj), frame_dir)
        total_saved += saved
        total_blur_skip += skipped_blur
        total_dup_skip += skipped_dup
        print(f"  ok {category}/{video_path_obj.name:<45} "
              f"saved={saved} blur_skip={skipped_blur} dup_skip={skipped_dup}")

    print(f"\n  Total saved       : {total_saved}")
    print(f"  Blur-skipped      : {total_blur_skip}")
    print(f"  Duplicate-skipped : {total_dup_skip}")


if __name__ == "__main__":
    main()
