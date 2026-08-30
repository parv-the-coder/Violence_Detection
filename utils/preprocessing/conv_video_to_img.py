"""
Downsample surveillance videos to a fixed FPS and save the kept frames as JPEGs.

Walks RAW_VIDEO_DIR recursively, samples frames at TARGET_FPS, drops frames that
are too blurry to be useful, and writes the survivors to

    FRAME_OUTPUT_DIR/<category>/<video_stem>/<frame_number>.jpg
"""

import os

import cv2
from PIL import Image

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import RAW_VIDEO_DIR as TRAIN_VIDEO_FOLDER
from config import FRAME_OUTPUT_DIR as TRAIN_FRAME_FOLDER

# --- Sampling ---
TARGET_FPS   = 5            # frames kept per second of source video

# --- Blur filter ---
BLUR_THRESH  = 80.0         # Laplacian variance below this -> frame skipped

# --- Output ---
JPEG_QUALITY = 95


def laplacian_blur_score(gray):
    """Higher = sharper. Below BLUR_THRESH the frame is too blurry to keep."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def process_video(video_path, frame_dir):
    """Sample one video down to TARGET_FPS and save the sharp frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return 0, 0

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        # A few clips report 0 fps in their metadata; assume a sane default
        source_fps = 25.0
    frame_interval = max(1, int(source_fps / TARGET_FPS))

    saved = 0
    skipped_blur = 0
    frame_count = 0

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

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        pil_frame.save(os.path.join(frame_dir, f"{frame_count}.jpg"), quality=JPEG_QUALITY)
        saved += 1
        frame_count += 1

    cap.release()
    return saved, skipped_blur


def main():
    import pathlib

    root_path = pathlib.Path(TRAIN_VIDEO_FOLDER)
    total_saved = 0
    total_blur_skip = 0

    for video_path_obj in root_path.rglob("*.mp4"):
        category = video_path_obj.parent.name
        frame_dir = os.path.join(TRAIN_FRAME_FOLDER, category, video_path_obj.stem)

        # Resume-friendly: skip videos that already have frames on disk
        if os.path.exists(frame_dir) and len(os.listdir(frame_dir)) > 0:
            print(f"  [SKIP] Already done: {category}/{video_path_obj.name}")
            continue

        os.makedirs(frame_dir, exist_ok=True)
        saved, skipped_blur = process_video(str(video_path_obj), frame_dir)
        total_saved += saved
        total_blur_skip += skipped_blur
        print(f"  ok {category}/{video_path_obj.name:<45} saved={saved} blur_skip={skipped_blur}")

    print(f"\n  Total saved  : {total_saved}")
    print(f"  Blur-skipped : {total_blur_skip}")


if __name__ == "__main__":
    main()
