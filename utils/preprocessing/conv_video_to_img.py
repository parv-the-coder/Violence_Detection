"""
Downsample MP4 videos to adaptive FPS, remove duplicate frames,
and process multiple videos in parallel across CPU cores.

Improvements over previous version:
  1. Adaptive FPS   — measures per-video motion level (optical flow magnitude)
                      and picks FPS_LOW (2-3) for static scenes or
                      FPS_HIGH (5-10) for high-motion scenes automatically.
  2. Duplicate removal — after FPS sampling, skips frames that are too
                      similar to the previous saved frame using two strategies:
                        a) Perceptual hash (pHash) — fast, O(1), good default
                        b) SSIM — more accurate, ~10× slower, opt-in via flag
  3. Parallel processing — all videos are processed simultaneously using
                      ProcessPoolExecutor across NUM_WORKERS CPU cores.
                      Each worker is independent (no shared state), so
                      multiprocessing (not threading) is used to bypass GIL.

Previous fixes retained:
  - BGR→RGB conversion (OpenCV reads BGR, PIL/DINOv2 expect RGB)
  - Safe frame_interval guard (handles 0-fps or low-fps source videos)
  - Laplacian blur check (skips frames too blurry to be useful)
  - JPEG quality=95 (less compression loss than default 75)
  - Resume-friendly (skips already-processed video folders)
"""

import os
import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from skimage.metrics import structural_similarity as ssim_fn

# ── Config ────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import RAW_VIDEO_DIR as TRAIN_VIDEO_FOLDER
from config import FRAME_OUTPUT_DIR as TRAIN_FRAME_FOLDER

# --- Target Frames ---
# Video Duration	Recommended Frames
# <10 sec	        14
# 10–30 sec	        25 (20-30)
# 30–60 sec	        35 (30-40)
# >60 sec	        70
def get_target_frame_count(duration_sec):
    if duration_sec < 10: return 14
    elif duration_sec <= 30: return 25
    elif duration_sec <= 60: return 35
    else: return 70

# --- Blur filter ---
BLUR_THRESH        = 80.0         # Laplacian variance below this → frame skipped

# --- Duplicate removal ---
DEDUP_METHOD       = "phash"      # "phash" (fast) or "ssim" (accurate, slower)
PHASH_DIFF_THRESH  = 2            # Lower threshold: remove only obvious duplicates
SSIM_DIFF_THRESH   = 0.97         # SSIM similarity — higher = stricter (0.0-1.0)

# --- Parallel processing ---
NUM_WORKERS        = max(1, os.cpu_count() - 1)   # leave 1 core for OS

# --- Output ---
JPEG_QUALITY       = 95
# ─────────────────────────────────────────────────────────────────────────────


# (Adaptive FPS logic removed in favor of duration-based targets)


# ── 2. Duplicate removal helpers ──────────────────────────────────────────────

def phash(gray: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """
    Perceptual hash (pHash) of a grayscale frame.

    How it works:
      1. Resize to (hash_size*4) × (hash_size*4) — tiny thumbnail
      2. Apply DCT (discrete cosine transform) — captures low-freq structure
      3. Take top-left hash_size×hash_size of DCT — most significant frequencies
      4. Threshold at median → 64-bit binary fingerprint

    Two nearly identical frames → hamming distance close to 0.
    Two different frames        → hamming distance > PHASH_DIFF_THRESH.

    Why pHash over pixel diff?
      - Robust to minor brightness/contrast shifts (common in CCTV)
      - O(1) comparison (just XOR two 64-bit arrays)
      - Not fooled by compression noise
    """
    resized = cv2.resize(gray, (hash_size * 4, hash_size * 4),
                         interpolation=cv2.INTER_AREA).astype(np.float32)
    dct     = cv2.dct(resized)
    dct_top = dct[:hash_size, :hash_size]
    median  = np.median(dct_top)
    return (dct_top > median).flatten()   # bool array of 64 bits


def phash_distance(h1: np.ndarray, h2: np.ndarray) -> int:
    """Hamming distance between two pHash fingerprints."""
    return int(np.sum(h1 != h2))


def is_duplicate_phash(gray: np.ndarray, prev_hash) -> tuple[bool, np.ndarray]:
    """Returns (is_duplicate, new_hash)."""
    current_hash = phash(gray)
    if prev_hash is None:
        return False, current_hash
    dist = phash_distance(current_hash, prev_hash)
    return dist <= PHASH_DIFF_THRESH, current_hash


def is_duplicate_ssim(gray: np.ndarray, prev_gray: np.ndarray | None) -> bool:
    """
    Returns True if the frame is too similar to the previous saved frame.

    SSIM (Structural Similarity Index) measures luminance, contrast, and
    structure simultaneously. More perceptually accurate than pHash but ~10×
    slower per comparison. Use when duplicate threshold tuning matters more
    than speed.
    """
    if prev_gray is None:
        return False
    # Resize for speed — SSIM on 224×224 is fast enough
    a = cv2.resize(gray,      (224, 224))
    b = cv2.resize(prev_gray, (224, 224))
    score = ssim_fn(a, b, data_range=255)
    return score >= SSIM_DIFF_THRESH


# ── 3. Blur helper ────────────────────────────────────────────────────────────

def laplacian_blur_score(gray: np.ndarray) -> float:
    """Higher = sharper. Below BLUR_THRESH → frame is too blurry."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ── 4. Core per-video function (runs in worker process) ───────────────────────

def process_video(args: tuple) -> dict:
    """
    Process a single video: adaptive FPS → blur filter → dedup → save.

    Designed to be called inside a worker process (no shared state).
    All config constants are module-level so they are inherited by fork.

    Args:
        args: (video_path, frame_dir, category, video_file)

    Returns:
        dict with stats for the main process to aggregate.
    """
    video_path, frame_dir, category, video_file = args

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "video": f"{category}/{video_file}",
            "saved": 0, "skipped_blur": 0,
            "skipped_dup": 0, "total_read": 0,
            "fps_used": 0, "motion_level": 0.0,
            "error": "Cannot open video",
        }

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0:
        source_fps = 25.0
    
    duration = total_frames / source_fps if source_fps > 0 else 0
    target_count = get_target_frame_count(duration)

    saved         = 0
    skipped_blur  = 0
    skipped_dup   = 0
    frame_count   = 0
    prev_hash     = None
    prev_gray_saved = None
    
    valid_frames = []

    # Pass 1: Collect valid frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur filter
        if laplacian_blur_score(gray) < BLUR_THRESH:
            skipped_blur += 1
            frame_count += 1
            continue

        # Duplicate filter
        if DEDUP_METHOD == "phash":
            is_dup, current_hash = is_duplicate_phash(gray, prev_hash)
            if is_dup:
                skipped_dup += 1
                frame_count += 1
                continue
            prev_hash = current_hash
        else:
            is_dup = is_duplicate_ssim(gray, prev_gray_saved)
            if is_dup:
                skipped_dup += 1
                frame_count += 1
                continue
            prev_gray_saved = gray

        # Keep frame in memory (RGB format to save conversion time later)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        valid_frames.append((frame_count, frame_rgb))
        frame_count += 1

    cap.release()
    
    # Pass 2: Uniformly sample target_count frames to guarantee frame limits
    if len(valid_frames) == 0:
        return {
            "video": f"{category}/{video_file}",
            "saved": 0, "skipped_blur": skipped_blur, "skipped_dup": skipped_dup,
            "total_read": frame_count, "fps_used": 0, "motion_level": 0.0,
            "error": "No valid frames found",
        }
        
    # Uniform sample
    if len(valid_frames) <= target_count:
        sampled_frames = valid_frames
    else:
        indices = np.linspace(0, len(valid_frames) - 1, target_count, dtype=int)
        sampled_frames = [valid_frames[i] for i in indices]

    for idx, (orig_frame_num, frame_rgb) in enumerate(sampled_frames):
        pil_frame = Image.fromarray(frame_rgb)
        pil_frame.save(
            os.path.join(frame_dir, f"{orig_frame_num}.jpg"),
            quality=JPEG_QUALITY
        )
        saved += 1

    return {
        "video":        f"{category}/{video_file}",
        "saved":        saved,
        "skipped_blur": skipped_blur,
        "skipped_dup":  skipped_dup,
        "total_read":   frame_count,
        "fps_used":     0,
        "motion_level": duration,  # repurposed to show duration in logs
        "error":        None,
    }


# ── 5. Main — collect jobs, run in parallel ───────────────────────────────────

def collect_jobs() -> list[tuple]:
    """Walk TRAIN_VIDEO_FOLDER recursively and build the list of (video_path, frame_dir, ...) tuples."""
    import pathlib
    jobs = []
    root_path = pathlib.Path(TRAIN_VIDEO_FOLDER)
    
    for video_path_obj in root_path.rglob("*.mp4"):
        video_path = str(video_path_obj)
        category = video_path_obj.parent.name
        video_file = video_path_obj.name
        video_stem = video_path_obj.stem
        
        frame_dir = os.path.join(TRAIN_FRAME_FOLDER, category, video_stem)

        # Resume-friendly: skip already-processed
        if os.path.exists(frame_dir) and len(os.listdir(frame_dir)) > 0:
            print(f"  [SKIP] Already done: {category}/{video_file}")
            continue

        os.makedirs(frame_dir, exist_ok=True)
        jobs.append((video_path, frame_dir, category, video_file))

    return jobs


def main():
    jobs = collect_jobs()
    if not jobs:
        print("Nothing to process.")
        return

    print(f"\nProcessing {len(jobs)} videos on {NUM_WORKERS} workers "
          f"(dedup={DEDUP_METHOD}, thresh={PHASH_DIFF_THRESH})\n")

    total_saved      = 0
    total_blur_skip  = 0
    total_dup_skip   = 0
    errors           = []

    # ProcessPoolExecutor forks a fresh Python process per worker.
    # Each worker calls process_video() independently — no shared memory,
    # no GIL contention, safe for OpenCV which is not thread-safe.
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(process_video, job): job for job in jobs}

        for future in as_completed(futures):
            stats = future.result()

            if stats["error"]:
                errors.append(stats)
                print(f"  [ERROR] {stats['video']} — {stats['error']}")
                continue

            total_saved     += stats["saved"]
            total_blur_skip += stats["skipped_blur"]
            total_dup_skip  += stats["skipped_dup"]

            print(
                f"  ✓ {stats['video']:<45} "
                f"dur={stats['motion_level']:.1f}s  "
                f"saved={stats['saved']}  "
                f"blur_skip={stats['skipped_blur']}  "
                f"dup_skip={stats['skipped_dup']}"
            )

    print(f"\n{'─'*60}")
    print(f"  Total saved       : {total_saved}")
    print(f"  Blur-skipped      : {total_blur_skip}")
    print(f"  Duplicate-skipped : {total_dup_skip}")
    print(f"  Errors            : {len(errors)}")
    print(f"{'─'*60}")


if __name__ == "__main__":
    # Required on Windows / macOS (spawn start method needs __main__ guard)
    main()