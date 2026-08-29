import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import cv2
from collections import defaultdict

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_VIDEO_DIR

ROOT = RAW_VIDEO_DIR

NUM_WORKERS = min(12, cpu_count())

# ==========================


def process_video(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()

    if fps <= 0:
        duration = 0
    else:
        duration = frames / fps

    cls = video_path.parent.name

    return {
        "class": cls,
        "duration": duration,
        "fps": fps,
        "frames": frames,
    }


def collect_videos(root):
    videos = []

    for part in [
        "Anomaly-Videos-Part-1",
        "Anomaly-Videos-Part-2",
        "Anomaly-Videos-Part-3",
        "Anomaly-Videos-Part-4",
    ]:
        folder = Path(root) / part

        if not folder.exists():
            continue

        videos.extend(folder.rglob("*.mp4"))

    return videos


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60

    return f"{h:02d}:{m:02d}:{s:05.2f}"


def main():
    videos = collect_videos(ROOT)

    print(f"\nFound {len(videos)} videos\n")

    stats = defaultdict(list)

    with Pool(NUM_WORKERS) as pool:
        for result in pool.imap_unordered(process_video, videos):

            if result is None:
                continue

            stats[result["class"]].append(result)

    print("=" * 90)
    print(
        f"{'Class':20}"
        f"{'Videos':>8}"
        f"{'Avg Time':>15}"
        f"{'Total Time':>18}"
        f"{'Avg FPS':>12}"
        f"{'Avg Frames':>15}"
    )
    print("=" * 90)

    grand_total = 0

    for cls in sorted(stats):

        items = stats[cls]

        total_duration = sum(x["duration"] for x in items)
        avg_duration = total_duration / len(items)

        avg_fps = sum(x["fps"] for x in items) / len(items)
        avg_frames = sum(x["frames"] for x in items) / len(items)

        grand_total += total_duration

        print(
            f"{cls:20}"
            f"{len(items):8d}"
            f"{format_time(avg_duration):>15}"
            f"{format_time(total_duration):>18}"
            f"{avg_fps:12.2f}"
            f"{avg_frames:15.0f}"
        )

    print("=" * 90)
    print("Overall Video Time:", format_time(grand_total))


if __name__ == "__main__":
    main()