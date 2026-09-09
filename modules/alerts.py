"""Alert helpers for high-confidence violence detections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class AlertPayload:
    timestamp: str
    confidence: float
    snapshot_path: str
    summary: str


def save_snapshot(frame_bgr: np.ndarray, output_dir: str, filename: str) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_path / filename
    cv2.imwrite(str(snapshot_path), frame_bgr)
    return str(snapshot_path)


def build_alert_summary(timestamp: str, confidence: float) -> str:
    return f"Violence detected at {timestamp} with confidence {confidence:.2%}."
