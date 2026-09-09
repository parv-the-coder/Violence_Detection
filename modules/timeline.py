"""Timeline utilities for surveillance event aggregation and export."""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List


@dataclass
class TimelineEvent:
    timestamp: str
    confidence: float
    start_time: float
    end_time: float
    duration: float
    event_type: str


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds_remainder:05.2f}"


def merge_window_events(events: Iterable[TimelineEvent]) -> List[TimelineEvent]:
    merged: List[TimelineEvent] = []

    for event in events:
        if not merged:
            merged.append(event)
            continue

        previous = merged[-1]
        if previous.event_type == event.event_type == "Violence Detected" and event.start_time <= previous.end_time:
            previous.end_time = max(previous.end_time, event.end_time)
            previous.duration = previous.end_time - previous.start_time
            previous.confidence = max(previous.confidence, event.confidence)
            previous.timestamp = format_timestamp(previous.start_time)
            continue

        merged.append(event)

    return merged


def export_timeline_csv(events: Iterable[TimelineEvent], output_path: str) -> str:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["timestamp", "confidence", "start_time", "end_time", "duration", "event_type"]
    with output_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))

    return str(output_file)
