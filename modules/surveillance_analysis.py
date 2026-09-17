"""Shared video analysis pipeline for the CLI and Streamlit dashboard."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from config import (
    DEFAULT_DETECTION_THRESHOLD,
    EMAIL_STATE_FILE,
    FEATURE_DIM,
    HEATMAP_DIR,
    IMAGE_SIZE,
    OUTPUT_DIR,
    PERIODIC_REPORT_INTERVAL_SECONDS,
    REPORT_DIR,
    SEQUENCE_LENGTH,
    SNAPSHOT_DIR,
    TEMP_DIR,
    TIMELINE_CSV,
    MODEL_CHECKPOINT_PATH,
    TRAIN_OUTPUT_DIR,
)
from models.spatial_extractor import DINOv2SpatialExtractor
from models.temporal_extractor import TemporalTransformer
from modules.alerts import AlertPayload, build_alert_summary, save_snapshot
from modules.email_service import EmailSettings, send_email
from modules.gradcam import DinoV2AttentionRollout
from modules.report_generator import generate_pdf_report
from modules.timeline import TimelineEvent, export_timeline_csv, format_timestamp, merge_window_events


@dataclass
class VideoPrediction:
    start_time: float
    end_time: float
    confidence: float
    event_type: str
    mid_frame_path: Optional[str] = None


@dataclass
class SurveillanceResult:
    video_name: str
    video_path: str
    duration_seconds: float
    fps: float
    total_frames: int
    processing_time: float
    processing_fps: float
    highest_confidence: float
    average_confidence: float
    total_incidents: int
    window_predictions: List[VideoPrediction] = field(default_factory=list)
    timeline: List[TimelineEvent] = field(default_factory=list)
    annotated_video_path: Optional[str] = None
    report_path: Optional[str] = None
    timeline_csv_path: Optional[str] = None
    snapshot_paths: List[str] = field(default_factory=list)
    heatmap_raw_paths: List[str] = field(default_factory=list)
    heatmap_paths: List[str] = field(default_factory=list)
    heatmap_modes: List[str] = field(default_factory=list)
    alerts: List[AlertPayload] = field(default_factory=list)


@dataclass
class LiveMonitoringState:
    camera_source: str
    start_time: float
    total_alerts: int = 0
    highest_confidence: float = 0.0
    confidence_sum: float = 0.0
    sample_count: int = 0
    last_summary_email_time: float = 0.0
    latest_report_path: Optional[str] = None
    latest_timeline_path: Optional[str] = None
    latest_alert_timestamp: str = ""
    latest_alert_confidence: float = 0.0
    latest_status: str = "🟢 Normal"
    incidents: List[dict] = field(default_factory=list)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def average_confidence(self) -> float:
        return (self.confidence_sum / self.sample_count) if self.sample_count else 0.0


def _build_email_settings(
    email_sender: Optional[str] = None,
    email_password: Optional[str] = None,
    email_recipient: Optional[str] = None,
) -> EmailSettings:
    settings = EmailSettings()
    if email_sender:
        settings.sender = email_sender
    if email_password:
        settings.password = email_password
    if email_recipient:
        settings.recipient = email_recipient
    return settings


def preprocess_frame(frame_bgr: np.ndarray) -> Image.Image:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    width, height = image.width, image.height
    short_side = min(width, height)
    left = (width - short_side) // 2
    top = (height - short_side) // 2
    right = left + short_side
    bottom = top + short_side
    image = image.crop((left, top, right, bottom))
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return image


def _extract_state_dict(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format. Expected a state dict or a dict containing 'model_state_dict'.")


def load_models(device: Optional[torch.device] = None, checkpoint_path: Optional[str] = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spatial_model = DINOv2SpatialExtractor().to(device).eval()
    transform = spatial_model.get_preprocess_transform()

    temporal_model = TemporalTransformer().to(device).eval()
    candidate_paths = [
        checkpoint_path,
        MODEL_CHECKPOINT_PATH,
        os.path.join(TRAIN_OUTPUT_DIR, "models", "best_model.pt"),
        os.path.join(TRAIN_OUTPUT_DIR, "best_model.pt"),
    ]
    best_model_path = next((path for path in candidate_paths if path and os.path.exists(path)), None)
    if best_model_path is None:
        raise FileNotFoundError(
            "Best model not found. Checked: " + ", ".join(candidate_paths) + ". Please place the checkpoint at one of these locations or set SURVEILLANCE_MODEL_PATH."
        )

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    temporal_model.load_state_dict(_extract_state_dict(checkpoint))
    return spatial_model, temporal_model, transform


def _generate_heatmap(heatmap_engine: DinoV2AttentionRollout, frame_bgr: np.ndarray) -> tuple:
    """Return (heatmap_bgr, mode) for a single frame."""

    rgb_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return heatmap_engine.generate(rgb_image)


def _annotate_frame(
    frame_bgr: np.ndarray,
    confidence: float,
    is_violence: bool,
    heatmap_bgr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw the confidence readout, and a red border for detections, onto one frame."""

    if heatmap_bgr is not None:
        annotated = DinoV2AttentionRollout.overlay(frame_bgr, heatmap_bgr, alpha=0.70)
    else:
        annotated = frame_bgr.copy()

    text = f"Violence: {confidence:.1%}"
    cv2.putText(annotated, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
    cv2.putText(
        annotated, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        (0, 0, 255) if is_violence else (0, 255, 0), 2,
    )
    if is_violence:
        frame_height, frame_width = annotated.shape[:2]
        cv2.rectangle(annotated, (0, 0), (frame_width - 1, frame_height - 1), (0, 0, 255), 8)
    return annotated


def _window_to_prediction(start_frame: int, sequence_length: int, fps: float, confidence: float, threshold: float) -> VideoPrediction:
    start_time = start_frame / fps
    end_time = (start_frame + sequence_length) / fps
    event_type = "Violence Detected" if confidence >= threshold else "Normal"
    return VideoPrediction(start_time=start_time, end_time=end_time, confidence=confidence, event_type=event_type)


def analyze_video(
    video_path: str,
    *,
    output_video_path: Optional[str] = None,
    detection_threshold: float = DEFAULT_DETECTION_THRESHOLD,
    enable_gradcam: bool = False,
    enable_report: bool = False,
    enable_email: bool = False,
    checkpoint_path: Optional[str] = None,
    email_sender: Optional[str] = None,
    email_password: Optional[str] = None,
    email_recipient: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    alert_callback: Optional[Callable[[AlertPayload], None]] = None,
) -> SurveillanceResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spatial_model, temporal_model, transform = load_models(device, checkpoint_path=checkpoint_path)
    heatmap_engine = DinoV2AttentionRollout(spatial_model) if enable_gradcam else None
    email_settings = _build_email_settings(email_sender, email_password, email_recipient) if enable_email else None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = total_frames / fps if fps else 0.0

    output_writer = None
    if output_video_path:
        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
        output_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


    frame_buffer_bgr: List[np.ndarray] = []
    frame_buffer_preprocessed: List[Image.Image] = []
    frame_buffer_original_paths: List[str] = []
    window_predictions: List[VideoPrediction] = []
    snapshot_paths: List[str] = []
    heatmap_raw_paths: List[str] = []
    heatmap_paths: List[str] = []
    heatmap_modes: List[str] = []
    alerts: List[AlertPayload] = []
    confidences: List[float] = []
    start_time = time.time()
    window_index = 0

    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer_bgr.append(frame.copy())
        frame_buffer_preprocessed.append(preprocess_frame(frame))
        if len(frame_buffer_preprocessed) < SEQUENCE_LENGTH:
            continue

        tensor_images = torch.stack([transform(img) for img in frame_buffer_preprocessed]).to(device)
        with torch.no_grad():
            spatial_features = spatial_model(tensor_images)
            temporal_input = spatial_features.unsqueeze(0)
            output = temporal_model(temporal_input)
            violence_confidence = float(output[0, 0].item())

        confidences.append(violence_confidence)
        window_prediction = _window_to_prediction(window_index * SEQUENCE_LENGTH, SEQUENCE_LENGTH, fps, violence_confidence, detection_threshold)
        window_predictions.append(window_prediction)

        if progress_callback is not None:
            progress_callback(min((window_index + 1) * SEQUENCE_LENGTH, total_frames), total_frames, violence_confidence)

        is_violence = violence_confidence >= detection_threshold
        middle_frame = frame_buffer_bgr[len(frame_buffer_bgr) // 2].copy()
        heatmap_path = None
        snapshot_path = None

        if is_violence:
            heatmap = None
            if heatmap_engine is not None:
                heatmap, heatmap_mode = _generate_heatmap(heatmap_engine, middle_frame)

            # The heatmap is computed from the middle frame but overlaid on every
            # frame in the window, so the detected segment keeps playing at full
            # motion instead of freezing on a single still.
            if output_writer is not None:
                for bgr_frame in frame_buffer_bgr:
                    output_writer.write(_annotate_frame(bgr_frame, violence_confidence, True, heatmap))

            snapshot_name = f"snapshot_{window_index:05d}.jpg"
            snapshot_path = save_snapshot(middle_frame, SNAPSHOT_DIR, snapshot_name)
            snapshot_paths.append(snapshot_path)

            if heatmap is not None:
                heatmap_overlay = DinoV2AttentionRollout.overlay(middle_frame, heatmap, alpha=0.70)
                heatmap_path = str(Path(HEATMAP_DIR) / f"heatmap_{window_index:05d}.jpg")
                raw_heatmap_path = str(Path(HEATMAP_DIR) / f"heatmap_raw_{window_index:05d}.jpg")
                cv2.imwrite(raw_heatmap_path, heatmap)
                cv2.imwrite(heatmap_path, heatmap_overlay)
                heatmap_raw_paths.append(raw_heatmap_path)
                heatmap_paths.append(heatmap_path)
                heatmap_modes.append(heatmap_mode)

            alert = AlertPayload(
                timestamp=format_timestamp(window_prediction.start_time),
                confidence=violence_confidence,
                snapshot_path=snapshot_path,
                summary=build_alert_summary(format_timestamp(window_prediction.start_time), violence_confidence),
            )
            alerts.append(alert)
            if alert_callback is not None:
                alert_callback(alert)
            if enable_email and email_settings is not None:
                send_email(
                    subject="URGENT: Violence Detected",
                    body=f"Timestamp: {alert.timestamp}\nConfidence: {alert.confidence:.2%}\n{alert.summary}",
                    attachments=[snapshot_path] if snapshot_path else [],
                    settings=email_settings,
                )

            frame_buffer_original_paths.append(snapshot_path)

        elif output_writer is not None:
            for bgr_frame in frame_buffer_bgr:
                output_writer.write(_annotate_frame(bgr_frame, violence_confidence, False))

        frame_buffer_bgr = []
        frame_buffer_preprocessed = []
        window_index += 1

    # The final frames do not fill a whole window, so they are never scored.
    # Pass them through unscored rather than truncating the annotated output.
    if output_writer is not None and frame_buffer_bgr:
        for bgr_frame in frame_buffer_bgr:
            output_writer.write(bgr_frame)

    cap.release()
    if output_writer is not None:
        output_writer.release()

    timeline_events = merge_window_events(
        TimelineEvent(
            timestamp=format_timestamp(prediction.start_time),
            confidence=prediction.confidence,
            start_time=prediction.start_time,
            end_time=prediction.end_time,
            duration=prediction.end_time - prediction.start_time,
            event_type=prediction.event_type,
        )
        for prediction in window_predictions
    )

    timeline_csv_path = export_timeline_csv(timeline_events, TIMELINE_CSV)
    highest_confidence = max(confidences) if confidences else 0.0
    average_confidence = float(np.mean(confidences)) if confidences else 0.0
    total_incidents = sum(1 for event in timeline_events if event.event_type == "Violence Detected")
    processing_time = time.time() - start_time
    processing_fps = total_frames / processing_time if processing_time > 0 else 0.0

    result = SurveillanceResult(
        video_name=os.path.basename(video_path),
        video_path=video_path,
        duration_seconds=duration_seconds,
        fps=fps,
        total_frames=total_frames,
        processing_time=processing_time,
        processing_fps=processing_fps,
        highest_confidence=highest_confidence,
        average_confidence=average_confidence,
        total_incidents=total_incidents,
        window_predictions=window_predictions,
        timeline=timeline_events,
        annotated_video_path=output_video_path,
        timeline_csv_path=timeline_csv_path,
        snapshot_paths=snapshot_paths,
        heatmap_raw_paths=heatmap_raw_paths,
        heatmap_paths=heatmap_paths,
        heatmap_modes=heatmap_modes,
        alerts=alerts,
    )

    if enable_report:
        summary = (
            f"Processed {result.video_name} in {result.processing_time:.1f}s. "
            f"Detected {result.total_incidents} violent events with a maximum confidence of {result.highest_confidence:.2%}."
        )
        recommendations = "Review violent segments immediately and escalate to security personnel when repeat incidents are detected."
        report_path = Path(REPORT_DIR) / f"{Path(video_path).stem}_incident_report.pdf"
        result.report_path = generate_pdf_report(
            str(report_path),
            project_title="AI-Powered Smart Surveillance System",
            video_name=result.video_name,
            processing_date=time.strftime("%Y-%m-%d %H:%M:%S"),
            video_duration=result.duration_seconds,
            model_used="DINOv2 + Temporal Transformer",
            timeline=result.timeline,
            incidents=[
                {
                    "timestamp": format_timestamp(event.start_time),
                    "summary": event.event_type,
                    "snapshot_path": snapshot_paths[i] if i < len(snapshot_paths) else None,
                    "heatmap_path": heatmap_paths[i] if i < len(heatmap_paths) else None,
                }
                for i, event in enumerate(result.timeline)
                if event.event_type == "Violence Detected"
            ],
            max_confidence=result.highest_confidence,
            average_confidence=result.average_confidence,
            summary=summary,
            recommendations=recommendations,
        )

    if enable_email and email_settings is not None:
        summary_body = (
            f"Processed video: {result.video_name}\n"
            f"Duration: {result.duration_seconds:.1f}s\n"
            f"Total incidents: {result.total_incidents}\n"
            f"Highest confidence: {result.highest_confidence:.2%}\n"
            f"Average confidence: {result.average_confidence:.2%}\n"
            f"Processing FPS: {result.processing_fps:.2f}\n"
        )
        send_email(
            subject="Surveillance Summary Report",
            body=summary_body,
            attachments=[result.report_path] if result.report_path else [],
            settings=email_settings,
        )

    return result


def _build_event_timeline(window_predictions: List[VideoPrediction]) -> List[TimelineEvent]:
    return merge_window_events(
        TimelineEvent(
            timestamp=format_timestamp(prediction.start_time),
            confidence=prediction.confidence,
            start_time=prediction.start_time,
            end_time=prediction.end_time,
            duration=prediction.end_time - prediction.start_time,
            event_type=prediction.event_type,
        )
        for prediction in window_predictions
    )


def _write_live_report(state: LiveMonitoringState, output_dir: str = REPORT_DIR) -> Optional[str]:
    if state.sample_count == 0:
        return None

    timeline = []
    for incident in state.incidents:
        timeline.append(
            TimelineEvent(
                timestamp=incident.get("timestamp", ""),
                confidence=float(incident.get("confidence", 0.0)),
                start_time=float(incident.get("start_time", 0.0)),
                end_time=float(incident.get("end_time", 0.0)),
                duration=float(incident.get("duration", 0.0)),
                event_type="Violence Detected",
            )
        )

    report_path = Path(output_dir) / f"live_report_{int(time.time())}.pdf"
    generate_pdf_report(
        str(report_path),
        project_title="AI-Powered Smart Surveillance System - Live Monitoring",
        video_name=state.camera_source,
        processing_date=time.strftime("%Y-%m-%d %H:%M:%S"),
        video_duration=state.elapsed_seconds,
        model_used="DINOv2 + Temporal Transformer",
        timeline=timeline,
        incidents=state.incidents,
        max_confidence=state.highest_confidence,
        average_confidence=state.average_confidence,
        summary=(
            f"Live monitoring of {state.camera_source} for {state.elapsed_seconds:.1f}s. "
            f"Alerts: {state.total_alerts}. Highest confidence: {state.highest_confidence:.2%}."
        ),
        recommendations="Review the latest camera activity and dispatch security staff if repeated incidents occur.",
    )
    state.latest_report_path = str(report_path)
    return str(report_path)


def _send_live_alert(camera_source: str, timestamp: str, confidence: float, snapshot_path: str) -> bool:
    summary = f"Violence detected at {timestamp} from {camera_source} with confidence {confidence:.2%}."
    return send_email(
        subject="URGENT: Violence Detected",
        body=f"Timestamp: {timestamp}\nConfidence: {confidence:.2%}\nCamera source: {camera_source}\n{summary}",
        attachments=[snapshot_path],
    )


def _send_live_summary(state: LiveMonitoringState) -> bool:
    report_path = _write_live_report(state)
    body = (
        f"Monitoring duration: {state.elapsed_seconds:.1f}s\n"
        f"Total incidents: {state.total_alerts}\n"
        f"Average confidence: {state.average_confidence:.2%}\n"
        f"Highest confidence: {state.highest_confidence:.2%}\n"
        f"Latest alert time: {state.latest_alert_timestamp or 'N/A'}\n"
        f"Camera source: {state.camera_source}\n"
    )
    return send_email(
        subject="Surveillance Summary Report",
        body=body,
        attachments=[report_path] if report_path else [],
    )


def monitor_live_source(
    camera_source: str,
    *,
    detection_threshold: float = DEFAULT_DETECTION_THRESHOLD,
    enable_gradcam: bool = False,
    enable_email: bool = False,
    email_sender: Optional[str] = None,
    email_password: Optional[str] = None,
    email_recipient: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    status_callback: Optional[Callable[[dict], None]] = None,
    alert_callback: Optional[Callable[[AlertPayload], None]] = None,
    snapshot_callback: Optional[Callable[[str], None]] = None,
    frame_callback: Optional[Callable[[np.ndarray], None]] = None,
    periodic_summary_interval_seconds: int = PERIODIC_REPORT_INTERVAL_SECONDS,
    max_frames: Optional[int] = None,
) -> LiveMonitoringState:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spatial_model, temporal_model, transform = load_models(device)
    heatmap_engine = DinoV2AttentionRollout(spatial_model) if enable_gradcam else None
    email_settings = _build_email_settings(email_sender, email_password, email_recipient) if enable_email else None

    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        raise ValueError(f"Could not open live source {camera_source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    state = LiveMonitoringState(camera_source=camera_source, start_time=time.time())
    frame_buffer_bgr: List[np.ndarray] = []
    frame_buffer_preprocessed: List[Image.Image] = []
    frame_index = 0
    last_summary_time = time.time()

    try:
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            frame_buffer_bgr.append(frame.copy())
            frame_buffer_preprocessed.append(preprocess_frame(frame))
            if frame_callback is not None:
                frame_callback(frame)
            if len(frame_buffer_preprocessed) < SEQUENCE_LENGTH:
                if status_callback is not None:
                    status_callback(
                        {
                            "status": "🟢 Normal",
                            "confidence": 0.0,
                            "timestamp": format_timestamp(time.time() - state.start_time),
                            "total_alerts": state.total_alerts,
                            "elapsed_seconds": state.elapsed_seconds,
                            "fps": 0.0,
                            "latency_ms": 0.0,
                        }
                    )
                continue

            start_infer = time.time()
            tensor_images = torch.stack([transform(img) for img in frame_buffer_preprocessed]).to(device)
            with torch.no_grad():
                spatial_features = spatial_model(tensor_images)
                temporal_input = spatial_features.unsqueeze(0)
                output = temporal_model(temporal_input)
                violence_confidence = float(output[0, 0].item())
            latency_ms = (time.time() - start_infer) * 1000.0

            state.sample_count += 1
            state.confidence_sum += violence_confidence
            state.highest_confidence = max(state.highest_confidence, violence_confidence)

            is_violence = violence_confidence >= detection_threshold
            timestamp_text = format_timestamp(time.time() - state.start_time)
            state.latest_alert_timestamp = timestamp_text
            state.latest_alert_confidence = violence_confidence
            state.latest_status = "🔴 Violence Detected" if is_violence else "🟢 Normal"

            if status_callback is not None:
                status_callback(
                    {
                        "status": state.latest_status,
                        "confidence": violence_confidence,
                        "timestamp": timestamp_text,
                        "total_alerts": state.total_alerts,
                        "elapsed_seconds": state.elapsed_seconds,
                        "fps": fps,
                        "latency_ms": latency_ms,
                    }
                )

            if progress_callback is not None:
                progress_callback(frame_index, frame_index, violence_confidence)

            if is_violence:
                state.total_alerts += 1
                middle_frame = frame_buffer_bgr[len(frame_buffer_bgr) // 2].copy()
                snapshot_name = f"live_snapshot_{state.total_alerts:05d}.jpg"
                snapshot_path = save_snapshot(middle_frame, SNAPSHOT_DIR, snapshot_name)
                if snapshot_callback is not None:
                    snapshot_callback(snapshot_path)

                if heatmap_engine is not None:
                    heatmap, _ = _generate_heatmap(heatmap_engine, middle_frame)
                    heatmap_overlay = DinoV2AttentionRollout.overlay(middle_frame, heatmap, alpha=0.70)
                    cv2.imwrite(str(Path(HEATMAP_DIR) / f"live_heatmap_{state.total_alerts:05d}.jpg"), heatmap_overlay)

                alert = AlertPayload(
                    timestamp=timestamp_text,
                    confidence=violence_confidence,
                    snapshot_path=snapshot_path,
                    summary=f"Violence detected from {camera_source} at {timestamp_text} with confidence {violence_confidence:.2%}.",
                )
                state.incidents.append(
                    {
                        "timestamp": timestamp_text,
                        "confidence": violence_confidence,
                        "start_time": time.time() - state.start_time,
                        "end_time": time.time() - state.start_time,
                        "duration": 0.0,
                        "camera_source": camera_source,
                        "snapshot_path": snapshot_path,
                    }
                )
                if alert_callback is not None:
                    alert_callback(alert)
                if enable_email and email_settings is not None:
                    send_email(
                        subject="URGENT: Violence Detected",
                        body=f"Timestamp: {timestamp_text}\nConfidence: {violence_confidence:.2%}\nCamera source: {camera_source}\n{alert.summary}",
                        attachments=[snapshot_path],
                        settings=email_settings,
                    )

            if time.time() - last_summary_time >= periodic_summary_interval_seconds:
                if enable_email and email_settings is not None:
                    report_path = _write_live_report(state)
                    send_email(
                        subject="Surveillance Summary Report",
                        body=(
                            f"Monitoring duration: {state.elapsed_seconds:.1f}s\n"
                            f"Total incidents: {state.total_alerts}\n"
                            f"Average confidence: {state.average_confidence:.2%}\n"
                            f"Highest confidence: {state.highest_confidence:.2%}\n"
                            f"Latest alert time: {state.latest_alert_timestamp or 'N/A'}\n"
                            f"Camera source: {state.camera_source}\n"
                        ),
                        attachments=[report_path] if report_path else [],
                        settings=email_settings,
                    )
                last_summary_time = time.time()

            frame_buffer_bgr = []
            frame_buffer_preprocessed = []

    finally:
        cap.release()

    return state


def maybe_send_periodic_report(result: SurveillanceResult, pdf_path: Optional[str] = None) -> bool:
    """Send a periodic report if the configured interval has elapsed."""

    state_path = Path(EMAIL_STATE_FILE)
    last_sent = 0.0
    if state_path.exists():
        try:
            last_sent = float(json.loads(state_path.read_text(encoding="utf-8")).get("last_sent", 0.0))
        except Exception:
            last_sent = 0.0

    if time.time() - last_sent < PERIODIC_REPORT_INTERVAL_SECONDS:
        return False

    attachments = [pdf_path] if pdf_path else []
    body = (
        f"Summary for {result.video_name}\n\n"
        f"Incidents: {result.total_incidents}\n"
        f"Highest confidence: {result.highest_confidence:.2%}\n"
        f"Average confidence: {result.average_confidence:.2%}\n"
        f"Processing FPS: {result.processing_fps:.2f}\n"
    )
    sent = send_email("Surveillance Summary Report", body, attachments=attachments)
    if sent:
        state_path.write_text(json.dumps({"last_sent": time.time()}), encoding="utf-8")
    return sent
