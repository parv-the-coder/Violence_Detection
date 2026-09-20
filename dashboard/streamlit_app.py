"""Streamlit dashboard for the AI-powered smart surveillance system."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_DETECTION_THRESHOLD, EMAIL_SENDER, MODEL_CHECKPOINT_PATH, OUTPUT_DIR, TEMP_DIR, ensure_runtime_dirs
from modules.email_service import load_saved_recipient, save_recipient
from modules.surveillance_analysis import analyze_video, monitor_live_source
from utils.charts import confidence_histogram, confidence_line_chart, create_speed_figure, incident_distribution_chart


def _format_seconds(seconds: float) -> str:
    minutes, remaining = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{remaining:05.2f}"


def _save_upload(uploaded_file) -> str:
    ensure_runtime_dirs()
    destination = Path(TEMP_DIR) / uploaded_file.name
    destination.write_bytes(uploaded_file.getbuffer())
    return str(destination)


def _transcode_for_browser(input_path: str) -> str:
    """Convert a video to H.264/AAC MP4 so Streamlit can play it reliably."""

    input_file = Path(input_path)
    browser_ready_path = input_file.with_name(f"{input_file.stem}_browser.mp4")

    ffmpeg_exists = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True).returncode == 0
    if not ffmpeg_exists:
        return input_path

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_file),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(browser_ready_path),
    ]

    subprocess.run(command, check=True, capture_output=True)
    return str(browser_ready_path)


def main() -> None:
    st.set_page_config(page_title="Smart Surveillance Dashboard", page_icon="🎥", layout="wide")
    ensure_runtime_dirs()

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; }
        .metric-card {
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
            color: white;
            padding: 1rem;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.06);
        }
        .alert-banner {
            background: #7f1d1d;
            color: white;
            padding: 1rem 1.2rem;
            border-radius: 12px;
            border: 1px solid #ef4444;
            margin-bottom: 1rem;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("AI-Powered Smart Surveillance System")
    st.caption("Upload a surveillance video to run the existing DINOv2 + Temporal Transformer pipeline with dashboards, alerts, reports, and exports.")

    with st.sidebar:
        st.header("Controls")
        input_source = st.radio("Input Source", ["Upload Video", "Live Webcam", "RTSP/IP Camera"], index=0)
        uploaded_file = None
        rtsp_url = ""

        if input_source == "Upload Video":
            uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
        elif input_source == "RTSP/IP Camera":
            rtsp_url = st.text_input("RTSP/IP stream URL", value="rtsp://")

        detection_threshold = st.slider("Detection threshold", 0.0, 1.0, float(DEFAULT_DETECTION_THRESHOLD), 0.01)
        enable_gradcam = st.checkbox("Enable Grad-CAM / Attention Rollout", value=True)
        enable_report = st.checkbox("Enable Incident Report", value=True)
        enable_email = st.checkbox("Enable Email Alerts", value=False)
        process_button = st.button("Run Surveillance Analysis", type="primary")

        st.markdown("---")
        st.subheader("Email Settings")
        admin_email = st.text_input("Recipient Email", value=load_saved_recipient())
        if st.button("Save Settings"):
            save_recipient(admin_email)
            st.success("Recipient saved.")
        if enable_email and not EMAIL_SENDER:
            st.warning("No sender account configured. Set SURVEILLANCE_EMAIL_SENDER and SURVEILLANCE_EMAIL_PASSWORD in your .env file.")

    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None

    if process_button and uploaded_file is not None:
        video_path = _save_upload(uploaded_file)
        output_video_path = str(Path(OUTPUT_DIR) / f"annotated_{Path(uploaded_file.name).stem}.mp4")
        browser_video_path = output_video_path
        browser_original_path = video_path

        if not Path(MODEL_CHECKPOINT_PATH).exists():
            st.error(
                f"Missing model checkpoint at {MODEL_CHECKPOINT_PATH}. Place your trained .pt file there, "
                "or point SURVEILLANCE_MODEL_PATH at it, and rerun the analysis."
            )
            st.stop()

        progress = st.progress(0)
        progress_text = st.empty()
        alert_slot = st.empty()

        def on_progress(processed_frames: int, total_frames: int, confidence: float) -> None:
            fraction = min(processed_frames / max(total_frames, 1), 1.0)
            progress.progress(fraction)
            progress_text.info(f"Processed {processed_frames}/{total_frames} frames | Confidence: {confidence:.2%}")

        def on_alert(payload) -> None:
            alert_slot.markdown(
                f"<div class='alert-banner'>🚨 Violence Detected<br/>Confidence: {payload.confidence:.2%}<br/>Timestamp: {payload.timestamp}</div>",
                unsafe_allow_html=True,
            )

        with st.spinner("Processing video with the existing inference pipeline..."):
            result = analyze_video(
                video_path,
                output_video_path=output_video_path,
                detection_threshold=detection_threshold,
                enable_gradcam=enable_gradcam,
                enable_report=enable_report,
                enable_email=enable_email,
                email_recipient=admin_email,
                progress_callback=on_progress,
                alert_callback=on_alert,
            )

        st.session_state.analysis_result = result
        st.session_state.video_path = video_path
        st.session_state.output_video_path = output_video_path

        if Path(video_path).exists():
            try:
                browser_original_path = _transcode_for_browser(video_path)
            except Exception:
                browser_original_path = video_path

        if Path(output_video_path).exists():
            try:
                browser_video_path = _transcode_for_browser(output_video_path)
            except Exception:
                browser_video_path = output_video_path

        st.session_state.browser_original_video_path = browser_original_path
        st.session_state.browser_output_video_path = browser_video_path

    if process_button and input_source in {"Live Webcam", "RTSP/IP Camera"}:
        live_source = 0 if input_source == "Live Webcam" else rtsp_url.strip()
        if input_source == "RTSP/IP Camera" and not live_source:
            st.error("Please enter an RTSP or HTTP camera URL.")
            st.stop()

        live_placeholder = st.empty()
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()
        camera_frame_placeholder = st.empty()
        alert_placeholder = st.empty()
        snapshot_placeholder = st.empty()

        def on_live_status(payload: dict) -> None:
            status_placeholder.markdown(
                f"**Current Status:** {payload['status']}  \n"
                f"**Confidence:** {payload['confidence']:.2%}  \n"
                f"**Timestamp:** {payload['timestamp']}  \n"
                f"**Total Alerts:** {payload['total_alerts']}  \n"
                f"**Elapsed Monitoring Time:** {payload['elapsed_seconds']:.1f}s  \n"
                f"**FPS:** {payload['fps']:.2f}  \n"
                f"**Latency:** {payload['latency_ms']:.1f} ms"
            )
            metrics_placeholder.metric("Live Confidence", f"{payload['confidence']:.2%}")
            metrics_placeholder.metric("Total Alerts", payload["total_alerts"])

        def on_live_alert(payload) -> None:
            alert_placeholder.markdown(
                f"<div class='alert-banner'>🚨 Violence Detected<br/>Confidence: {payload.confidence:.2%}<br/>Timestamp: {payload.timestamp}<br/>Camera Source: {live_source}</div>",
                unsafe_allow_html=True,
            )

        def on_snapshot(snapshot_path: str) -> None:
            if Path(snapshot_path).exists():
                snapshot_placeholder.image(snapshot_path, caption=Path(snapshot_path).name, use_container_width=True)

        def on_frame(frame_bgr: np.ndarray) -> None:
            camera_frame_placeholder.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), channels="RGB", caption="Live Camera Feed", use_container_width=True)

        with st.spinner("Starting live monitoring..."):
            state = monitor_live_source(
                live_source,
                detection_threshold=detection_threshold,
                enable_gradcam=enable_gradcam,
                enable_email=enable_email,
                email_recipient=admin_email,
                status_callback=on_live_status,
                alert_callback=on_live_alert,
                snapshot_callback=on_snapshot,
                frame_callback=on_frame,
                periodic_summary_interval_seconds=2 * 60 * 60,
            )

        st.success(f"Live monitoring completed. Total alerts: {state.total_alerts}")
        st.session_state.analysis_result = None
        return

    result = st.session_state.get("analysis_result")
    if not result:
        st.info("Upload a video and run the analysis to populate the dashboard.")
        return

    metric_cols = st.columns(6)
    metric_cols[0].metric("Video Duration", _format_seconds(result.duration_seconds))
    metric_cols[1].metric("Processing FPS", f"{result.processing_fps:.2f}")
    metric_cols[2].metric("Total Frames", f"{result.total_frames}")
    metric_cols[3].metric("Total Incidents", f"{result.total_incidents}")
    metric_cols[4].metric("Highest Confidence", f"{result.highest_confidence:.2%}")
    metric_cols[5].metric("Processing Time", f"{result.processing_time:.1f}s")

    video_cols = st.columns(2)
    with video_cols[0]:
        st.subheader("Original Video")
        original_browser_path = st.session_state.get("browser_original_video_path")
        if original_browser_path and Path(original_browser_path).exists():
            st.video(original_browser_path)
        else:
            st.info("Original video is not available yet.")
    with video_cols[1]:
        st.subheader("Annotated Video")
        browser_video_path = st.session_state.get("browser_output_video_path")
        if browser_video_path and Path(browser_video_path).exists():
            st.video(browser_video_path)
        else:
            st.info("Annotated output video is not available yet.")

    timeline_rows = [
        {
            "Timestamp": event.timestamp,
            "Event Type": event.event_type,
            "Confidence": f"{event.confidence:.2%}",
            "Start Time": f"{event.start_time:.2f}",
            "End Time": f"{event.end_time:.2f}",
            "Duration": f"{event.duration:.2f}",
        }
        for event in result.timeline
    ]

    chart_cols = st.columns(2)
    times = [prediction.start_time for prediction in result.window_predictions]
    confidences = [prediction.confidence for prediction in result.window_predictions]
    with chart_cols[0]:
        st.subheader("Violence Probability Chart")
        if times and confidences:
            st.plotly_chart(confidence_line_chart(times, confidences), use_container_width=True)
            st.plotly_chart(confidence_histogram(confidences), use_container_width=True)
        else:
            st.info("No window predictions available.")
    with chart_cols[1]:
        st.subheader("Event Timeline")
        if timeline_rows:
            st.dataframe(pd.DataFrame(timeline_rows), use_container_width=True, hide_index=True)
            st.plotly_chart(incident_distribution_chart(result.timeline), use_container_width=True)
        else:
            st.info("No events detected.")

    st.subheader("Frame Processing Speed")
    if times:
        fps_values = [result.fps for _ in times]
        st.plotly_chart(create_speed_figure(times, fps_values), use_container_width=True)

    st.subheader("Incident Summary")
    summary_text = (
        f"{result.total_incidents} incident(s) detected in {result.video_name}. "
        f"Highest confidence {result.highest_confidence:.2%}, average confidence {result.average_confidence:.2%}."
    )
    st.write(summary_text)

    if result.timeline_csv_path and Path(result.timeline_csv_path).exists():
        with open(result.timeline_csv_path, "rb") as file_handle:
            st.download_button("Download timeline.csv", file_handle, file_name="timeline.csv")

    if result.report_path and Path(result.report_path).exists():
        with open(result.report_path, "rb") as file_handle:
            st.download_button("Download incident report PDF", file_handle, file_name=Path(result.report_path).name)

    if result.snapshot_paths:
        st.subheader("Incident Snapshots")
        snapshot_cols = st.columns(min(3, len(result.snapshot_paths)))
        for idx, snapshot_path in enumerate(result.snapshot_paths[: len(snapshot_cols)]):
            snapshot_cols[idx].image(snapshot_path, caption=Path(snapshot_path).name, use_container_width=True)

    if result.heatmap_paths:
        st.subheader("Explainable AI Heatmaps")
        heatmap_rows = min(len(result.heatmap_paths), len(result.heatmap_raw_paths))
        for idx in range(heatmap_rows):
            row_cols = st.columns(3)
            row_cols[0].image(result.snapshot_paths[idx], caption=f"Original frame {idx + 1}", use_container_width=True)
            row_cols[1].image(result.heatmap_raw_paths[idx], caption=f"Heatmap {idx + 1} ({result.heatmap_modes[idx] if idx < len(result.heatmap_modes) else 'unknown'})", use_container_width=True)
            row_cols[2].image(result.heatmap_paths[idx], caption=f"Overlay {idx + 1}", use_container_width=True)


if __name__ == "__main__":
    main()
