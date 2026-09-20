"""Chart builders for the surveillance dashboard."""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from modules.timeline import TimelineEvent


def confidence_line_chart(times: List[float], confidences: List[float]) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=times, y=confidences, mode="lines+markers", name="Violence Probability"))
    figure.update_layout(title="Violence Confidence vs Time", xaxis_title="Time (s)", yaxis_title="Confidence", height=340)
    return figure


def incident_distribution_chart(events: List[TimelineEvent]) -> go.Figure:
    violent = sum(1 for event in events if event.event_type == "Violence Detected")
    normal = sum(1 for event in events if event.event_type == "Normal")
    figure = go.Figure(data=[go.Bar(x=["Normal", "Violence"], y=[normal, violent], marker_color=["#22c55e", "#ef4444"])])
    figure.update_layout(title="Incident Distribution", height=320)
    return figure


def confidence_histogram(confidences: List[float]) -> go.Figure:
    figure = go.Figure(data=[go.Histogram(x=confidences, nbinsx=20, marker_color="#b91c1c")])
    figure.update_layout(title="Confidence Histogram", xaxis_title="Confidence", yaxis_title="Count", height=320)
    return figure


def create_speed_figure(times: List[float], fps_values: List[float]) -> go.Figure:
    figure = go.Figure(data=[go.Scatter(x=times, y=fps_values, mode="lines", line=dict(color="#0f766e"))])
    figure.update_layout(title="Frame Processing Speed", xaxis_title="Time (s)", yaxis_title="FPS", height=300)
    return figure
