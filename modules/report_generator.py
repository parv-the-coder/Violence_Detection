"""PDF incident report generation using ReportLab."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from modules.timeline import TimelineEvent


def _safe_image(path: str, width: float = 5.8 * inch) -> Image:
    image = Image(path)
    image.drawWidth = width
    image.drawHeight = image.imageHeight * (width / image.imageWidth)
    return image


def generate_pdf_report(
    output_path: str,
    *,
    project_title: str,
    video_name: str,
    processing_date: str,
    video_duration: float,
    model_used: str,
    timeline: List[TimelineEvent],
    incidents: List[dict],
    max_confidence: float,
    average_confidence: float,
    summary: str,
    recommendations: str,
) -> str:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SectionTitle", parent=styles["Heading2"], textColor=colors.HexColor("#b91c1c")))
    styles.add(ParagraphStyle(name="BodyLeft", parent=styles["BodyText"], alignment=TA_LEFT, leading=14))

    story = [
        Paragraph(project_title, styles["Title"]),
        Spacer(1, 0.15 * inch),
    ]

    info_rows = [
        ["Video Name", video_name],
        ["Processing Date", processing_date],
        ["Video Duration", f"{video_duration:.2f} s"],
        ["Model Used", model_used],
        ["Number of Events", str(len(timeline))],
        ["Maximum Confidence", f"{max_confidence:.2%}"],
        ["Average Confidence", f"{average_confidence:.2%}"],
    ]
    table = Table(info_rows, colWidths=[1.8 * inch, 4.4 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#111827")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
            ]
        )
    )
    story.extend([table, Spacer(1, 0.2 * inch), Paragraph("Summary", styles["SectionTitle"]), Paragraph(summary, styles["BodyLeft"]), Spacer(1, 0.15 * inch), Paragraph("Recommendations", styles["SectionTitle"]), Paragraph(recommendations, styles["BodyLeft"]), Spacer(1, 0.2 * inch)])

    if timeline:
        story.append(Paragraph("Timeline", styles["SectionTitle"]))
        timeline_rows = [["Timestamp", "Event Type", "Confidence", "Start", "End", "Duration"]]
        for event in timeline:
            timeline_rows.append([
                event.timestamp,
                event.event_type,
                f"{event.confidence:.2%}",
                f"{event.start_time:.2f}",
                f"{event.end_time:.2f}",
                f"{event.duration:.2f}",
            ])
        timeline_table = Table(timeline_rows, repeatRows=1)
        timeline_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1d5db")),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
                ]
            )
        )
        story.extend([timeline_table, Spacer(1, 0.2 * inch)])

    if incidents:
        story.append(Paragraph("Detected Incidents", styles["SectionTitle"]))
        for incident in incidents:
            story.append(Paragraph(f"{incident.get('timestamp', '')} - {incident.get('summary', '')}", styles["BodyLeft"]))
            snapshot = incident.get("snapshot_path")
            if snapshot:
                story.append(_safe_image(snapshot))
                story.append(Spacer(1, 0.1 * inch))
            heatmap = incident.get("heatmap_path")
            if heatmap:
                story.append(_safe_image(heatmap))
                story.append(Spacer(1, 0.15 * inch))

    doc = SimpleDocTemplate(str(output_file), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=42, bottomMargin=36)
    doc.build(story)
    return str(output_file)
