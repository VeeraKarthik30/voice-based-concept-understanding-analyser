from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def build_pdf_report(
    output_path: str,
    transcription_text: str,
    scores: dict[str, float],
    strengths: list[str],
    weaknesses: list[str],
    suggestions: list[str],
    audio_metrics: dict[str, Any],
    semantic_metrics: dict[str, Any],
    chart_paths: list[str] | None = None,
    user_details: dict[str, Any] | None = None,
) -> str:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    document = SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=1.5 * cm, leftMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Voice-Based Concept Understanding Analyser Report", styles["Title"]))
    story.append(Spacer(1, 0.35 * cm))

    if user_details:
        story.append(Paragraph("Candidate Profile", styles["Heading2"]))
        profile_rows = [
            ["Name", escape(str(user_details.get("full_name", "n/a")))],
            ["Email", escape(str(user_details.get("email", "n/a")))],
            ["Concept", escape(str(user_details.get("concept_name", "n/a")))],
            ["Generated At", escape(str(user_details.get("generated_at", "n/a")))],
            ["Source Audio", escape(str(user_details.get("audio_filename", "n/a")))],
        ]
        profile_table = Table([["Field", "Value"]] + profile_rows, colWidths=[4.5 * cm, 11.5 * cm])
        profile_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#13203a")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#eef4ff")),
                ]
            )
        )
        story.append(profile_table)
        story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Assessment Summary", styles["Heading2"]))

    score_rows = [["Metric", "Score"]] + [[key.title(), f"{value:.2f}"] for key, value in scores.items()]
    score_table = Table(score_rows, colWidths=[7 * cm, 3.5 * cm])
    score_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2336")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5f7fb")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story.append(score_table)
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Audio Metrics", styles["Heading2"]))
    audio_rows = [
        ["Duration (s)", str(audio_metrics.get("duration_seconds", "n/a"))],
        ["Speech Rate (WPM)", str(audio_metrics.get("speech_rate_wpm", "n/a"))],
        ["Pause Count", str(audio_metrics.get("pause_count", "n/a"))],
        ["Filler Count", str(audio_metrics.get("filler_count", "n/a"))],
        ["Mean RMS Energy", str(audio_metrics.get("rms_energy_mean", "n/a"))],
        ["Zero Crossing Rate", str(audio_metrics.get("zero_crossing_rate", "n/a"))],
    ]
    audio_table = Table([["Metric", "Value"]] + audio_rows, colWidths=[7 * cm, 3.5 * cm])
    audio_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f3ff")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    story.append(audio_table)
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Semantic Metrics", styles["Heading2"]))
    semantic_rows = [
        ["Similarity Score", str(semantic_metrics.get("similarity_score", "n/a"))],
        ["Concept Coverage", str(semantic_metrics.get("concept_coverage", "n/a"))],
        ["Matched Concepts", escape(", ".join(semantic_metrics.get("matched_concepts", [])) or "None")],
        ["Missing Concepts", escape(", ".join(semantic_metrics.get("missing_concepts", [])) or "None")],
    ]
    semantic_table = Table([["Metric", "Value"]] + semantic_rows, colWidths=[7 * cm, 9 * cm])
    semantic_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0ebff")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    story.append(semantic_table)
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Strengths", styles["Heading2"]))
    for item in strengths:
        story.append(Paragraph(f"• {escape(str(item))}", styles["BodyText"]))
    if not strengths:
        story.append(Paragraph("• n/a", styles["BodyText"]))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("Weaknesses", styles["Heading2"]))
    for item in weaknesses:
        story.append(Paragraph(f"• {escape(str(item))}", styles["BodyText"]))
    if not weaknesses:
        story.append(Paragraph("• n/a", styles["BodyText"]))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("Improvement Suggestions", styles["Heading2"]))
    for item in suggestions:
        story.append(Paragraph(f"• {escape(str(item))}", styles["BodyText"]))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("Transcription", styles["Heading2"]))
    story.append(Paragraph(escape(transcription_text).replace("\n", "<br/>"), styles["BodyText"]))
    story.append(Spacer(1, 0.25 * cm))

    for chart_path in chart_paths or []:
        if Path(chart_path).exists():
            story.append(Image(chart_path, width=16 * cm, height=8.8 * cm))
            story.append(Spacer(1, 0.25 * cm))

    document.build(story)
    return str(destination)
