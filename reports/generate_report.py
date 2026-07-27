"""Render final_report.md as a simple submission-ready PDF."""

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPORT_DIR = Path(__file__).resolve().parent
source = (REPORT_DIR / "final_report.md").read_text(encoding="utf-8")

lines = []
for raw_line in source.splitlines():
    if raw_line.startswith("# "):
        lines.extend(["", raw_line[2:].upper(), ""])
    elif raw_line.startswith("## "):
        lines.extend(["", raw_line[3:], ""])
    elif raw_line:
        lines.extend(textwrap.wrap(raw_line, width=92))
    else:
        lines.append("")

with PdfPages(REPORT_DIR / "final_report.pdf") as pdf:
    for start in range(0, len(lines), 48):
        page = lines[start : start + 48]
        figure = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        figure.text(
            0.08,
            0.95,
            "\n".join(page),
            va="top",
            ha="left",
            family="DejaVu Sans",
            fontsize=9.5,
            linespacing=1.35,
        )
        figure.text(
            0.5,
            0.025,
            f"SmartHire · Page {start // 48 + 1}",
            ha="center",
            fontsize=8,
            color="#666666",
        )
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

    figure_files = [
        ("Resume classifier confusion matrix", "confusion_matrix.png"),
        ("Clustering model selection", "clustering_metrics.png"),
        ("Two-dimensional cluster projection", "cluster_projection.png"),
        ("Supervised fit predictor ROC curve", "fit_predictor_roc.png"),
    ]
    for title, filename in figure_files:
        image_path = REPORT_DIR / "figures" / filename
        if not image_path.exists():
            continue
        image = plt.imread(image_path)
        figure, axis = plt.subplots(figsize=(8.27, 11.69), facecolor="white")
        axis.imshow(image)
        axis.set_title(title, fontsize=16, pad=18)
        axis.axis("off")
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
