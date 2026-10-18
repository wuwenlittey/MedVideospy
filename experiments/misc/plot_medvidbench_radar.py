#!/usr/bin/env python3
"""Plot VideoSpy before/after evolution from VideoSpyMedicalTable in table.tex.

Requires matplotlib (python -m pip install matplotlib).
Run from any directory:
    python experiments/misc/plot_medvidbench_radar.py
    python experiments/misc/plot_medvidbench_radar.py --model both
    python experiments/misc/plot_medvidbench_radar.py --model both --compact
    python experiments/misc/plot_medvidbench_radar.py --model 27B
    python experiments/misc/plot_medvidbench_radar.py --split leaderboard

Each spoke has its own zero-based range, shown below its metric label.
Axis maxima are rounded up from both models in the selected split, so the
4B and 27B figures share scales. Polygon area is not an aggregate score.
PDF and 300-dpi PNG files are saved under output/figures by default.
"""

import argparse
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE = ROOT / "docs/research/evolving_video_agent/iclr2027/table.tex"
LABELS = (
    "CVS acc", "NAP acc", "SA acc", "STG mIoU",
    "TAG mIoU@0.3", "TAG mIoU@0.5", "DVC F1",
    "DVC LLM", "VS LLM", "RC LLM",
)
SCALES = (1, 1, 1, 1, 1, 1, 1, 5, 5, 5)
STATES = ((r"\mathbf{0}", r"VideoSpy$^{0}$", "#4B7CA9"),
          (r"\mathcal{E}", r"VideoSpy$^{\mathcal{E}}$", "#D57452"))


def read_scores(table: Path, split: str, model: str) -> dict[str, list[float]]:
    """Select the named table, split, backbone, and two method rows."""
    source = table.read_text(encoding="utf-8")
    source = source.split(r"\newcommand{\VideoSpyMedicalTable}", 1)[1]
    source = source.split(r"\newcommand", 1)[0]
    current_split = None
    scores = {}
    for line in source.splitlines():
        line = line.strip()
        if line.startswith("%"):
            continue
        if "(a) Local test split" in line:
            current_split = "local"
        elif "(b) Leaderboard split" in line:
            current_split = "leaderboard"
        if current_split != split or not line.startswith("VideoSpy"):
            continue
        cells = line.split("&")
        if cells[1].strip() != model:
            continue
        for marker, _, _ in STATES:
            if marker not in cells[0]:
                continue
            # Numeric cells may be wrapped in LaTeX bold commands.
            values = [float(re.search(r"\d+(?:\.\d+)?", cell).group())
                      for cell in cells[2:]]
            if len(values) != len(LABELS):
                raise ValueError(f"Expected {len(LABELS)} metrics: {line}")
            if any(not 0 <= value <= scale
                   for value, scale in zip(values, SCALES)):
                raise ValueError(f"Metric outside its plotting range: {values}")
            scores[marker] = values
    if len(scores) != 2:
        raise ValueError(f"Expected both VideoSpy rows for {split=}, {model=}")
    return scores


def axis_maxima(rows: list[list[float]]) -> list[float]:
    """Leave 6% headroom and round up to half a power of ten."""
    maxima = []
    for values in zip(*rows):
        peak = max(values)
        if peak == 0:
            maxima.append(0.1)
            continue
        step = 0.5 * 10 ** math.floor(math.log10(peak))
        maxima.append(math.ceil(peak * 1.06 / step) * step)
    return maxima


def plot_radar(scores: dict[str, list[float]], maxima: list[float],
               output: Path, compact: bool = False) -> None:
    angles = [2 * math.pi * i / len(LABELS) for i in range(len(LABELS))]
    angles.append(angles[0])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.6, 3.1) if compact else (7, 6.6),
                           subplot_kw={"projection": "polar"})
    fig.subplots_adjust(left=0.22, right=0.78,
                        bottom=0.17 if compact else 0.22, top=0.83 if compact else 0.86)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.set_facecolor("white")
    ax.xaxis.grid(color="#E2E7EC", linewidth=0.55)
    ax.yaxis.grid(color="#DCE3E9", linewidth=0.55, linestyle=(0, (2, 3)))
    ax.spines["polar"].set_color("#C9D2DB")
    ax.spines["polar"].set_linewidth(0.7)
    for index, (marker, label, color) in enumerate(STATES):
        values = [value / scale for value, scale in zip(scores[marker], maxima)]
        values.append(values[0])
        ax.plot(angles, values, color=color, linewidth=1.4 if compact else 2.0,
                linestyle=(0, (4, 2)) if index == 0 else "-", marker="o",
                markersize=3 if compact else 4.5,
                markerfacecolor="white" if index == 0 else color,
                markeredgecolor=color if index == 0 else "white",
                markeredgewidth=0.8, label=label, zorder=3 + index)
        ax.fill(angles, values, color=color, alpha=0.07 if index == 0 else 0.11,
                zorder=2)
    # Keep labels and ranges outside the data area, with consistent alignment.
    for angle, label, maximum in zip(angles[:-1], LABELS, maxima):
        horizontal = math.sin(angle)
        alignment = "left" if horizontal > 0.15 else "right" if horizontal < -0.15 else "center"
        ax.annotate(label, (angle, 1.15), xytext=(0, 4 if compact else 6),
                    textcoords="offset points", ha=alignment, va="center",
                    fontsize=9 if compact else 13, color="#263642",
                    annotation_clip=False)
        ax.annotate(f"[0, {maximum:g}]", (angle, 1.15),
                    xytext=(0, -6 if compact else -9), textcoords="offset points",
                    ha=alignment, va="center", fontsize=7 if compact else 10,
                    color="#73818D", annotation_clip=False)
    if not compact:
        fig.legend(*ax.get_legend_handles_labels(), loc="lower center",
                   bbox_to_anchor=(0.5, 0.035), ncol=2, frameon=False, fontsize=12,
                   handlelength=2.4, columnspacing=2.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = output.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--split", choices=("local", "leaderboard"), default="local")
    parser.add_argument("--model", choices=("4B", "27B", "both"), default="4B")
    parser.add_argument("--output", type=Path, help="Output path stem (without extension)")
    parser.add_argument("--compact", action="store_true",
                        help="Compact paper layout with no legend, for a shared paper legend")
    args = parser.parse_args()
    if args.split == "leaderboard" and args.model in ("27B", "both"):
        parser.error("The leaderboard table contains only 4B results.")
    available = ("4B", "27B") if args.split == "local" else ("4B",)
    all_scores = {model: read_scores(args.table, args.split, model) for model in available}
    maxima = axis_maxima([row for scores in all_scores.values() for row in scores.values()])
    models = available if args.model == "both" else (args.model,)
    for model in models:
        output = args.output or ROOT / "output/figures" / f"medvidbench_{args.split}_{model}_radar"
        if args.output and args.model == "both":
            output = output.with_name(f"{output.name}_{model}")
        plot_radar(all_scores[model], maxima, output, compact=args.compact)


if __name__ == "__main__":
    main()
