"""Draw a 12-stage *predicted* CAPE continual-learning figure.

This is an illustrative planning figure, not an experimental result.  It reads
the heterogeneous-long-stream aggregate trajectory/final means reported in
Table II of ``cape_ieee_journal.tex`` and constructs a deterministic 12-point
profile satisfying both constraints:

1. the mean of the 12 predicted stage values equals the reported trajectory
   average; and
2. the last predicted stage value equals the reported final average.

No stage-wise measurement or uncertainty is inferred.  Replace this figure
with curves computed from real ``history.json`` files before submission.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = (
    PROJECT_ROOT
    / "paper"
    / "cape"
    / "CAPE__Continual_Audio_Visual_Pattern_Experts_for_Open_World_Deepfake_Detection"
)
TEX_PATH = PAPER_DIR / "cape_ieee_journal.tex"
OUTPUT_STEM = PAPER_DIR / "capefig3_predicted_heterogeneous"

ALL_METHODS = [
    "Seq-FT",
    "EWC",
    "LwF",
    "ER",
    "DER",
    "DER++",
    "iCaRL",
    "MEMO",
    "L2P",
    "DualPrompt",
    "CODA-Prompt",
    "DyTox",
    "PROOF",
    "MoE-Adapters",
    "CAPE",
]

# Five readable curves: naive baseline, replay, prompt, expert, and CAPE.
DISPLAY_METHODS = ["Seq-FT", "DER++", "CODA-Prompt", "MoE-Adapters", "CAPE"]

TASK_LABELS = [
    "FaceSwap",
    "FSGAN",
    "Wav2Lip",
    "RTVC",
    "Joint A-V",
    "Visual-only",
    "Audio-only",
    "Audio-visual",
    "Kling2.5",
    "Veo3.1",
    "Wan2.5",
    "Seedance1.0",
]

RESULT_RE = re.compile(r"\\tbl(?:res|best)\{([0-9.]+)\}\{([0-9.]+)\}")


@dataclass(frozen=True)
class Result:
    mean: float
    sd: float


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def panel_segment(tex: str, start_marker: str, end_marker: str) -> str:
    start = tex.find(start_marker)
    if start < 0:
        raise ValueError(f"Cannot find table marker: {start_marker}")
    end = tex.find(end_marker, start + len(start_marker))
    if end < 0:
        raise ValueError(f"Cannot find table end marker: {end_marker}")
    return tex[start:end]


def parse_panel(segment: str) -> Dict[str, List[Result]]:
    raw_names = {method: method for method in ALL_METHODS if method != "CAPE"}
    raw_names[r"\method"] = "CAPE"
    lines = segment.splitlines()
    parsed: Dict[str, List[Result]] = {}

    for index, line in enumerate(lines):
        raw_name = line.strip()
        if raw_name not in raw_names:
            continue
        values: List[Result] = []
        cursor = index + 1
        while cursor < len(lines) and len(values) < 6:
            match = RESULT_RE.search(lines[cursor])
            if match:
                values.append(Result(float(match.group(1)), float(match.group(2))))
            cursor += 1
        if len(values) != 6:
            raise ValueError(f"Expected six cells for {raw_name}, found {len(values)}")
        parsed[raw_names[raw_name]] = values

    missing = [method for method in ALL_METHODS if method not in parsed]
    if missing:
        raise ValueError(f"Missing methods in Table II panel: {missing}")
    return parsed


def load_table_results(tex_path: Path) -> Dict[str, Dict[str, List[Result]]]:
    tex = tex_path.read_text(encoding="utf-8")
    auc = panel_segment(tex, "% Panel (a): AUC", "% Panel (b): AP")
    ap = panel_segment(tex, "% Panel (b): AP", r"\end{table*}")
    return {"AUC": parse_panel(auc), "AP": parse_panel(ap)}


def normalized_profile() -> np.ndarray:
    """Return a monotone domain-shift profile with mean 0 and last value 1.

    Larger increments at stages 6 and 9 encode the two dataset transitions.
    The normalization is what makes the trajectory/final constraints exact.
    """

    raw = np.asarray(
        [-1.15, -0.92, -0.78, -0.64, -0.50, -0.25,
         -0.12, 0.02, 0.28, 0.48, 0.72, 1.00],
        dtype=float,
    )
    centered = raw - raw.mean()
    profile = centered / centered[-1]
    assert np.isclose(profile.mean(), 0.0)
    assert np.isclose(profile[-1], 1.0)
    return profile


def predict_curve(trajectory_mean: float, final_mean: float) -> np.ndarray:
    """Construct a 12-point profile satisfying the two Table II means."""

    curve = trajectory_mean + (final_mean - trajectory_mean) * normalized_profile()
    if not np.isclose(curve.mean(), trajectory_mean, atol=1e-12):
        raise AssertionError("Predicted curve does not preserve trajectory average")
    if not np.isclose(curve[-1], final_mean, atol=1e-12):
        raise AssertionError("Predicted curve does not preserve final average")
    return curve


def heterogeneous_pair(
    results: Mapping[str, Mapping[str, Sequence[Result]]], metric: str, method: str
) -> tuple[Result, Result]:
    # Table II cell order: AV1M trajectory/final, HiFi trajectory/final,
    # heterogeneous trajectory/final.
    return results[metric][method][4], results[metric][method][5]


def write_prediction_csv(
    path: Path,
    results: Mapping[str, Mapping[str, Sequence[Result]]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "status",
                "method",
                "metric",
                "stage",
                "task",
                "predicted_value",
                "source_trajectory_mean",
                "source_trajectory_sd",
                "source_final_mean",
                "source_final_sd",
            ]
        )
        for metric in ("AUC", "AP"):
            for method in DISPLAY_METHODS:
                trajectory, final = heterogeneous_pair(results, metric, method)
                curve = predict_curve(trajectory.mean, final.mean)
                for stage, (task, value) in enumerate(zip(TASK_LABELS, curve), start=1):
                    writer.writerow(
                        [
                            "illustrative_prediction_not_measurement",
                            method,
                            metric,
                            stage,
                            task,
                            f"{value:.6f}",
                            f"{trajectory.mean:.4f}",
                            f"{trajectory.sd:.4f}",
                            f"{final.mean:.4f}",
                            f"{final.sd:.4f}",
                        ]
                    )


def draw_domain_background(ax: plt.Axes) -> None:
    ax.axvspan(0.5, 5.5, color="#EAF3FA", zorder=0)
    ax.axvspan(5.5, 8.5, color="#F7F0E3", zorder=0)
    ax.axvspan(8.5, 12.5, color="#EAF5EA", zorder=0)
    ax.axvline(5.5, color="#777777", linestyle="--", linewidth=0.8)
    ax.axvline(8.5, color="#777777", linestyle="--", linewidth=0.8)
    domain_text = {"transform": ax.get_xaxis_transform(), "ha": "center", "va": "top", "fontsize": 7.0}
    ax.text(3.0, 0.975, "FakeAVCeleb", **domain_text)
    ax.text(7.0, 0.975, "AV-Deepfake1M", **domain_text)
    ax.text(10.5, 0.975, "HiFi-AVDF", **domain_text)


def make_figure(
    results: Mapping[str, Mapping[str, Sequence[Result]]],
) -> plt.Figure:
    colors = {
        "Seq-FT": "#777777",
        "DER++": "#E69F00",
        "CODA-Prompt": "#009E73",
        "MoE-Adapters": "#CC79A7",
        "CAPE": "#0072B2",
    }
    markers = {
        "Seq-FT": "o",
        "DER++": "s",
        "CODA-Prompt": "^",
        "MoE-Adapters": "D",
        "CAPE": "o",
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.35), sharex=True, sharey=False)
    x = np.arange(1, 13)

    for ax, metric, panel in zip(axes, ("AUC", "AP"), ("(a)", "(b)")):
        draw_domain_background(ax)
        all_values = []
        for method in DISPLAY_METHODS:
            trajectory, final = heterogeneous_pair(results, metric, method)
            curve = predict_curve(trajectory.mean, final.mean)
            all_values.extend(curve.tolist())
            is_cape = method == "CAPE"
            ax.plot(
                x,
                curve,
                color=colors[method],
                marker=markers[method],
                linewidth=2.4 if is_cape else 1.25,
                markersize=5.0 if is_cape else 3.6,
                markeredgecolor="white" if is_cape else colors[method],
                markeredgewidth=0.65,
                zorder=4 if is_cape else 3,
                label=method,
            )

        lower = max(0.45, min(all_values) - 0.035)
        ax.set_ylim(lower, 1.0)
        ax.set_xlim(0.5, 12.5)
        ax.set_title(f"{panel} Predicted average {metric}", loc="left", fontweight="bold")
        ax.set_ylabel(f"Average {metric} over seen stages")
        ax.set_xlabel("Training stage completed")
        ax.set_xticks(x)
        ax.set_xticklabels(TASK_LABELS, rotation=50, ha="right")
        ax.grid(axis="y", color="#D7D7D7", linewidth=0.55, alpha=0.85, zorder=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.5,
            0.05,
            "PREDICTION — NOT MEASURED",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#7B1E1E",
            alpha=0.16,
            rotation=16,
            zorder=2,
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=colors[method],
            marker=markers[method],
            linewidth=2.4 if method == "CAPE" else 1.25,
            markersize=4.6,
            label=method,
        )
        for method in DISPLAY_METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(handles),
        frameon=False,
        columnspacing=1.2,
        handlelength=2.1,
    )
    fig.text(
        0.5,
        0.005,
        "Illustrative profiles constrained by Table II aggregate means; "
        "not stage-wise experimental observations.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#555555",
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.31, top=0.82, wspace=0.18)
    return fig


def main() -> None:
    configure_style()
    results = load_table_results(TEX_PATH)
    write_prediction_csv(OUTPUT_STEM.with_suffix(".csv"), results)
    fig = make_figure(results)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUTPUT_STEM.with_suffix(f".{extension}"))
    plt.close(fig)
    print(f"Source: {TEX_PATH}")
    print(f"Saved predicted planning figure: {OUTPUT_STEM.with_suffix('.pdf')}")
    print("Status: illustrative prediction, not an experimental result")


if __name__ == "__main__":
    main()
