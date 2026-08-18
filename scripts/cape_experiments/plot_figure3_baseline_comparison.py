"""Generate CAPE Figure 3 from the aggregate results in the IEEE manuscript.

The manuscript currently contains mean and sample standard deviation for five
random seeds, but the complete per-stage histories for all baselines/protocols
are not present in the repository.  This script therefore visualizes only the
reported aggregate trajectory-average and final-average results; it never
reconstructs or interpolates stage-wise values.

Outputs
-------
* capefig3_baseline_comparison.pdf   (vector, manuscript-ready)
* capefig3_baseline_comparison.svg   (editable vector)
* capefig3_baseline_comparison.png   (600 dpi preview)
* capefig3_baseline_comparison.csv   (plotted data)
* capefig3_all_methods.csv           (all rows parsed from the table)

Example
-------
python scripts/cape_experiments/plot_figure3_baseline_comparison.py
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from string import ascii_lowercase
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = (
    PROJECT_ROOT
    / "paper"
    / "cape"
    / "CAPE__Continual_Audio_Visual_Pattern_Experts_for_Open_World_Deepfake_Detection"
)
DEFAULT_TEX = PAPER_DIR / "cape_ieee_journal.tex"

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

# One representative from each major continual-learning family, plus CAPE.
DEFAULT_METHODS = [
    "Seq-FT",
    "EWC",
    "DER++",
    "MEMO",
    "CODA-Prompt",
    "MoE-Adapters",
    "CAPE",
]

PROTOCOLS = [
    ("AV1M Pattern-Inc.", 0),
    ("HiFi Generator-Inc.", 2),
    ("Heterogeneous Long-Stream", 4),
]

RESULT_RE = re.compile(r"\\tbl(?:res|best)\{([0-9.]+)\}\{([0-9.]+)\}")


@dataclass(frozen=True)
class Result:
    mean: float
    sd: float


def configure_style() -> None:
    """Configure an IEEE-compatible, colorblind-safe plotting style."""

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.6,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.3,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def panel_segment(tex: str, start_marker: str, end_marker: str) -> str:
    start = tex.find(start_marker)
    if start < 0:
        raise ValueError(f"Cannot find panel marker: {start_marker}")
    end = tex.find(end_marker, start + len(start_marker))
    if end < 0:
        raise ValueError(f"Cannot find panel end marker: {end_marker}")
    return tex[start:end]


def parse_panel(segment: str) -> Dict[str, List[Result]]:
    """Parse six result cells per method from one manuscript table panel."""

    raw_method_names = {name: name for name in ALL_METHODS if name != "CAPE"}
    raw_method_names[r"\method"] = "CAPE"
    lines = segment.splitlines()
    parsed: Dict[str, List[Result]] = {}

    for index, line in enumerate(lines):
        raw_name = line.strip()
        if raw_name not in raw_method_names:
            continue
        values: List[Result] = []
        cursor = index + 1
        while cursor < len(lines) and len(values) < 6:
            match = RESULT_RE.search(lines[cursor])
            if match:
                values.append(Result(float(match.group(1)), float(match.group(2))))
            cursor += 1
        if len(values) != 6:
            raise ValueError(
                f"Expected six values for {raw_name}, found {len(values)}"
            )
        parsed[raw_method_names[raw_name]] = values

    missing = [method for method in ALL_METHODS if method not in parsed]
    if missing:
        raise ValueError(f"Missing methods in parsed panel: {missing}")
    return parsed


def load_results(tex_path: Path) -> Dict[str, Dict[str, List[Result]]]:
    tex = tex_path.read_text(encoding="utf-8")
    auc_segment = panel_segment(tex, "% Panel (a): AUC", "% Panel (b): AP")
    ap_segment = panel_segment(tex, "% Panel (b): AP", r"\end{table*}")
    return {"AUC": parse_panel(auc_segment), "AP": parse_panel(ap_segment)}


def write_csv(
    path: Path,
    results: Mapping[str, Mapping[str, Sequence[Result]]],
    methods: Iterable[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "metric",
                "protocol",
                "summary",
                "mean",
                "sample_sd",
                "n_seeds",
                "source",
            ]
        )
        for method in methods:
            for metric in ("AUC", "AP"):
                for protocol, offset in PROTOCOLS:
                    trajectory = results[metric][method][offset]
                    final = results[metric][method][offset + 1]
                    for summary, value in (
                        ("trajectory_average", trajectory),
                        ("final_average", final),
                    ):
                        writer.writerow(
                            [
                                method,
                                metric,
                                protocol,
                                summary,
                                f"{value.mean:.4f}",
                                f"{value.sd:.4f}",
                                5,
                                "cape_ieee_journal.tex:tab:main-continual",
                            ]
                        )


def axis_limits(values: Sequence[Tuple[Result, Result]]) -> Tuple[float, float]:
    low = min(min(a.mean - a.sd, b.mean - b.sd) for a, b in values)
    high = max(max(a.mean + a.sd, b.mean + b.sd) for a, b in values)
    span = max(high - low, 0.02)
    lower = max(0.5, low - 0.10 * span)
    upper = min(1.0, high + 0.10 * span)
    return lower, upper


def draw_panel(
    ax: plt.Axes,
    panel_label: str,
    protocol: str,
    metric: str,
    offset: int,
    results: Mapping[str, Mapping[str, Sequence[Result]]],
    methods: Sequence[str],
) -> None:
    y = np.arange(len(methods), dtype=float)
    pairs = [(results[metric][m][offset], results[metric][m][offset + 1]) for m in methods]

    # Highlight the proposed method without changing the common metric scale.
    cape_index = methods.index("CAPE")
    ax.axhspan(cape_index - 0.42, cape_index + 0.42, color="#DCEEFF", zorder=0)

    trajectory_color = "#0072B2"  # Okabe-Ito blue
    final_color = "#D55E00"       # Okabe-Ito vermillion
    connector_color = "#A7A7A7"

    for row, method in enumerate(methods):
        trajectory, final = pairs[row]
        ax.plot(
            [trajectory.mean, final.mean],
            [row, row],
            color=connector_color,
            linewidth=1.0 if method != "CAPE" else 1.6,
            zorder=1,
        )
        common = {
            "elinewidth": 0.8,
            "capsize": 2.0,
            "capthick": 0.8,
            "zorder": 3,
        }
        ax.errorbar(
            trajectory.mean,
            row,
            xerr=trajectory.sd,
            fmt="o",
            color=trajectory_color,
            markeredgecolor="black" if method == "CAPE" else trajectory_color,
            markeredgewidth=0.65,
            markersize=5.1 if method == "CAPE" else 4.0,
            **common,
        )
        ax.errorbar(
            final.mean,
            row,
            xerr=final.sd,
            fmt="s",
            color=final_color,
            markeredgecolor="black" if method == "CAPE" else final_color,
            markeredgewidth=0.65,
            markersize=5.0 if method == "CAPE" else 3.9,
            **common,
        )

    ax.set_title(f"({panel_label}) {protocol} — {metric}", loc="left", fontweight="bold")
    ax.set_xlabel(metric)
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.invert_yaxis()
    ax.set_xlim(*axis_limits(pairs))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.55, alpha=0.9, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    for label in ax.get_yticklabels():
        if label.get_text() == "CAPE":
            label.set_fontweight("bold")


def make_figure(
    results: Mapping[str, Mapping[str, Sequence[Result]]],
    methods: Sequence[str],
) -> plt.Figure:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.16, 5.35),
        constrained_layout=False,
        sharey="row",
    )

    panel_index = 0
    for row, metric in enumerate(("AUC", "AP")):
        for column, (protocol, offset) in enumerate(PROTOCOLS):
            draw_panel(
                axes[row, column],
                ascii_lowercase[panel_index],
                protocol,
                metric,
                offset,
                results,
                methods,
            )
            panel_index += 1
            if column > 0:
                axes[row, column].tick_params(labelleft=False)

    legend = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="#0072B2",
            markeredgecolor="#0072B2", markersize=4.5,
            label=r"Trajectory average $\bar{\mathcal{A}}$",
        ),
        Line2D(
            [0], [0], marker="s", color="none", markerfacecolor="#D55E00",
            markeredgecolor="#D55E00", markersize=4.4,
            label=r"Final average $\mathcal{A}_B$",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.997),
        ncol=2,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.125, right=0.995, bottom=0.075, top=0.925, wspace=0.20, hspace=0.38)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the aggregate baseline-comparison Figure 3 from the IEEE table."
    )
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--output-dir", type=Path, default=PAPER_DIR)
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated methods to display; all values are still exported to CSV.",
    )
    parser.add_argument("--stem", default="capefig3_baseline_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tex_path = args.tex.resolve()
    output_dir = args.output_dir.resolve()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = [method for method in methods if method not in ALL_METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    if "CAPE" not in methods:
        raise ValueError("The displayed method list must include CAPE")

    configure_style()
    results = load_results(tex_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / f"{args.stem}.csv", results, methods)
    write_csv(output_dir / "capefig3_all_methods.csv", results, ALL_METHODS)

    fig = make_figure(results, methods)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output_dir / f"{args.stem}.{extension}")
    plt.close(fig)

    print(f"Source table: {tex_path}")
    print(f"Displayed methods ({len(methods)}): {', '.join(methods)}")
    print(f"Saved Figure 3 to: {output_dir / (args.stem + '.pdf')}")


if __name__ == "__main__":
    main()
