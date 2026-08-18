"""Generate a coherent five-figure TMM experiment-planning package for CAPE.

IMPORTANT
---------
These figures are planning drafts, not verified experimental results.

* Figure 3 is a stage-wise prediction constrained by the trajectory-average
  and final-average values already reported in manuscript Table II.
* Figure 4 visualizes the aggregate values already written in manuscript
  Table III.
* Figure 5 contains predicted mechanism/retention profiles constrained by the
  CAPE heterogeneous-stream aggregate in Table II.
* Figure 6 contains explicitly predicted robustness trends.
* Figure 7 combines explicitly predicted sensitivity/efficiency trends with
  the aggregate ablation values written in manuscript Table IV.

Every output is visibly watermarked and accompanied by a CSV and README.
Replace predicted panels with statistics computed from real multi-seed run
artifacts before inserting them into a submission.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

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
OUTPUT_DIR = PAPER_DIR / "predicted_tmm_experiment_figures"

STATUS = "planning_draft_not_verified_experimental_result"
# ASCII-only text prevents font/encoding substitution in PDF/SVG exports.
WATERMARK = "PLANNING DRAFT - VERIFY WITH REAL RUNS"

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
DISPLAY_METHODS = ["Seq-FT", "DER++", "CODA-Prompt", "MoE-Adapters", "CAPE"]

METHOD_COLORS = {
    "Seq-FT": "#7A7A7A",
    "DER++": "#E69F00",
    "CODA-Prompt": "#009E73",
    "MoE-Adapters": "#CC79A7",
    "CAPE": "#0072B2",
}
METHOD_MARKERS = {
    "Seq-FT": "o",
    "DER++": "s",
    "CODA-Prompt": "^",
    "MoE-Adapters": "D",
    "CAPE": "o",
}

RESULT_RE = re.compile(r"\\tbl(?:res|best)\{([0-9.]+)\}\{([0-9.]+)\}")
OW_RE = re.compile(r"\\ow(?:res|best)\{([0-9.]+)\}\{([0-9.]+)\}")
MEAN_SD_RE = re.compile(
    r"(?:\\mathbf\{)?([0-9.]+)\}?\s*\\pm\s*([0-9.]+)"
)
PARAM_RE = re.compile(r"([0-9.]+)\$?\\times")


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
            "font.size": 7.8,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.6,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def segment(text: str, start_marker: str, end_marker: str, start_at: int = 0) -> str:
    start = text.find(start_marker, start_at)
    if start < 0:
        raise ValueError(f"Cannot find marker: {start_marker}")
    end = text.find(end_marker, start + len(start_marker))
    if end < 0:
        raise ValueError(f"Cannot find end marker: {end_marker}")
    return text[start:end]


def parse_table2_panel(panel: str) -> Dict[str, List[Result]]:
    raw_names = {method: method for method in ALL_METHODS if method != "CAPE"}
    raw_names[r"\method"] = "CAPE"
    lines = panel.splitlines()
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
            raise ValueError(f"Expected six Table II values for {raw_name}")
        parsed[raw_names[raw_name]] = values
    missing = [method for method in ALL_METHODS if method not in parsed]
    if missing:
        raise ValueError(f"Missing Table II methods: {missing}")
    return parsed


def parse_table2(tex: str) -> Dict[str, Dict[str, List[Result]]]:
    auc = segment(tex, "% Panel (a): AUC", "% Panel (b): AP")
    ap = segment(tex, "% Panel (b): AP", r"\end{table*}")
    return {"AUC": parse_table2_panel(auc), "AP": parse_table2_panel(ap)}


def parse_table3(tex: str) -> Dict[str, List[Result]]:
    label_pos = tex.find(r"\label{tab:unknown}")
    if label_pos < 0:
        raise ValueError("Cannot find Table III label")
    table = segment(tex, "MSP", r"\bottomrule", start_at=label_pos)
    raw_names = {
        "MSP": "MSP",
        "MaxLogit": "MaxLogit",
        "Energy": "Energy",
        "OpenMax": "OpenMax",
        "Mahalanobis": "Mahalanobis",
        r"$k$NN distance": "kNN distance",
        r"\textbf{\method}": "CAPE",
    }
    lines = table.splitlines()
    parsed: Dict[str, List[Result]] = {}
    for index, line in enumerate(lines):
        raw = line.strip()
        if raw not in raw_names:
            continue
        values: List[Result] = []
        cursor = index + 1
        while cursor < len(lines) and len(values) < 8:
            match = OW_RE.search(lines[cursor])
            if match:
                values.append(Result(float(match.group(1)), float(match.group(2))))
            cursor += 1
        if len(values) != 8:
            raise ValueError(f"Expected eight Table III values for {raw}")
        parsed[raw_names[raw]] = values
    return parsed


def parse_table4(tex: str) -> Dict[str, Dict[str, object]]:
    label_pos = tex.find(r"\label{tab:ablation}")
    if label_pos < 0:
        raise ValueError("Cannot find Table IV label")
    table = segment(tex, r"\textbf{Full \method}", r"\bottomrule", start_at=label_pos)
    raw_names = {
        r"\textbf{Full \method}": "Full CAPE",
        "w/o discrepancy encoder": "w/o discrepancy",
        "w/o pattern guidance": "w/o pattern",
        "w/o confidence-density routing": "w/o routing",
        "w/o continual retention": "w/o retention",
        "w/o composite unknownness": "w/o unknownness",
        "w/o conformal calibration": "w/o conformal",
        "w/o dynamic expert expansion": "w/o expansion",
    }
    lines = table.splitlines()
    parsed: Dict[str, Dict[str, object]] = {}
    for index, line in enumerate(lines):
        raw = line.strip()
        if raw not in raw_names:
            continue
        values: List[Result] = []
        rel_params = None
        cursor = index + 1
        while cursor < len(lines) and (len(values) < 4 or rel_params is None):
            match = MEAN_SD_RE.search(lines[cursor])
            if match and len(values) < 4:
                values.append(Result(float(match.group(1)), float(match.group(2))))
            param_match = PARAM_RE.search(lines[cursor])
            if param_match:
                rel_params = float(param_match.group(1))
            cursor += 1
        if len(values) != 4 or rel_params is None:
            raise ValueError(f"Could not parse Table IV row: {raw}")
        parsed[raw_names[raw]] = {
            "hetero_auc": values[0],
            "forgetting": values[1],
            "unknown_auroc": values[2],
            "fcr": values[3],
            "rel_params": rel_params,
        }
    return parsed


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUTPUT_DIR / f"{stem}.{extension}")
    plt.close(fig)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_watermark(fig: plt.Figure) -> None:
    fig.text(
        0.50,
        0.47,
        WATERMARK,
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color="#8A1C1C",
        alpha=0.10,
        rotation=15,
        zorder=100,
    )


def add_method_legend(fig: plt.Figure, y: float = 0.995) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=2.3 if method == "CAPE" else 1.25,
            markersize=4.8,
            label=method,
        )
        for method in DISPLAY_METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=len(handles),
        frameon=False,
        columnspacing=1.05,
        handlelength=2.0,
    )


def normalized_profile(n_stages: int) -> np.ndarray:
    if n_stages == 3:
        raw = np.asarray([-0.85, 0.05, 1.0])
    elif n_stages == 4:
        raw = np.asarray([-1.0, -0.25, 0.25, 1.0])
    elif n_stages == 12:
        raw = np.asarray(
            [-1.15, -0.92, -0.78, -0.64, -0.50, -0.25,
             -0.12, 0.02, 0.28, 0.48, 0.72, 1.00]
        )
    else:
        raw = np.linspace(-1.0, 1.0, n_stages)
    centered = raw - raw.mean()
    profile = centered / centered[-1]
    assert np.isclose(profile.mean(), 0.0)
    assert np.isclose(profile[-1], 1.0)
    return profile


def constrained_curve(trajectory: float, final: float, n_stages: int) -> np.ndarray:
    curve = trajectory + (final - trajectory) * normalized_profile(n_stages)
    assert np.isclose(curve.mean(), trajectory)
    assert np.isclose(curve[-1], final)
    return curve


def figure3_stagewise(table2: Mapping[str, Mapping[str, Sequence[Result]]]) -> None:
    protocols = [
        ("AV1M Pattern-Inc.", 0, ["V-only", "A-only", "A-V"]),
        ("HiFi Generator-Inc.", 2, ["Kling", "Veo", "Wan", "Seedance"]),
        (
            "Heterogeneous Long-Stream",
            4,
            ["FS", "FSGAN", "W2L", "RTVC", "Joint", "V", "A", "A-V", "Kling", "Veo", "Wan", "Seed."],
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 5.05), sharey=True)
    csv_rows = []
    panel_index = 0
    for row, metric in enumerate(("AUC", "AP")):
        for col, (protocol, offset, tasks) in enumerate(protocols):
            ax = axes[row, col]
            x = np.arange(1, len(tasks) + 1)
            if len(tasks) == 12:
                ax.axvspan(0.5, 5.5, color="#EAF3FA", zorder=0)
                ax.axvspan(5.5, 8.5, color="#F7F0E3", zorder=0)
                ax.axvspan(8.5, 12.5, color="#EAF5EA", zorder=0)
                ax.axvline(5.5, color="#888888", linestyle="--", linewidth=0.65)
                ax.axvline(8.5, color="#888888", linestyle="--", linewidth=0.65)
            for method in DISPLAY_METHODS:
                trajectory = table2[metric][method][offset]
                final = table2[metric][method][offset + 1]
                values = constrained_curve(trajectory.mean, final.mean, len(tasks))
                is_cape = method == "CAPE"
                ax.plot(
                    x,
                    values,
                    color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method],
                    linewidth=2.3 if is_cape else 1.1,
                    markersize=4.5 if is_cape else 3.1,
                    markeredgecolor="white" if is_cape else METHOD_COLORS[method],
                    markeredgewidth=0.55,
                    zorder=5 if is_cape else 3,
                )
                for stage, (task, value) in enumerate(zip(tasks, values), start=1):
                    csv_rows.append(
                        {
                            "status": STATUS,
                            "protocol": protocol,
                            "metric": metric,
                            "method": method,
                            "stage": stage,
                            "task": task,
                            "predicted_value": f"{value:.6f}",
                            "source_trajectory_mean": trajectory.mean,
                            "source_final_mean": final.mean,
                        }
                    )
            panel = chr(ord("a") + panel_index)
            panel_index += 1
            ax.set_title(f"({panel}) {protocol} - {metric}", loc="left", fontweight="bold")
            ax.set_ylim(0.50, 1.005)
            ax.set_xlim(0.5, len(tasks) + 0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(tasks, rotation=42 if len(tasks) > 4 else 25, ha="right")
            ax.set_xlabel("Training stage completed")
            if col == 0:
                ax.set_ylabel(f"Average {metric} over seen stages")
            ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.8)
            ax.spines[["top", "right"]].set_visible(False)
    add_method_legend(fig)
    add_watermark(fig)
    fig.text(
        0.5,
        0.006,
        "Predicted stage profiles constrained by manuscript Table II aggregates; no stage-wise SD is inferred.",
        ha="center",
        fontsize=6.9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.13, top=0.91, hspace=0.48, wspace=0.18)
    save_figure(fig, "fig3_stagewise_main_results_predicted")
    write_csv(
        OUTPUT_DIR / "fig3_stagewise_main_results_predicted.csv",
        list(csv_rows[0].keys()),
        csv_rows,
    )


def figure4_open_world(table3: Mapping[str, Sequence[Result]]) -> None:
    methods = list(table3)
    y = np.arange(len(methods))
    cape_index = methods.index("CAPE")
    colors = ["#0072B2" if method == "CAPE" else "#9A9A9A" for method in methods]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.2))

    ax = axes[0, 0]
    auroc = np.asarray([table3[m][0].mean for m in methods])
    aupr = np.asarray([table3[m][1].mean for m in methods])
    auroc_sd = np.asarray([table3[m][0].sd for m in methods])
    aupr_sd = np.asarray([table3[m][1].sd for m in methods])
    ax.barh(y - 0.17, auroc, height=0.32, xerr=auroc_sd, color="#56B4E9", label="AUROC", capsize=2)
    ax.barh(y + 0.17, aupr, height=0.32, xerr=aupr_sd, color="#009E73", label="AUPR", capsize=2)
    ax.set_yticks(y, methods)
    ax.invert_yaxis()
    ax.set_xlim(0.65, 0.95)
    ax.set_title("(a) Unknown detection", loc="left", fontweight="bold")
    ax.set_xlabel("Score (higher is better)")
    ax.legend(frameon=False, ncol=2)

    ax = axes[0, 1]
    fpr = np.asarray([table3[m][2].mean for m in methods])
    fpr_sd = np.asarray([table3[m][2].sd for m in methods])
    ax.barh(y, fpr, xerr=fpr_sd, color=colors, capsize=2)
    ax.set_yticks(y, methods)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.56)
    ax.set_title("(b) False-positive rate at 95% TPR", loc="left", fontweight="bold")
    ax.set_xlabel("FPR95 (lower is better)")

    ax = axes[1, 0]
    delay = np.asarray([table3[m][3].mean for m in methods])
    ari = np.asarray([table3[m][4].mean for m in methods])
    delay_sd = np.asarray([table3[m][3].sd for m in methods])
    ari_sd = np.asarray([table3[m][4].sd for m in methods])
    fcr = np.asarray([table3[m][5].mean for m in methods])
    for i, method in enumerate(methods):
        ax.errorbar(delay[i], ari[i], xerr=delay_sd[i], yerr=ari_sd[i], fmt="none", ecolor=colors[i], alpha=0.65)
        ax.scatter(
            delay[i],
            ari[i],
            s=42 + 230 * fcr[i],
            color=colors[i],
            edgecolor="black" if method == "CAPE" else "white",
            linewidth=0.7,
            zorder=3,
        )
        ax.annotate(method, (delay[i], ari[i]), xytext=(3, 3), textcoords="offset points", fontsize=6.2)
    ax.set_title("(c) Candidate discovery", loc="left", fontweight="bold")
    ax.set_xlabel("Detection delay (lower is better)")
    ax.set_ylabel("ARI (higher is better)")
    ax.grid(color="#DDDDDD", linewidth=0.5)

    ax = axes[1, 1]
    new_auc = np.asarray([table3[m][6].mean for m in methods])
    old_drop = np.asarray([table3[m][7].mean for m in methods])
    new_sd = np.asarray([table3[m][6].sd for m in methods])
    old_sd = np.asarray([table3[m][7].sd for m in methods])
    for i, method in enumerate(methods):
        ax.errorbar(old_drop[i], new_auc[i], xerr=old_sd[i], yerr=new_sd[i], fmt="none", ecolor=colors[i], alpha=0.65)
        ax.scatter(
            old_drop[i],
            new_auc[i],
            s=58,
            color=colors[i],
            edgecolor="black" if method == "CAPE" else "white",
            linewidth=0.7,
            zorder=3,
        )
        ax.annotate(method, (old_drop[i], new_auc[i]), xytext=(3, 3), textcoords="offset points", fontsize=6.2)
    ax.set_title("(d) Unknown-to-known adaptation", loc="left", fontweight="bold")
    ax.set_xlabel("Old-source AUC drop (lower is better)")
    ax.set_ylabel("New-source AUC (higher is better)")
    ax.grid(color="#DDDDDD", linewidth=0.5)

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    add_watermark(fig)
    fig.text(
        0.5,
        0.006,
        "Aggregate means ± sample SD are read from manuscript Table III and must be verified against run artifacts.",
        ha="center",
        fontsize=6.9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.11, top=0.97, hspace=0.37, wspace=0.35)
    save_figure(fig, "fig4_open_world_discovery_manuscript_values")

    metric_names = ["AUROC", "AUPR", "FPR95", "Delay", "ARI", "FCR", "New AUC", "Old Drop"]
    rows = []
    for method in methods:
        for metric, value in zip(metric_names, table3[method]):
            rows.append(
                {
                    "status": STATUS,
                    "method": method,
                    "metric": metric,
                    "mean": value.mean,
                    "sample_sd": value.sd,
                    "source": "cape_ieee_journal.tex:tab:unknown",
                }
            )
    write_csv(OUTPUT_DIR / "fig4_open_world_discovery_manuscript_values.csv", list(rows[0].keys()), rows)


def predicted_routing_matrix() -> np.ndarray:
    matrix = np.asarray(
        [
            [0.48, 0.18, 0.18, 0.06, 0.05, 0.05],
            [0.43, 0.26, 0.17, 0.05, 0.05, 0.04],
            [0.20, 0.43, 0.23, 0.05, 0.05, 0.04],
            [0.17, 0.38, 0.30, 0.06, 0.05, 0.04],
            [0.24, 0.26, 0.39, 0.04, 0.04, 0.03],
            [0.45, 0.16, 0.25, 0.06, 0.04, 0.04],
            [0.15, 0.45, 0.25, 0.06, 0.05, 0.04],
            [0.23, 0.24, 0.42, 0.04, 0.04, 0.03],
            [0.10, 0.10, 0.24, 0.38, 0.10, 0.08],
            [0.08, 0.08, 0.22, 0.29, 0.24, 0.09],
            [0.07, 0.07, 0.20, 0.23, 0.34, 0.09],
            [0.06, 0.06, 0.18, 0.20, 0.22, 0.28],
        ],
        dtype=float,
    )
    return matrix / matrix.sum(axis=1, keepdims=True)


def figure5_mechanism(table2: Mapping[str, Mapping[str, Sequence[Result]]]) -> None:
    cape_traj = table2["AUC"]["CAPE"][4].mean
    cape_final = table2["AUC"]["CAPE"][5].mean
    stage_auc = constrained_curve(cape_traj, cape_final, 12)
    task_labels = ["FS", "FSGAN", "W2L", "RTVC", "Joint", "V", "A", "A-V", "Kling", "Veo", "Wan", "Seed."]
    matrix = np.full((12, 12), np.nan)
    for b in range(12):
        offsets = np.linspace(-0.014, 0.014, b + 1)
        offsets -= offsets.mean()
        matrix[b, : b + 1] = np.clip(stage_auc[b] + offsets, 0.5, 1.0)
    routing = predicted_routing_matrix()
    reuse = (routing >= 0.20).sum(axis=0)
    active_experts = np.asarray([2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 6])
    rel_params = 0.55 + 0.45 * (active_experts - 1) / 5

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.5))

    ax = axes[0, 0]
    im = ax.imshow(np.ma.masked_invalid(matrix), vmin=0.50, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_title("(a) Predicted post-stage AUC matrix", loc="left", fontweight="bold")
    ax.set_xlabel("Evaluation task")
    ax.set_ylabel("Training stage completed")
    ax.set_xticks(np.arange(12), task_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(12), np.arange(1, 13))
    cbar_auc = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.025)
    cbar_auc.set_label("AUC", labelpad=2)

    ax = axes[0, 1]
    im2 = ax.imshow(routing, vmin=0, vmax=0.50, cmap="YlGnBu", aspect="auto")
    ax.set_title("(b) Predicted task-to-expert routing", loc="left", fontweight="bold")
    ax.set_xlabel("Pattern expert")
    ax.set_ylabel("Task", labelpad=2)
    ax.set_xticks(np.arange(6), [f"E{i}" for i in range(1, 7)])
    ax.set_yticks(np.arange(12), task_labels)
    cbar_route = fig.colorbar(im2, ax=ax, fraction=0.040, pad=0.025)
    cbar_route.set_label("Routing mass", labelpad=2)

    ax = axes[1, 0]
    bars = ax.bar(np.arange(1, 7), reuse, color="#009E73", edgecolor="white")
    for bar, value in zip(bars, reuse):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.12, str(int(value)), ha="center", fontsize=7)
    ax.set_title("(c) Predicted cross-stage expert reuse", loc="left", fontweight="bold")
    ax.set_xlabel("Pattern expert")
    ax.set_ylabel("Stages with routing mass ≥ 0.20")
    ax.set_xticks(np.arange(1, 7))
    ax.set_ylim(0, max(reuse) + 1.2)

    ax = axes[1, 1]
    x = np.arange(1, 13)
    line1 = ax.plot(x, stage_auc, color="#0072B2", marker="o", label="Average AUC")
    ax.set_xlabel("Training stage completed")
    ax.set_ylabel("Average AUC", color="#0072B2")
    ax.tick_params(axis="y", labelcolor="#0072B2")
    ax.set_ylim(min(stage_auc) - 0.025, max(stage_auc) + 0.025)
    ax2 = ax.twinx()
    line2 = ax2.plot(x, rel_params, color="#D55E00", marker="s", linestyle="--", label="Relative parameters")
    ax2.set_ylabel("Relative parameters", color="#D55E00")
    ax2.tick_params(axis="y", labelcolor="#D55E00")
    ax2.set_ylim(0.45, 1.05)
    ax.set_title("(d) Predicted capacity-retention evolution", loc="left", fontweight="bold")
    ax.legend(line1 + line2, [item.get_label() for item in line1 + line2], frameon=False, loc="lower left")

    for ax in (axes[1, 0], axes[1, 1]):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    add_watermark(fig)
    fig.text(
        0.5,
        0.006,
        "Mechanism panels are expected trends, not measured routing or performance matrices.",
        ha="center",
        fontsize=6.9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.11, top=0.97, hspace=0.42, wspace=0.42)
    save_figure(fig, "fig5_retention_and_expert_mechanism_predicted")

    rows = []
    for stage in range(12):
        rows.append(
            {
                "status": STATUS,
                "record_type": "stage_summary",
                "stage": stage + 1,
                "task": task_labels[stage],
                "average_auc": f"{stage_auc[stage]:.6f}",
                "active_experts": active_experts[stage],
                "relative_params": f"{rel_params[stage]:.4f}",
                "expert": "",
                "routing_mass": "",
            }
        )
        for expert in range(6):
            rows.append(
                {
                    "status": STATUS,
                    "record_type": "routing",
                    "stage": stage + 1,
                    "task": task_labels[stage],
                    "average_auc": "",
                    "active_experts": "",
                    "relative_params": "",
                    "expert": f"E{expert + 1}",
                    "routing_mass": f"{routing[stage, expert]:.6f}",
                }
            )
    write_csv(OUTPUT_DIR / "fig5_retention_and_expert_mechanism_predicted.csv", list(rows[0].keys()), rows)


def robustness_curve(base: float, x: np.ndarray, max_distance: float, degradation: float) -> np.ndarray:
    return base - degradation * (np.abs(x) / max_distance) ** 1.25


def figure6_robustness(table2: Mapping[str, Mapping[str, Sequence[Result]]]) -> None:
    methods = ["Seq-FT", "DER++", "MoE-Adapters", "CAPE"]
    base_ap = {method: table2["AP"][method][5].mean for method in methods}
    sensitivity = {"Seq-FT": 1.00, "DER++": 0.76, "MoE-Adapters": 0.62, "CAPE": 0.38}
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.3))
    rows = []

    shifts = np.asarray([-8, -4, -2, 0, 2, 4, 8], dtype=float)
    ax = axes[0, 0]
    for method in methods:
        values = robustness_curve(base_ap[method], shifts, 8.0, 0.15 * sensitivity[method])
        ax.plot(shifts, values, color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], label=method)
        for x, value in zip(shifts, values):
            rows.append({"status": STATUS, "condition": "temporal_shift", "method": method, "level": x, "predicted_ap": value})
    ax.set_title("(a) Audio-video temporal shift", loc="left", fontweight="bold")
    ax.set_xlabel("Audio shift (feature steps)")
    ax.set_ylabel("Predicted AP")

    jpeg = np.asarray([100, 90, 80, 70, 60, 50, 40, 30], dtype=float)
    ax = axes[0, 1]
    for method in methods:
        distance = 100 - jpeg
        values = base_ap[method] - 0.18 * sensitivity[method] * (distance / 70) ** 1.15
        ax.plot(jpeg, values, color=METHOD_COLORS[method], marker=METHOD_MARKERS[method])
        for x, value in zip(jpeg, values):
            rows.append({"status": STATUS, "condition": "jpeg_quality", "method": method, "level": x, "predicted_ap": value})
    ax.invert_xaxis()
    ax.set_title("(b) JPEG compression", loc="left", fontweight="bold")
    ax.set_xlabel("JPEG quality factor")
    ax.set_ylabel("Predicted AP")

    blur = np.asarray([0, 1, 2, 3, 4], dtype=float)
    ax = axes[1, 0]
    for method in methods:
        values = base_ap[method] - 0.17 * sensitivity[method] * (blur / 4) ** 1.20
        ax.plot(blur, values, color=METHOD_COLORS[method], marker=METHOD_MARKERS[method])
        for x, value in zip(blur, values):
            rows.append({"status": STATUS, "condition": "gaussian_blur", "method": method, "level": x, "predicted_ap": value})
    ax.set_title("(c) Gaussian blur", loc="left", fontweight="bold")
    ax.set_xlabel(r"Blur strength $\sigma$")
    ax.set_ylabel("Predicted AP")

    conditions = ["Full A-V", "No audio", "No video", "Desync"]
    condition_drop = {
        "Seq-FT": [0.00, 0.16, 0.21, 0.13],
        "DER++": [0.00, 0.12, 0.16, 0.10],
        "MoE-Adapters": [0.00, 0.09, 0.13, 0.08],
        "CAPE": [0.00, 0.055, 0.075, 0.045],
    }
    ax = axes[1, 1]
    x = np.arange(len(conditions))
    width = 0.19
    for index, method in enumerate(methods):
        values = base_ap[method] - np.asarray(condition_drop[method])
        ax.bar(x + (index - 1.5) * width, values, width=width, color=METHOD_COLORS[method], label=method)
        for condition, value in zip(conditions, values):
            rows.append({"status": STATUS, "condition": "missing_modality", "method": method, "level": condition, "predicted_ap": value})
    ax.set_xticks(x, conditions, rotation=18, ha="right")
    ax.set_title("(d) Missing/corrupted modalities", loc="left", fontweight="bold")
    ax.set_ylabel("Predicted AP")

    for ax in axes.flat:
        ax.set_ylim(0.32, 0.84)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], marker=METHOD_MARKERS[m], label=m)
        for m in methods
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False)
    add_watermark(fig)
    fig.text(
        0.5,
        0.006,
        "All robustness profiles are expected trends and must be replaced by measured perturbation runs.",
        ha="center",
        fontsize=6.9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.12, top=0.91, hspace=0.42, wspace=0.22)
    save_figure(fig, "fig6_multimedia_robustness_predicted")
    write_csv(OUTPUT_DIR / "fig6_multimedia_robustness_predicted.csv", list(rows[0].keys()), rows)


def figure7_efficiency(
    table2: Mapping[str, Mapping[str, Sequence[Result]]],
    table4: Mapping[str, Mapping[str, object]],
) -> None:
    full_auc = float(table4["Full CAPE"]["hetero_auc"].mean)
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.25))
    rows = []

    ax = axes[0, 0]
    capacities = np.asarray([0, 256, 512, 1024, 2048])
    memory_auc = np.asarray([0.825, 0.846, 0.861, full_auc, full_auc + 0.002])
    ax.plot(capacities, memory_auc, color="#0072B2", marker="o", linewidth=2)
    ax.axvline(1024, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_title("(a) Predicted replay-memory sensitivity", loc="left", fontweight="bold")
    ax.set_xlabel("Replay capacity")
    ax.set_ylabel("Heterogeneous trajectory AUC")
    for x, value in zip(capacities, memory_auc):
        rows.append({"status": STATUS, "record": "replay_capacity", "x": x, "y": "", "value": value, "label": "CAPE"})

    ax = axes[0, 1]
    top_k = [1, 2, 3, 4]
    widths = [32, 64, 128]
    sensitivity = np.asarray(
        [
            [full_auc - 0.020, full_auc - 0.010, full_auc - 0.009],
            [full_auc - 0.009, full_auc, full_auc - 0.002],
            [full_auc - 0.012, full_auc - 0.004, full_auc - 0.003],
            [full_auc - 0.018, full_auc - 0.010, full_auc - 0.008],
        ]
    )
    im = ax.imshow(sensitivity, cmap="YlGnBu", vmin=full_auc - 0.025, vmax=full_auc + 0.003, aspect="auto")
    for i in range(len(top_k)):
        for j in range(len(widths)):
            ax.text(j, i, f"{sensitivity[i, j]:.3f}", ha="center", va="center", fontsize=7)
            rows.append({"status": STATUS, "record": "topk_width", "x": top_k[i], "y": widths[j], "value": sensitivity[i, j], "label": "CAPE"})
    ax.set_xticks(np.arange(len(widths)), widths)
    ax.set_yticks(np.arange(len(top_k)), top_k)
    ax.set_xlabel("Expert bottleneck width")
    ax.set_ylabel("Router top-k")
    ax.set_title("(b) Predicted routing-capacity sensitivity", loc="left", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Trajectory AUC")

    ax = axes[1, 0]
    assumed_params = {
        "Seq-FT": 0.72,
        "DER++": 0.78,
        "CODA-Prompt": 0.87,
        "MoE-Adapters": 0.93,
        "CAPE": 1.00,
    }
    assumed_latency = {
        "Seq-FT": 0.72,
        "DER++": 0.79,
        "CODA-Prompt": 0.90,
        "MoE-Adapters": 0.96,
        "CAPE": 1.00,
    }
    for method in DISPLAY_METHODS:
        auc = table2["AUC"][method][4].mean
        size = 45 + 105 * assumed_latency[method]
        ax.scatter(
            assumed_params[method],
            auc,
            s=size,
            color=METHOD_COLORS[method],
            edgecolor="black" if method == "CAPE" else "white",
            linewidth=0.7,
            zorder=3,
        )
        ax.annotate(method, (assumed_params[method], auc), xytext=(3, 3), textcoords="offset points", fontsize=6.3)
        rows.append({"status": STATUS, "record": "capacity_tradeoff", "x": assumed_params[method], "y": assumed_latency[method], "value": auc, "label": method})
    ax.set_title("(c) Expected capacity-performance tradeoff", loc="left", fontweight="bold")
    ax.set_xlabel("Assumed relative parameters")
    ax.set_ylabel("Table II heterogeneous trajectory AUC")

    ax = axes[1, 1]
    variants = list(table4)
    auc = np.asarray([table4[v]["hetero_auc"].mean for v in variants])
    fcr = np.asarray([table4[v]["fcr"].mean for v in variants])
    params = np.asarray([float(table4[v]["rel_params"]) for v in variants])
    for i, variant in enumerate(variants):
        color = "#0072B2" if variant == "Full CAPE" else "#9A9A9A"
        ax.scatter(fcr[i], auc[i], s=55 + 75 * params[i], color=color, edgecolor="white", linewidth=0.6)
        ax.annotate(variant, (fcr[i], auc[i]), xytext=(3, 3), textcoords="offset points", fontsize=5.8)
        rows.append({"status": STATUS, "record": "ablation_pareto", "x": fcr[i], "y": params[i], "value": auc[i], "label": variant})
    ax.set_title("(d) Manuscript ablation Pareto view", loc="left", fontweight="bold")
    ax.set_xlabel("FCR (lower is better)")
    ax.set_ylabel("Heterogeneous trajectory AUC")

    for ax in axes.flat:
        ax.grid(color="#DDDDDD", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    add_watermark(fig)
    fig.text(
        0.5,
        0.006,
        "Sensitivity and compute values are planning assumptions; Table IV ablations are manuscript aggregates awaiting artifact verification.",
        ha="center",
        fontsize=6.7,
        color="#555555",
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.97, hspace=0.40, wspace=0.30)
    save_figure(fig, "fig7_efficiency_sensitivity_and_ablation_draft")
    write_csv(OUTPUT_DIR / "fig7_efficiency_sensitivity_and_ablation_draft.csv", list(rows[0].keys()), rows)


def write_readme() -> None:
    content = """# CAPE TMM Experiment Figure Planning Package

## Integrity status

**These files are planning drafts, not submission-ready experimental results.**

- Every PDF/PNG/SVG is visibly watermarked.
- Figure 3 is constrained by Table II trajectory/final aggregates but its
  intermediate stage values are predictions.
- Figure 4 redraws the aggregate values currently written in Table III; those
  values still require verification against real run artifacts.
- Figure 5 contains predicted performance-matrix and routing behavior.
- Figure 6 is entirely predicted robustness behavior.
- Figure 7 mixes predicted sensitivity/compute behavior with aggregate values
  currently written in Tables II and IV.

Do not remove the watermark or insert these panels as experimental evidence
until each CSV row has been replaced by statistics derived from real runs.

## Expected trends and evaluation

### Figure 3 - Stage-wise main results

Expected appearance:
- CAPE should remain the upper curve on the 12-stage stream.
- Seq-FT should show the strongest decline as tasks accumulate.
- Larger changes are expected at the FakeAVCeleb→AV1M and AV1M→HiFi
  boundaries.
- DER++, CODA-Prompt, and MoE-Adapters should occupy progressively stronger
  intermediate positions.

Evaluation:
- This is the central TMM figure because it directly demonstrates plasticity
  and retention.
- Real curves should include five-seed sample-SD bands.
- Perfectly smooth or parallel real curves would be suspicious; genuine
  per-stage results should reflect task difficulty and seed variation.

### Figure 4 - Open-world discovery

Expected appearance:
- CAPE should be toward the high-AUROC/high-AUPR and low-FPR95 region.
- In candidate discovery it should combine shorter delay, higher ARI, and
  lower FCR.
- In adaptation it should be near the high-new-AUC/low-old-drop corner.

Evaluation:
- This figure is essential to distinguish CAPE from ordinary continual
  classification.
- The final version must report fold-then-seed aggregation exactly as defined
  in the manuscript.

### Figure 5 - Retention and expert mechanism

Expected appearance:
- The performance matrix should remain bright below the diagonal rather than
  fading strongly on early tasks.
- Routing should be structured but not an identity matrix: related modality
  patterns should reuse experts, while generator stages should activate
  additional experts.
- Capacity should grow stepwise while performance declines more slowly than
  fixed-capacity alternatives.

Evaluation:
- A completely uniform routing map would imply no specialization.
- A perfectly diagonal map would imply no expert reuse.
- The desired evidence is moderate specialization plus cross-stage reuse.

### Figure 6 - Multimedia robustness

Expected appearance:
- Temporal-shift performance should peak at zero and decline approximately
  symmetrically.
- JPEG and blur performance should decline monotonically with perturbation
  strength.
- CAPE should have the flattest degradation, not an unrealistically constant
  curve.
- Removing video is expected to hurt more than removing audio, while CAPE
  should retain the smallest relative drop.

Evaluation:
- TMM reviewers value realistic multimedia corruption tests.
- All perturbations must share identical samples and preprocessing across
  methods.

### Figure 7 - Efficiency, sensitivity, and ablation

Expected appearance:
- Replay gains should saturate around the manuscript budget K=1024.
- Router top-k=2 and bottleneck width=64 should form a broad optimum rather
  than a single implausibly sharp point.
- CAPE should show a defensible performance/capacity tradeoff.
- Removing retention or dynamic expansion should cause large retention
  damage; removing unknownness should mainly damage unknown AUROC/FCR.

Evaluation:
- Measured parameter counts, FLOPs, latency, and GPU memory must replace the
  assumed values.
- Latency should be measured with warm-up, fixed batch size, and synchronized
  CUDA timing.

## Files

- `fig3_stagewise_main_results_predicted.*`
- `fig4_open_world_discovery_manuscript_values.*`
- `fig5_retention_and_expert_mechanism_predicted.*`
- `fig6_multimedia_robustness_predicted.*`
- `fig7_efficiency_sensitivity_and_ablation_draft.*`
- One CSV accompanies each figure.

## Reproduction

```powershell
cd G:\\dimodif-main\\dimodif-main
C:\\Users\\52948\\miniconda3\\envs\\deepfake3\\python.exe `
  scripts\\cape_experiments\\generate_predicted_tmm_experiment_figures.py
```
"""
    (OUTPUT_DIR / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tex = TEX_PATH.read_text(encoding="utf-8")
    table2 = parse_table2(tex)
    table3 = parse_table3(tex)
    table4 = parse_table4(tex)

    figure3_stagewise(table2)
    figure4_open_world(table3)
    figure5_mechanism(table2)
    figure6_robustness(table2)
    figure7_efficiency(table2, table4)
    write_readme()

    manifest = {
        "status": STATUS,
        "source_tex": str(TEX_PATH),
        "output_dir": str(OUTPUT_DIR),
        "figures": [
            "fig3_stagewise_main_results_predicted",
            "fig4_open_world_discovery_manuscript_values",
            "fig5_retention_and_expert_mechanism_predicted",
            "fig6_multimedia_robustness_predicted",
            "fig7_efficiency_sensitivity_and_ablation_draft",
        ],
        "warning": "Do not use as experimental evidence until replaced by real run-derived values.",
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Generated five TMM planning figures in: {OUTPUT_DIR}")
    print("Status: planning drafts; not verified experimental results")


if __name__ == "__main__":
    main()
