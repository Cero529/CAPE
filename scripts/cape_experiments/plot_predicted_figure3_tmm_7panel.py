"""Create a seven-panel TMM-style Figure 3 planning preview for CAPE.

The preview follows a 2 x 4 small-multiples layout:

    AV1M AUC | HiFi AUC | Heterogeneous AUC | shared legend
    AV1M AP  | HiFi AP  | Heterogeneous AP  | Table-II AUC gap

IMPORTANT
---------
This script does not claim to reconstruct unobserved experimental results.
The stage-wise AUC/AP profiles are illustrative curves constrained so that:

1. every curve mean exactly equals the corresponding trajectory mean currently
   written in manuscript Table II;
2. every final point exactly equals the corresponding final mean in Table II;
3. every AV1M curve is strictly decreasing, with its start analytically solved
   from the selected curvature, Table-II trajectory mean, and Table-II final.

The reference-trend preview gives each method family its own low-frequency
trajectory.  Following the supplied continual-learning reference figure,
weak methods receive a larger early shock, stable methods flatten later, and
selected replay/prompt/expert methods may show one small, method-specific
recovery.  CAPE receives two mild post-boundary recoveries on the 12-stage
stream.  The construction forbids high-frequency zigzags and synchronized
turning points.

The manuscript currently reports a measured forgetting value only for CAPE;
it does not provide baseline forgetting values.  The seventh panel therefore
does not fabricate a baseline-forgetting comparison.  Instead, it reports the
heterogeneous trajectory-to-final AUC gap, computed exactly from Table II for
every displayed method.  Replace the six predicted stage profiles with real
multi-seed run artifacts before using the figure as experimental evidence.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

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
OUTPUT_STEM = "figure3_tmm_7panel_final_merged_table_aligned_predicted_preview"

STATUS = "planning_draft_not_verified_experimental_result"
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

# Nine representative baselines plus CAPE.
DISPLAY_METHODS = [
    "Seq-FT",
    "EWC",
    "LwF",
    "ER",
    "DER++",
    "CODA-Prompt",
    "DyTox",
    "PROOF",
    "MoE-Adapters",
    "CAPE",
]

METHOD_COLORS = {
    "Seq-FT": "#7F7F7F",
    "EWC": "#8C564B",
    "LwF": "#BCBD22",
    "ER": "#17BECF",
    "DER++": "#E69F00",
    "CODA-Prompt": "#009E73",
    "DyTox": "#9467BD",
    "PROOF": "#D55E00",
    "MoE-Adapters": "#CC79A7",
    "CAPE": "#0072B2",
}

METHOD_MARKERS = {
    "Seq-FT": "o",
    "EWC": "s",
    "LwF": "^",
    "ER": "v",
    "DER++": "P",
    "CODA-Prompt": "D",
    "DyTox": ">",
    "PROOF": "<",
    "MoE-Adapters": "X",
    "CAPE": "o",
}

METHOD_LINESTYLES = {
    "Seq-FT": "-",
    "EWC": "--",
    "LwF": "-.",
    "ER": ":",
    "DER++": "-",
    "CODA-Prompt": "--",
    "DyTox": "-.",
    "PROOF": ":",
    "MoE-Adapters": "--",
    "CAPE": "-",
}

# HiFi and heterogeneous stage-one values are planning assumptions.  AV1M
# starts are not stored here: they are solved analytically from the new
# Table-II trajectory/final values and the method-specific curvature below.
START_VALUES = {
    ("HiFi Generator-Incremental", "AUC"): {
        "Seq-FT": 0.984,
        "EWC": 0.987,
        "LwF": 0.990,
        "ER": 0.992,
        "DER++": 0.995,
        "CODA-Prompt": 0.996,
        "DyTox": 0.993,
        "PROOF": 0.998,
        "MoE-Adapters": 0.999,
        "CAPE": 0.997,
    },
    ("HiFi Generator-Incremental", "AP"): {
        "Seq-FT": 0.970,
        "EWC": 0.972,
        "LwF": 0.974,
        "ER": 0.976,
        "DER++": 0.978,
        "CODA-Prompt": 0.980,
        "DyTox": 0.977,
        "PROOF": 0.982,
        "MoE-Adapters": 0.974,
        "CAPE": 0.980,
    },
    ("Heterogeneous Long-Stream", "AUC"): {
        "Seq-FT": 0.912,
        "EWC": 0.914,
        "LwF": 0.916,
        "ER": 0.918,
        "DER++": 0.920,
        "CODA-Prompt": 0.923,
        "DyTox": 0.921,
        "PROOF": 0.929,
        "MoE-Adapters": 0.932,
        "CAPE": 0.928,
    },
    ("Heterogeneous Long-Stream", "AP"): {
        "Seq-FT": 0.896,
        "EWC": 0.898,
        "LwF": 0.900,
        "ER": 0.902,
        "DER++": 0.904,
        "CODA-Prompt": 0.907,
        "DyTox": 0.905,
        "PROOF": 0.912,
        "MoE-Adapters": 0.916,
        "CAPE": 0.913,
    },
}

AV1M_PROTOCOL = "AV1M Pattern-Incremental"

# Fraction of the total Stage-1-to-Stage-3 decline completed at Stage 2.
# Different fractions avoid synchronized slopes while keeping every AV1M
# trajectory smooth and strictly decreasing.
AV1M_MIDDLE_PROGRESS = {
    "AUC": {
        "Seq-FT": 0.72,
        "EWC": 0.78,
        "LwF": 0.80,
        "ER": 0.86,
        "DER++": 0.82,
        "CODA-Prompt": 0.80,
        "DyTox": 0.76,
        "PROOF": 0.82,
        "MoE-Adapters": 0.80,
        "CAPE": 0.72,
    },
    "AP": {
        "Seq-FT": 0.74,
        "EWC": 0.78,
        "LwF": 0.80,
        "ER": 0.86,
        "DER++": 0.84,
        "CODA-Prompt": 0.80,
        "DyTox": 0.78,
        "PROOF": 0.82,
        "MoE-Adapters": 0.80,
        "CAPE": 0.74,
    },
}

HIFI_PROTOCOL = "HiFi Generator-Incremental"

# Normalized separation between the two HiFi interior points.  Together with
# the fixed start, Table-II trajectory mean, and Table-II final value, this
# single degree of freedom creates a method-specific early-shock/plateau shape
# without changing either manuscript aggregate.
HIFI_INTERIOR_GAPS = {
    "AUC": {
        "Seq-FT": 0.22,
        "EWC": 0.18,
        "LwF": 0.14,
        "ER": 0.10,
        "DER++": 0.12,
        "CODA-Prompt": 0.08,
        "DyTox": 0.16,
        "PROOF": 0.10,
        "MoE-Adapters": 0.07,
        "CAPE": 0.06,
    },
    "AP": {
        "Seq-FT": 0.22,
        "EWC": 0.18,
        "LwF": 0.14,
        "ER": 0.10,
        "DER++": 0.12,
        "CODA-Prompt": 0.08,
        "DyTox": 0.16,
        "PROOF": 0.10,
        "MoE-Adapters": 0.015,
        "CAPE": 0.025,
    },
}

# One broad, low-frequency feature per method.  The feature is converted to a
# zero-endpoint, zero-mean basis before use, so it changes the location of a
# plateau without changing the Table-II trajectory mean or final value.
METHOD_SHAPE_SPECS = {
    "Seq-FT": (0.23, 0.20, 0.025),
    "EWC": (0.34, 0.22, -0.020),
    "LwF": (0.46, 0.18, 0.018),
    "ER": (0.58, 0.20, -0.022),
    "DER++": (0.72, 0.18, 0.018),
    "CODA-Prompt": (0.38, 0.16, -0.016),
    "DyTox": (0.55, 0.16, 0.020),
    "PROOF": (0.78, 0.18, -0.014),
    "MoE-Adapters": (0.66, 0.22, 0.010),
    "CAPE": (0.60, 0.24, 0.0035),
}

# Method-specific normalized reference trajectories, inspired by the supplied
# long-tailed continual-learning figure.  They encode qualitative mechanisms,
# not measured CAPE results:
#   - Seq-FT/EWC: large early shock followed by a low-slope tail;
#   - distillation/replay/prompt methods: one plateau or mild recovery at a
#     method-specific stage;
#   - MoE-Adapters/CAPE: smaller early slope and stronger late retention.
# Values are normalized progress from the planning start (0) to final (1).
NATURAL_REFERENCE_PROGRESS_12 = {
    "Seq-FT": [0.00, 0.18, 0.32, 0.44, 0.54, 0.63, 0.70, 0.76, 0.82, 0.87, 0.93, 1.00],
    "EWC": [0.00, 0.22, 0.39, 0.52, 0.62, 0.69, 0.74, 0.78, 0.81, 0.84, 0.90, 1.00],
    "LwF": [0.00, 0.16, 0.29, 0.27, 0.39, 0.50, 0.59, 0.67, 0.74, 0.82, 0.91, 1.00],
    "ER": [0.00, 0.13, 0.25, 0.36, 0.34, 0.45, 0.56, 0.65, 0.73, 0.81, 0.90, 1.00],
    "DER++": [0.00, 0.11, 0.22, 0.33, 0.43, 0.52, 0.50, 0.60, 0.70, 0.79, 0.89, 1.00],
    "CODA-Prompt": [0.00, 0.09, 0.19, 0.29, 0.39, 0.48, 0.56, 0.56, 0.65, 0.75, 0.87, 1.00],
    "DyTox": [0.00, 0.10, 0.21, 0.32, 0.43, 0.53, 0.62, 0.70, 0.78, 0.75, 0.86, 1.00],
    "PROOF": [0.00, 0.08, 0.18, 0.28, 0.38, 0.36, 0.47, 0.58, 0.68, 0.78, 0.89, 1.00],
    "MoE-Adapters": [0.00, 0.07, 0.16, 0.25, 0.34, 0.43, 0.43, 0.52, 0.62, 0.73, 0.86, 1.00],
    "CAPE": [0.00, 0.10, 0.20, 0.30, 0.40, 0.51, 0.49, 0.60, 0.71, 0.69, 0.83, 1.00],
}

MAX_LOCAL_RECOVERIES = {
    "Seq-FT": 0,
    "EWC": 0,
    "LwF": 1,
    "ER": 1,
    "DER++": 1,
    "CODA-Prompt": 0,
    "DyTox": 1,
    "PROOF": 1,
    "MoE-Adapters": 0,
    "CAPE": 2,
}

RESULT_RE = re.compile(r"\\tbl(?:res|best)\{([0-9.]+)\}\{([0-9.]+)\}")


@dataclass(frozen=True)
class Result:
    mean: float
    sd: float


def configure_style() -> None:
    """Apply compact IEEE/TMM-friendly typography and vector settings."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 6.0,
            "axes.labelsize": 6.2,
            "axes.titlesize": 6.5,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "legend.fontsize": 5.3,
            "axes.linewidth": 0.65,
            "lines.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
        }
    )


def extract_segment(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
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
    auc_panel = extract_segment(tex, "% Panel (a): AUC", "% Panel (b): AP")
    ap_panel = extract_segment(tex, "% Panel (b): AP", r"\end{table*}")
    return {
        "AUC": parse_table2_panel(auc_panel),
        "AP": parse_table2_panel(ap_panel),
    }


def desired_start(protocol: str, metric: str, method: str) -> float:
    """Return the protocol-specific planning start value."""
    return START_VALUES[(protocol, metric)][method]


def av1m_declining_curve(
    trajectory_mean: float, final_mean: float, metric: str, method: str
) -> np.ndarray:
    """Return a strictly decreasing AV1M curve aligned to the new Table II.

    For middle progress fraction ``p``, the middle point is
    ``start + p * (final - start)``.  Solving the three-point arithmetic-mean
    equation gives the start below, leaving no aggregate mismatch.
    """
    middle_progress = AV1M_MIDDLE_PROGRESS[metric][method]
    start = (
        3.0 * trajectory_mean - (1.0 + middle_progress) * final_mean
    ) / (2.0 - middle_progress)
    middle = start + middle_progress * (final_mean - start)
    curve = np.asarray([start, middle, final_mean], dtype=float)
    if not np.all(np.diff(curve) < -1e-10):
        raise AssertionError(f"AV1M curve is not strictly decreasing: {metric}, {method}")
    if not np.isclose(curve.mean(), trajectory_mean, atol=1e-12):
        raise AssertionError(f"AV1M curve mean mismatch: {metric}, {method}")
    return curve


def hifi_curved_curve(
    trajectory_mean: float, final_mean: float, metric: str, method: str
) -> np.ndarray:
    """Return a curved four-stage HiFi profile with exact mean and endpoint.

    The two interior normalized progress values have a method-specific gap.
    Their sum is solved analytically from the required arithmetic mean, so the
    curve remains exactly aligned with both HiFi entries in Table II.
    """
    start = desired_start(HIFI_PROTOCOL, metric, method)
    total_drop = start - final_mean
    if total_drop <= 0.0:
        raise ValueError(f"HiFi start must exceed final value: {metric}, {method}")

    interior_sum = 4.0 * trajectory_mean - start - final_mean
    normalized_progress_sum = (2.0 * start - interior_sum) / total_drop
    gap = HIFI_INTERIOR_GAPS[metric][method]
    progress_2 = 0.5 * (normalized_progress_sum - gap)
    progress_3 = 0.5 * (normalized_progress_sum + gap)
    if not 0.0 < progress_2 < progress_3 < 1.0:
        raise AssertionError(
            f"Infeasible HiFi interior progress: {metric}, {method}, "
            f"{progress_2:.4f}, {progress_3:.4f}"
        )

    curve = np.asarray(
        [
            start,
            start - total_drop * progress_2,
            start - total_drop * progress_3,
            final_mean,
        ],
        dtype=float,
    )
    if not np.all(np.diff(curve) < -1e-10):
        raise AssertionError(f"HiFi curve is not strictly decreasing: {metric}, {method}")
    if not np.isclose(curve.mean(), trajectory_mean, atol=1e-12):
        raise AssertionError(f"HiFi curve mean mismatch: {metric}, {method}")
    return curve


def mean_matched_power_progress(n_stages: int, target_mean: float) -> np.ndarray:
    """Return a monotone power curve with fixed endpoints and exact mean.

    For equally spaced ``x`` in [0, 1], ``mean(x**q)`` decreases continuously
    from (n-1)/n to 1/n as q increases.  A bisection search therefore gives a
    low-curvature progress profile satisfying the required aggregate without
    inserting artificial stage-wise oscillations.
    """
    lower_bound = 1.0 / n_stages
    upper_bound = (n_stages - 1.0) / n_stages
    if not lower_bound - 1e-12 <= target_mean <= upper_bound + 1e-12:
        raise ValueError(
            f"Progress mean {target_mean:.6f} is outside the monotone "
            f"{n_stages}-stage range [{lower_bound:.6f}, {upper_bound:.6f}]"
        )

    x = np.linspace(0.0, 1.0, n_stages)
    low_q, high_q = 1e-4, 100.0
    for _ in range(160):
        mid_q = 0.5 * (low_q + high_q)
        candidate_mean = float(np.mean(x**mid_q))
        if candidate_mean > target_mean:
            low_q = mid_q
        else:
            high_q = mid_q

    progress = x ** (0.5 * (low_q + high_q))
    progress[0] = 0.0
    progress[-1] = 1.0

    # Remove the last floating-point residual without moving either endpoint.
    if n_stages > 2:
        progress[1:-1] += (
            (target_mean - float(progress.mean())) * n_stages / (n_stages - 2)
        )
    return progress


def zero_mean_family_feature(
    n_stages: int, metric: str, method: str
) -> tuple[np.ndarray, float]:
    """Return a broad method-specific feature with zero endpoints and mean."""
    if n_stages <= 3:
        return np.zeros(n_stages), 0.0

    center, width, amplitude = METHOD_SHAPE_SPECS[method]
    method_index = DISPLAY_METHODS.index(method)
    if metric == "AP":
        # AP is not a vertical copy of AUC: alternate the feature direction
        # and shift its plateau location by a small, deterministic amount.
        center += 0.06 if method_index % 2 == 0 else -0.045
        center = float(np.clip(center, 0.16, 0.84))
        amplitude *= -0.80 if method_index % 2 == 0 else 0.75

    x = np.linspace(0.0, 1.0, n_stages)
    window = np.sin(np.pi * x) ** 1.5
    local = np.exp(-0.5 * ((x - center) / width) ** 2)
    centering = float(np.sum(window * local) / np.sum(window))
    feature = window * (local - centering)
    feature[0] = 0.0
    feature[-1] = 0.0
    max_abs = float(np.max(np.abs(feature)))
    if max_abs > 0:
        feature /= max_abs
    return feature, amplitude


def method_family_progress(
    n_stages: int, target_mean: float, metric: str, method: str
) -> np.ndarray:
    """Build a reference-like method trajectory with limited recoveries."""
    base = mean_matched_power_progress(n_stages, target_mean)

    # With three points, fixed start/mean/final already determine the sole
    # interior point.  No additional shape can be introduced without changing
    # a manuscript aggregate.
    if n_stages <= 3:
        return base

    full_reference = np.asarray(
        NATURAL_REFERENCE_PROGRESS_12[method], dtype=float
    )
    reference_x = np.linspace(0.0, 1.0, len(full_reference))
    target_x = np.linspace(0.0, 1.0, n_stages)
    reference = np.interp(target_x, reference_x, full_reference)
    reference[0] = 0.0
    reference[-1] = 1.0

    # Subtract a power curve having the same mean.  The resulting reference
    # feature has zero endpoints and zero mean, so it can alter local slopes
    # without altering the manuscript trajectory/final constraints.
    matched_reference = mean_matched_power_progress(
        n_stages, float(reference.mean())
    )
    natural_feature = reference - matched_reference
    metric_feature, metric_amplitude = zero_mean_family_feature(
        n_stages, metric, method
    )
    combined_feature = natural_feature + metric_amplitude * metric_feature

    strength = 1.0
    progress = base + strength * combined_feature
    allowed_recoveries = MAX_LOCAL_RECOVERIES[method]
    max_recovery_step = 0.045 if n_stages >= 8 else 0.060

    for _ in range(60):
        progress = base + strength * combined_feature
        progress[0] = 0.0
        progress[-1] = 1.0
        progress[1:-1] += (
            (target_mean - float(progress.mean())) * n_stages / (n_stages - 2)
        )
        steps = np.diff(progress)
        recovery_count = int(np.sum(steps < -1e-8))
        if (
            recovery_count <= allowed_recoveries
            and float(steps.min()) >= -max_recovery_step
            and progress.min() >= -1e-12
            and progress.max() <= 1.0 + 1e-12
        ):
            break
        strength *= 0.82
    else:
        raise AssertionError(f"Could not make a bounded natural profile for {method}")

    progress[0] = 0.0
    progress[-1] = 1.0
    if n_stages > 2:
        progress[1:-1] += (
            (target_mean - float(progress.mean())) * n_stages / (n_stages - 2)
        )
    if not np.isclose(progress.mean(), target_mean, atol=1e-12):
        raise AssertionError("Progress mean correction failed")
    if int(np.sum(np.diff(progress) < -1e-8)) > allowed_recoveries:
        raise AssertionError(f"Too many local recoveries survived for {method}")
    return progress


def constrained_curve(
    trajectory_mean: float,
    final_mean: float,
    n_stages: int,
    protocol: str,
    metric: str,
    method: str,
) -> np.ndarray:
    """Construct a plausible preview satisfying the applicable constraints.

    AV1M uses a strict three-point decline.  Every protocol preserves both the
    Table-II trajectory mean and final value while following a method-specific
    early-shock/plateau/recovery template.
    """
    if protocol == AV1M_PROTOCOL:
        if n_stages != 3:
            raise AssertionError("The AV1M protocol must contain exactly three stages")
        return av1m_declining_curve(
            trajectory_mean, final_mean, metric, method
        )
    if protocol == HIFI_PROTOCOL:
        if n_stages != 4:
            raise AssertionError("The HiFi protocol must contain exactly four stages")
        return hifi_curved_curve(trajectory_mean, final_mean, metric, method)

    start = desired_start(protocol, metric, method)
    scale = final_mean - start
    if np.isclose(scale, 0.0):
        raise ValueError(f"Start and final values coincide for {method}")
    normalized_mean = (trajectory_mean - start) / scale
    progress = method_family_progress(
        n_stages=n_stages,
        target_mean=normalized_mean,
        metric=metric,
        method=method,
    )
    curve = start + scale * progress

    curve[0] = start
    curve[-1] = final_mean
    if not np.isclose(curve.mean(), trajectory_mean):
        raise AssertionError("Curve mean does not match trajectory mean")
    if not np.isclose(curve[-1], final_mean):
        raise AssertionError("Curve endpoint does not match final mean")
    if not np.isclose(curve[0], start):
        raise AssertionError("Curve start does not match the protocol start")
    direction = np.sign(final_mean - start)
    signed_steps = direction * np.diff(curve)
    recovery_count = int(np.sum(signed_steps < -1e-8))
    if recovery_count > MAX_LOCAL_RECOVERIES[method]:
        raise AssertionError(
            f"Too many recoveries for {protocol}, {metric}, {method}"
        )
    if signed_steps.min() < -0.045 * abs(scale) - 1e-10:
        raise AssertionError(f"Recovery too large for {protocol}, {metric}, {method}")
    if np.any((curve < 0.45) | (curve > 1.0)):
        raise AssertionError(
            f"Implausible planning curve range for {protocol}, {metric}, {method}: "
            f"{curve.min():.4f} to {curve.max():.4f}"
        )
    return curve


def table2_heterogeneous_auc_gaps(
    table2: Mapping[str, Mapping[str, Sequence[Result]]]
) -> Dict[str, float]:
    """Return exact heterogeneous trajectory-to-final AUC gaps.

    These values are fully determined by the two heterogeneous AUC columns in
    Table II.  They are a retention summary, not the forgetting metric
    :math:`\\mathcal{F}`, which requires the complete stage-by-task matrix.
    """
    gaps: Dict[str, float] = {}
    for method in DISPLAY_METHODS:
        trajectory = table2["AUC"][method][4].mean
        final = table2["AUC"][method][5].mean
        gaps[method] = trajectory - final
    return gaps


def add_stream_regions(ax: plt.Axes) -> None:
    """Add subtle domain regions to a twelve-stage heterogeneous panel."""
    ax.axvspan(0.5, 5.5, color="#EAF3FA", zorder=0)
    ax.axvspan(5.5, 8.5, color="#F7F0E3", zorder=0)
    ax.axvspan(8.5, 12.5, color="#EAF5EA", zorder=0)
    ax.axvline(5.5, color="#999999", linestyle="--", linewidth=0.55, zorder=1)
    ax.axvline(8.5, color="#999999", linestyle="--", linewidth=0.55, zorder=1)


def style_axis(ax: plt.Axes, n_stages: int, ylabel: str | None = None) -> None:
    ax.set_xlim(0.7, n_stages + 0.3)
    if n_stages <= 4:
        ticks = np.arange(1, n_stages + 1)
    else:
        ticks = np.asarray([1, 3, 5, 7, 9, 11, 12])
    ax.set_xticks(ticks)
    ax.set_xlabel("Stage index")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(color="#D8D8D8", linewidth=0.45, alpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.0, width=0.55, pad=1.5)


def plot_method_curves(
    ax: plt.Axes,
    table2: Mapping[str, Mapping[str, Sequence[Result]]],
    metric: str,
    table_offset: int,
    n_stages: int,
    protocol: str,
    csv_rows: List[Dict[str, object]],
) -> None:
    x = np.arange(1, n_stages + 1)
    if n_stages == 12:
        add_stream_regions(ax)

    for method in DISPLAY_METHODS:
        trajectory = table2[metric][method][table_offset]
        final = table2[metric][method][table_offset + 1]
        values = constrained_curve(
            trajectory_mean=trajectory.mean,
            final_mean=final.mean,
            n_stages=n_stages,
            protocol=protocol,
            metric=metric,
            method=method,
        )
        figure_curve_mean = float(np.mean(values))
        if protocol == AV1M_PROTOCOL:
            table_alignment = "mean_and_final"
            note = (
                "strictly decreasing planning curve; start is solved from the "
                "selected curvature so curve mean and final both match Table II"
            )
        else:
            table_alignment = "mean_and_final"
            note = (
                "intermediate stages are constrained planning values with "
                "reference-like method-specific natural trends"
            )
        is_cape = method == "CAPE"
        is_strong = method in {"DER++", "CODA-Prompt", "PROOF", "MoE-Adapters"}

        ax.plot(
            x,
            values,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=1.75 if is_cape else (1.05 if is_strong else 0.78),
            markersize=3.0 if is_cape else 2.1,
            markeredgewidth=0.30,
            markeredgecolor="white" if is_cape else METHOD_COLORS[method],
            alpha=1.0 if is_cape else 0.86,
            zorder=5 if is_cape else (4 if is_strong else 3),
        )

        for stage, value in enumerate(values, start=1):
            csv_rows.append(
                {
                    "status": STATUS,
                    "record_type": "stage_metric",
                    "protocol": protocol,
                    "metric": metric,
                    "method": method,
                    "stage": stage,
                    "predicted_value": f"{value:.6f}",
                    "figure_curve_mean": f"{figure_curve_mean:.6f}",
                    "source_trajectory_mean": f"{trajectory.mean:.6f}",
                    "source_final_mean": f"{final.mean:.6f}",
                    "table_alignment": table_alignment,
                    "note": note,
                }
            )


def write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_figure(
    table2: Mapping[str, Mapping[str, Sequence[Result]]]
) -> List[Dict[str, object]]:
    fig = plt.figure(figsize=(7.16, 4.05))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        hspace=0.48,
        wspace=0.33,
    )

    axes = {
        "a": fig.add_subplot(grid[0, 0]),
        "b": fig.add_subplot(grid[0, 1]),
        "c": fig.add_subplot(grid[0, 2]),
        "legend": fig.add_subplot(grid[0, 3]),
        "d": fig.add_subplot(grid[1, 0]),
        "e": fig.add_subplot(grid[1, 1]),
        "f": fig.add_subplot(grid[1, 2]),
        "g": fig.add_subplot(grid[1, 3]),
    }

    csv_rows: List[Dict[str, object]] = []
    panels = [
        ("a", "AV1M - AUC", "AUC", 0, 3, "AV1M Pattern-Incremental"),
        ("b", "HiFi - AUC", "AUC", 2, 4, "HiFi Generator-Incremental"),
        ("c", "Heterogeneous - AUC", "AUC", 4, 12, "Heterogeneous Long-Stream"),
        ("d", "AV1M - AP", "AP", 0, 3, "AV1M Pattern-Incremental"),
        ("e", "HiFi - AP", "AP", 2, 4, "HiFi Generator-Incremental"),
        ("f", "Heterogeneous - AP", "AP", 4, 12, "Heterogeneous Long-Stream"),
    ]

    for key, title, metric, offset, n_stages, protocol in panels:
        ax = axes[key]
        plot_method_curves(
            ax=ax,
            table2=table2,
            metric=metric,
            table_offset=offset,
            n_stages=n_stages,
            protocol=protocol,
            csv_rows=csv_rows,
        )
        ax.set_title(f"({key}) {title}", loc="left", fontweight="bold", pad=3.0)
        ax.set_ylim(0.50, 1.005)
        ylabel = "Avg. AUC over seen stages" if key == "a" else None
        if key == "d":
            ylabel = "Avg. AP over seen stages"
        style_axis(ax, n_stages, ylabel)

    auc_gaps = table2_heterogeneous_auc_gaps(table2)
    ax = axes["g"]
    method_indices = np.arange(1, len(DISPLAY_METHODS) + 1)
    gap_values = np.asarray([auc_gaps[method] for method in DISPLAY_METHODS])
    bars = ax.bar(
        method_indices,
        gap_values,
        width=0.72,
        color=[METHOD_COLORS[method] for method in DISPLAY_METHODS],
        edgecolor=["#004C78" if method == "CAPE" else "#FFFFFF" for method in DISPLAY_METHODS],
        linewidth=[1.0 if method == "CAPE" else 0.35 for method in DISPLAY_METHODS],
        alpha=0.90,
        zorder=3,
    )
    bars[-1].set_zorder(5)
    for method, value in zip(DISPLAY_METHODS, gap_values):
        trajectory = table2["AUC"][method][4].mean
        final = table2["AUC"][method][5].mean
        csv_rows.append(
            {
                "status": STATUS,
                "record_type": "table2_auc_gap",
                "protocol": "Heterogeneous Long-Stream",
                "metric": "Trajectory-final AUC gap",
                "method": method,
                "stage": 12,
                "predicted_value": f"{value:.6f}",
                "figure_curve_mean": "",
                "source_trajectory_mean": f"{trajectory:.6f}",
                "source_final_mean": f"{final:.6f}",
                "table_alignment": "exact_table2_derived",
                "note": (
                    "exact Table-II derived value: heterogeneous trajectory "
                    "mean minus heterogeneous final mean; not forgetting F"
                ),
            }
        )
    ax.set_title(
        "(g) Heterogeneous - AUC gap",
        loc="left",
        fontweight="bold",
        pad=3.0,
    )
    ax.set_xlim(0.35, len(DISPLAY_METHODS) + 0.65)
    ax.set_ylim(0.0, max(gap_values) * 1.14)
    ax.set_xticks(method_indices)
    ax.set_xticklabels([str(index) for index in method_indices])
    ax.set_xlabel("Method (legend order)")
    ax.set_ylabel("Traj.-final AUC")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.45, alpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.0, width=0.55, pad=1.5)

    legend_ax = axes["legend"]
    legend_ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=1.75 if method == "CAPE" else 0.95,
            markersize=3.0 if method == "CAPE" else 2.4,
            label=method,
        )
        for method in DISPLAY_METHODS
    ]
    legend_ax.text(
        0.02,
        0.99,
        "Methods",
        ha="left",
        va="top",
        fontsize=6.6,
        fontweight="bold",
        transform=legend_ax.transAxes,
    )
    legend_ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.00, 0.92),
        frameon=False,
        ncol=1,
        handlelength=2.1,
        labelspacing=0.28,
        borderaxespad=0.0,
        columnspacing=0.4,
    )
    legend_ax.text(
        0.02,
        0.03,
        "Heterogeneous regions:\n"
        "1-5 FakeAVCeleb\n"
        "6-8 AV-Deepfake1M\n"
        "9-12 HiFi-AVDF",
        ha="left",
        va="bottom",
        fontsize=5.1,
        color="#444444",
        linespacing=1.18,
        transform=legend_ax.transAxes,
    )

    fig.text(
        0.50,
        0.50,
        WATERMARK,
        ha="center",
        va="center",
        fontsize=17,
        color="#A33A2B",
        alpha=0.085,
        rotation=18,
        fontweight="bold",
    )
    fig.text(
        0.50,
        0.007,
        "Planning preview: all six stage panels match Table-II trajectory means and finals; "
        "intermediate stage values remain method-specific planning curves.",
        ha="center",
        va="bottom",
        fontsize=5.3,
        color="#555555",
    )
    fig.subplots_adjust(left=0.067, right=0.995, bottom=0.105, top=0.965)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUTPUT_DIR / f"{OUTPUT_STEM}.{extension}")
    plt.close(fig)
    return csv_rows


def validate_rows(rows: Sequence[Mapping[str, object]]) -> None:
    stage_rows = [row for row in rows if row["record_type"] == "stage_metric"]
    grouped: Dict[tuple[str, str, str], List[Mapping[str, object]]] = {}
    for row in stage_rows:
        key = (str(row["protocol"]), str(row["metric"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    expected_counts = {
        "AV1M Pattern-Incremental": 3,
        "HiFi Generator-Incremental": 4,
        "Heterogeneous Long-Stream": 12,
    }
    starts_by_panel: Dict[tuple[str, str], List[float]] = {}
    for (protocol, metric, method), group in grouped.items():
        group = sorted(group, key=lambda item: int(item["stage"]))
        values = np.asarray([float(item["predicted_value"]) for item in group])
        trajectory = float(group[0]["source_trajectory_mean"])
        final = float(group[0]["source_final_mean"])
        table_alignment = str(group[0]["table_alignment"])
        starts_by_panel.setdefault((protocol, metric), []).append(float(values[0]))
        if len(values) != expected_counts[protocol]:
            raise AssertionError(f"Unexpected point count: {protocol}, {metric}, {method}")
        if table_alignment != "mean_and_final":
            raise AssertionError(
                f"Unexpected alignment tag: {protocol}, {metric}, {method}"
            )
        if not np.isclose(values.mean(), trajectory, atol=1e-6):
            raise AssertionError(f"Trajectory mismatch: {protocol}, {metric}, {method}")
        if protocol == AV1M_PROTOCOL:
            if not np.all(np.diff(values) < -1e-6):
                raise AssertionError(f"AV1M must strictly decline: {metric}, {method}")
        if not np.isclose(values[-1], final, atol=1e-6):
            raise AssertionError(f"Final mismatch: {protocol}, {metric}, {method}")
        direction = np.sign(final - values[0])
        signed_steps = direction * np.diff(values)
        recovery_count = int(np.sum(signed_steps < -2e-6))
        if recovery_count > MAX_LOCAL_RECOVERIES[method]:
            raise AssertionError(
                f"Too many local recoveries: {protocol}, {metric}, {method}"
            )
        if signed_steps.min() < -0.045 * abs(final - values[0]) - 2e-6:
            raise AssertionError(
                f"Local recovery too large: {protocol}, {metric}, {method}"
            )

    for (protocol, metric), starts in starts_by_panel.items():
        spread = max(starts) - min(starts)
        if protocol == AV1M_PROTOCOL:
            minimum_spread, maximum_spread = 0.035, 0.075
        elif protocol == HIFI_PROTOCOL:
            minimum_spread, maximum_spread = 0.010, 0.025
        else:
            minimum_spread, maximum_spread = 0.015, 0.025
        if not minimum_spread - 1e-9 <= spread <= maximum_spread + 1e-9:
            raise AssertionError(
                f"Stage-one spread outside {minimum_spread:.3f}--"
                f"{maximum_spread:.3f}: "
                f"{protocol}, {metric}, {spread:.6f}"
            )

    gap_rows = [row for row in rows if row["record_type"] == "table2_auc_gap"]
    if len(gap_rows) != len(DISPLAY_METHODS):
        raise AssertionError("Expected one exact Table-II AUC gap per method")
    for row in gap_rows:
        gap = float(row["predicted_value"])
        trajectory = float(row["source_trajectory_mean"])
        final = float(row["source_final_mean"])
        if not np.isclose(gap, trajectory - final, atol=1e-6):
            raise AssertionError(f"Table-II AUC gap mismatch: {row['method']}")


def build_diagnostics(
    rows: Sequence[Mapping[str, object]]
) -> Dict[str, object]:
    """Summarize exact constraints, smoothness, and start separation."""
    stage_rows = [row for row in rows if row["record_type"] == "stage_metric"]
    grouped: Dict[tuple[str, str, str], List[Mapping[str, object]]] = {}
    for row in stage_rows:
        key = (str(row["protocol"]), str(row["metric"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    curve_records: List[Dict[str, object]] = []
    panel_starts: Dict[tuple[str, str], List[float]] = {}
    max_mean_aligned_error = 0.0
    max_final_error = 0.0
    for (protocol, metric, method), group in grouped.items():
        group = sorted(group, key=lambda item: int(item["stage"]))
        values = np.asarray([float(item["predicted_value"]) for item in group])
        source_mean = float(group[0]["source_trajectory_mean"])
        source_final = float(group[0]["source_final_mean"])
        table_alignment = str(group[0]["table_alignment"])
        mean_error = abs(float(values.mean()) - source_mean)
        final_error = abs(float(values[-1]) - source_final)
        if table_alignment != "mean_and_final":
            raise AssertionError(
                f"Unexpected stage-curve alignment tag in diagnostics: "
                f"{protocol}, {metric}, {method}"
            )
        max_mean_aligned_error = max(max_mean_aligned_error, mean_error)
        max_final_error = max(max_final_error, final_error)
        second_difference = np.diff(values, n=2)
        direction = np.sign(source_final - values[0])
        signed_steps = direction * np.diff(values)
        panel_starts.setdefault((protocol, metric), []).append(float(values[0]))
        curve_records.append(
            {
                "protocol": protocol,
                "metric": metric,
                "method": method,
                "start": float(values[0]),
                "final": float(values[-1]),
                "trajectory_mean": float(values.mean()),
                "source_table_trajectory_mean": source_mean,
                "table_alignment": table_alignment,
                "total_absolute_second_difference": float(
                    np.sum(np.abs(second_difference))
                ),
                "max_absolute_second_difference": float(
                    np.max(np.abs(second_difference))
                    if len(second_difference)
                    else 0.0
                ),
                "direction_reversals": int(np.sum(signed_steps < -2e-6)),
            }
        )

    start_spreads = [
        {
            "protocol": protocol,
            "metric": metric,
            "minimum_start": min(values),
            "maximum_start": max(values),
            "spread": max(values) - min(values),
        }
        for (protocol, metric), values in sorted(panel_starts.items())
    ]
    heterogeneous_curvature = {
        metric: sorted(
            [
                {
                    "method": record["method"],
                    "total_absolute_second_difference": record[
                        "total_absolute_second_difference"
                    ],
                }
                for record in curve_records
                if record["protocol"] == "Heterogeneous Long-Stream"
                and record["metric"] == metric
            ],
            key=lambda item: float(item["total_absolute_second_difference"]),
        )
        for metric in ("AUC", "AP")
    }
    return {
        "maximum_rounded_mean_error_for_mean_aligned_curves": max_mean_aligned_error,
        "maximum_rounded_final_error": max_final_error,
        "all_stage_metric_direction_reversals": int(
            sum(int(record["direction_reversals"]) for record in curve_records)
        ),
        "stage_one_spreads": start_spreads,
        "heterogeneous_curvature_ranking_low_to_high": heterogeneous_curvature,
        "curve_records": curve_records,
    }


def main() -> None:
    configure_style()
    tex = TEX_PATH.read_text(encoding="utf-8")
    table2 = parse_table2(tex)
    rows = create_figure(table2)
    validate_rows(rows)
    write_csv(OUTPUT_DIR / f"{OUTPUT_STEM}.csv", list(rows[0].keys()), rows)
    diagnostics = build_diagnostics(rows)

    metadata = {
        "status": STATUS,
        "source_tex": str(TEX_PATH),
        "script": str(Path(__file__).resolve()),
        "figure_stem": OUTPUT_STEM,
        "layout": "2x4 grid with seven data panels and one legend panel",
        "preview_version": "final_merged_table_all_panels_mean_and_final_aligned",
        "methods": DISPLAY_METHODS,
        "constraints": {
            "stage_metric_profiles": (
                "AV1M: strict decline with start analytically solved for exact "
                "Table-II trajectory mean and final; HiFi: method-specific "
                "early-shock/plateau curvature with exact Table-II trajectory "
                "mean and final; heterogeneous: fixed planning start, exact "
                "Table-II trajectory mean and final, limited method-specific "
                "local recovery"
            ),
            "panel_g": (
                "exact heterogeneous Table-II trajectory AUC minus final AUC "
                "for every method; this is not forgetting F"
            ),
        },
        "diagnostics": diagnostics,
        "warning": (
            "Planning preview only. Although every displayed aggregate now "
            "matches Table II, replace predicted stage profiles and "
            "replace panel (g) with forgetting only after baseline stage-by-task "
            "matrices are available."
        ),
    }
    (OUTPUT_DIR / f"{OUTPUT_STEM}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Generated: {OUTPUT_DIR / (OUTPUT_STEM + '.pdf')}")
    print(f"Generated: {OUTPUT_DIR / (OUTPUT_STEM + '.png')}")
    print(f"Generated: {OUTPUT_DIR / (OUTPUT_STEM + '.svg')}")
    print(f"Generated: {OUTPUT_DIR / (OUTPUT_STEM + '.csv')}")
    print(
        "Maximum rounded mean error (mean-aligned curves only): "
        f"{diagnostics['maximum_rounded_mean_error_for_mean_aligned_curves']:.3e}"
    )
    print(
        "Stage-metric local recoveries: "
        f"{diagnostics['all_stage_metric_direction_reversals']}"
    )
    print("Status: planning preview, not verified experimental evidence")


if __name__ == "__main__":
    main()
