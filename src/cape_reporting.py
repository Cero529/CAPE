import json
import math
import os


def load_history(path):
    with open(path, "r") as f:
        return json.load(f)


def final_seen(history):
    if not history:
        return {}
    if len(history) == 1 and history[0].get("current_task") == "joint":
        return history[0].get("seen", {})
    return history[-1].get("seen", {})


def _metric(metrics, key):
    value = metrics.get(key, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def summarize_history(history):
    seen = final_seen(history)
    aucs = [_metric(metrics, "auc") for metrics in seen.values()]
    aps = [_metric(metrics, "ap") for metrics in seen.values()]
    avg_auc = _nanmean(aucs)
    avg_ap = _nanmean(aps)
    worst_auc = _nanmin(aucs)
    last_task_auc = aucs[-1] if aucs else float("nan")
    forgetting = _forgetting(history)
    trajectory_auc = _trajectory_average(history, "auc")
    trajectory_ap = _trajectory_average(history, "ap")
    return {
        "final_auc": avg_auc,
        "trajectory_auc": trajectory_auc,
        "final_ap": avg_ap,
        "trajectory_ap": trajectory_ap,
        "forgetting_auc": forgetting,
        "worst_auc": worst_auc,
        "last_task_auc": last_task_auc,
    }


def _trajectory_average(history, key):
    stage_values = []
    for row in history:
        seen = row.get("seen", {})
        if not seen:
            continue
        stage_values.append(_nanmean([_metric(metrics, key) for metrics in seen.values()]))
    return _nanmean(stage_values)


def _forgetting(history):
    final = final_seen(history)
    if len(final) <= 1:
        return float("nan")
    drops = []
    # The manuscript definition averages forgetting over the B-1 tasks that
    # existed before the final stage.  Including the final task would append a
    # guaranteed zero and incorrectly divide the result by B.
    for task_id, final_metrics in list(final.items())[:-1]:
        best = float("-inf")
        for row in history:
            if task_id in row.get("seen", {}):
                best = max(best, _metric(row["seen"][task_id], "auc"))
        if best > float("-inf"):
            drops.append(max(0.0, best - _metric(final_metrics, "auc")))
    return _nanmean(drops)


def _nanmean(values):
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else float("nan")


def _nanmin(values):
    clean = [v for v in values if not math.isnan(v)]
    return min(clean) if clean else float("nan")


def fmt(value):
    if value is None:
        return "TBD"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "TBD"
    return f"{value:.4f}"


def collect_result_dirs(root_dir):
    result_dirs = []
    for current, _, files in os.walk(root_dir):
        if "history.json" in files:
            result_dirs.append(current)
    return sorted(result_dirs)


def build_rows(result_dirs):
    rows = []
    for result_dir in result_dirs:
        history = load_history(os.path.join(result_dir, "history.json"))
        summary = summarize_history(history)
        rows.append({"method": os.path.basename(result_dir), "path": result_dir, **summary})
    return rows


def write_markdown_table(rows, output_path, title="Continual Learning Results"):
    lines = [
        f"# {title}",
        "",
        "| Method | Final AUC | Traj. AUC | Final AP | Traj. AP | Forgetting | Worst AUC | Last-task AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {final_auc} | {trajectory_auc} | {final_ap} | {trajectory_ap} | "
            "{forgetting_auc} | {worst_auc} | {last_task_auc} |".format(
                method=row["method"],
                final_auc=fmt(row["final_auc"]),
                trajectory_auc=fmt(row["trajectory_auc"]),
                final_ap=fmt(row["final_ap"]),
                trajectory_ap=fmt(row["trajectory_ap"]),
                forgetting_auc=fmt(row["forgetting_auc"]),
                worst_auc=fmt(row["worst_auc"]),
                last_task_auc=fmt(row["last_task_auc"]),
            )
        )
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_latex_rows(rows, output_path):
    lines = []
    for row in rows:
        lines.append(
            "{method} & {final_auc} & {trajectory_auc} & {final_ap} & {trajectory_ap} & "
            "{forgetting_auc} & {worst_auc} & {last_task_auc} \\\\".format(
                method=row["method"].replace("_", r"\_"),
                final_auc=fmt(row["final_auc"]),
                trajectory_auc=fmt(row["trajectory_auc"]),
                final_ap=fmt(row["final_ap"]),
                trajectory_ap=fmt(row["trajectory_ap"]),
                forgetting_auc=fmt(row["forgetting_auc"]),
                worst_auc=fmt(row["worst_auc"]),
                last_task_auc=fmt(row["last_task_auc"]),
            )
        )
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
