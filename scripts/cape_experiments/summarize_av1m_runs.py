import argparse
import json
import math
from pathlib import Path


TASKS = ["1", "2", "3"]


def load_history(path):
    with open(path, "r") as f:
        return json.load(f)


def mean(values):
    values = [value for value in values if value is not None and not math.isnan(value)]
    return sum(values) / len(values) if values else float("nan")


def summarize_history(history):
    final_seen = history[-1].get("seen", {}) if history else {}
    row = {}
    for metric in ("auc", "ap"):
        row[f"final_{metric}"] = mean([final_seen.get(task, {}).get(metric, float("nan")) for task in TASKS])
        trajectory = []
        for index, item in enumerate(history, start=1):
            seen = TASKS[:index]
            trajectory.append(mean([item.get("seen", {}).get(task, {}).get(metric, float("nan")) for task in seen]))
        row[f"trajectory_{metric}"] = mean(trajectory)
        forgetting = []
        for task in TASKS:
            values = [item.get("seen", {}).get(task, {}).get(metric) for item in history if task in item.get("seen", {})]
            if values:
                forgetting.append(max(values) - values[-1])
        row[f"forgetting_{metric}"] = mean(forgetting)
    for task in TASKS:
        row[f"task{task}_auc"] = final_seen.get(task, {}).get("auc", float("nan"))
        row[f"task{task}_ap"] = final_seen.get(task, {}).get("ap", float("nan"))
    return row


def find_histories(result_roots):
    paths = []
    for root in result_roots:
        root_path = Path(root)
        if root_path.is_file() and root_path.name == "history.json":
            paths.append(root_path)
        elif root_path.exists():
            paths.extend(root_path.rglob("history.json"))
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser(description="Summarize and rank AV1M CAPE runs.")
    parser.add_argument("result_roots", nargs="*", default=["results/cape_av1m_sweep", "results/cape_pattern_av1m_e1"])
    parser.add_argument("--output", default="results/cape_av1m_sweep_summary.md")
    args = parser.parse_args()

    rows = []
    for history_path in find_histories(args.result_roots):
        try:
            history = load_history(history_path)
            metrics = summarize_history(history)
        except Exception as exc:
            print(f"[Skip] {history_path}: {exc}")
            continue
        run_dir = history_path.parent
        rows.append({"run": str(run_dir), **metrics})

    rows.sort(key=lambda row: (row["final_auc"], row["final_ap"], -row["forgetting_auc"]), reverse=True)

    header = [
        "rank",
        "run",
        "final_auc",
        "final_ap",
        "trajectory_auc",
        "trajectory_ap",
        "forgetting_auc",
        "task1_auc",
        "task2_auc",
        "task3_auc",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for rank, row in enumerate(rows, start=1):
        values = [
            str(rank),
            row["run"],
            f"{row['final_auc']:.4f}",
            f"{row['final_ap']:.4f}",
            f"{row['trajectory_auc']:.4f}",
            f"{row['trajectory_ap']:.4f}",
            f"{row['forgetting_auc']:.4f}",
            f"{row['task1_auc']:.4f}",
            f"{row['task2_auc']:.4f}",
            f"{row['task3_auc']:.4f}",
        ]
        lines.append("| " + " | ".join(values) + " |")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[: min(len(lines), 12)]))
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
