import argparse
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cape_continual import CAPEContinualTrainer
from src.cape_metrics import safe_ap, safe_auc


DEFAULT_KNOWN_TASKS = "generator:kling2.5,generator:veo3.1,generator:wan2.5"
DEFAULT_UNKNOWN_TASKS = "generator:seedance1.0"
DEFAULT_BACKBONE_CKPT = ""


@torch.no_grad()
def collect_unknown_scores(trainer, split, task_ids, unknown_label):
    labels, scores = [], []
    for task_id in task_ids:
        loader = trainer.loader(split, task_id=task_id)
        for batch in loader:
            batch = trainer._move(batch)
            outputs = trainer.model(
                batch["video_features"],
                batch["audio_features"],
                valid_lengths=batch["valid_length"],
                backbone_features=batch.get("backbone_features"),
            )
            labels.extend([unknown_label] * len(outputs["unknown_score"]))
            scores.extend(outputs["unknown_score"].detach().cpu().tolist())
    return labels, scores


def main():
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description="Held-out generator unknown-discovery evaluation for CAPE.")
    parser.add_argument("--metadata", default="data/cape_metadata.csv")
    parser.add_argument("--known-tasks", default=DEFAULT_KNOWN_TASKS)
    parser.add_argument("--unknown-tasks", default=DEFAULT_UNKNOWN_TASKS)
    parser.add_argument("--max-length", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--d-hid", type=int, default=3072)
    parser.add_argument("--nlayers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--expert-bottleneck", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--prototype-momentum", type=float, default=0.95)
    parser.add_argument("--bandwidth-init", type=float, default=0.35)
    parser.add_argument("--bandwidth-momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="results/cape_unknown")
    parser.add_argument("--backbone-ckpt", default=DEFAULT_BACKBONE_CKPT)
    parser.add_argument("--backbone-mode", choices=["precomputed_avhubert", "legacy_dimodif"], default="precomputed_avhubert")
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--no-confidence-density", action="store_true")
    parser.add_argument("--unknown-alpha", type=float, default=0.05)
    parser.add_argument("--unknown-normalization-fraction", type=float, default=0.5)
    parser.add_argument("--unknown-min-cluster-size", type=int, default=8)
    parser.add_argument("--unknown-cluster-eps", type=float, default=0.8)
    parser.add_argument("--unknown-queue-capacity", type=int, default=4096)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    known_tasks = [task.strip() for task in args.known_tasks.split(",") if task.strip()]
    unknown_tasks = [task.strip() for task in args.unknown_tasks.split(",") if task.strip()]
    trainer = CAPEContinualTrainer(
        metadata_csv=args.metadata,
        task_sequence=known_tasks,
        max_length=args.max_length,
        d_model=args.d_model,
        nhead=args.nhead,
        d_hid=args.d_hid,
        nlayers=args.nlayers,
        batch_size=args.batch_size,
        expert_bottleneck=args.expert_bottleneck,
        top_k=args.top_k,
        prototype_momentum=args.prototype_momentum,
        bandwidth_init=args.bandwidth_init,
        bandwidth_momentum=args.bandwidth_momentum,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
        freeze_backbone=not args.train_backbone,
        backbone_ckpt=args.backbone_ckpt or None,
        backbone_mode=args.backbone_mode,
        use_confidence_density=not args.no_confidence_density,
        unknown_alpha=args.unknown_alpha,
        unknown_normalization_fraction=args.unknown_normalization_fraction,
        unknown_min_cluster_size=args.unknown_min_cluster_size,
        unknown_cluster_eps=args.unknown_cluster_eps,
        unknown_queue_capacity=args.unknown_queue_capacity,
    )
    trainer.run()
    trainer.fit_unknown_calibration()

    known_labels, known_scores = collect_unknown_scores(trainer, "test", known_tasks, unknown_label=0)
    unknown_labels, unknown_scores = collect_unknown_scores(trainer, "test", unknown_tasks, unknown_label=1)
    y_true = known_labels + unknown_labels
    y_score = known_scores + unknown_scores
    queue_summary = trainer.collect_unknown_candidates("test", unknown_tasks, alpha=args.unknown_alpha)
    result = {
        "known_tasks": known_tasks,
        "unknown_tasks": unknown_tasks,
        "unknown_auc": safe_auc(y_true, y_score),
        "unknown_ap": safe_ap(y_true, y_score),
        "known_score_mean": float(torch.tensor(known_scores).mean()) if known_scores else float("nan"),
        "unknown_score_mean": float(torch.tensor(unknown_scores).mean()) if unknown_scores else float("nan"),
        "unknown_alpha": args.unknown_alpha,
        "unknown_queue": queue_summary,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "unknown_summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
