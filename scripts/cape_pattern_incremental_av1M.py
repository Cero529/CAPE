import argparse
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cape_continual import CAPEContinualTrainer

AV1M_TASKS = "1,2,3"
DEFAULT_BACKBONE_CKPT = ""


def default_output_dir(epochs):
    return f"results/cape_pattern_av1m_e{epochs}_evalbest"


def main():
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description="Run CAPE AV-Deepfake1M pattern-incremental training.")
    parser.add_argument("--metadata", default="data/cape_metadata.csv")
    parser.add_argument("--tasks", default=AV1M_TASKS, help="Default AV1M pattern tasks: 1=visual, 2=audio, 3=audio-visual.")
    parser.add_argument("--eval-split", default="test", help="Paper protocol reports the official test split.")
    parser.add_argument("--max-length", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--d-hid", type=int, default=3072)
    parser.add_argument("--nlayers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay-capacity", type=int, default=1024)
    parser.add_argument("--expert-bottleneck", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--prototype-momentum", type=float, default=0.95)
    parser.add_argument("--bandwidth-init", type=float, default=0.35)
    parser.add_argument("--bandwidth-momentum", type=float, default=0.9)
    parser.add_argument("--min-bandwidth", type=float, default=0.05)
    parser.add_argument("--max-bandwidth", type=float, default=1.25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument("--max-train-batches", type=int, default=0, help="Debug speed limit; 0 means full epoch.")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="Debug speed limit; 0 means full eval.")
    parser.add_argument(
        "--no-eval-every-epoch",
        action="store_true",
        help="Disable the default per-epoch validation pass.",
    )
    parser.add_argument(
        "--no-save-best",
        action="store_true",
        help="Disable saving best_final.json and best_final.pt.",
    )
    parser.add_argument("--best-metric", default="auc", choices=["auc", "ap"])
    parser.add_argument("--backbone-ckpt", default=DEFAULT_BACKBONE_CKPT)
    parser.add_argument(
        "--backbone-mode",
        choices=["precomputed_avhubert", "legacy_dimodif"],
        default="precomputed_avhubert",
        help="Paper profile requires NPZ backbone_features from frozen AV-HuBERT.",
    )
    parser.add_argument("--train-backbone", action="store_true", help="Legacy mode only: do not freeze the DiMoDif backbone.")
    parser.add_argument("--no-confidence-density", action="store_true")
    parser.add_argument("--unknown-alpha", type=float, default=0.05)
    parser.add_argument("--unknown-normalization-fraction", type=float, default=0.5)
    parser.add_argument("--calibration-capacity", type=int, default=1024)
    parser.add_argument("--lambda-distill", type=float, default=1.0)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = default_output_dir(args.epochs)

    print(f"[CAPE-AV1M] project_root={PROJECT_ROOT}")
    print(f"[CAPE-AV1M] metadata={args.metadata}")
    print(f"[CAPE-AV1M] tasks={args.tasks}")
    print(f"[CAPE-AV1M] output_dir={args.output_dir}")
    print(f"[CAPE-AV1M] eval_split={args.eval_split}")
    print(f"[CAPE-AV1M] device={args.device}; batch_size={args.batch_size}; epochs={args.epochs}")
    print(f"[CAPE-AV1M] lr={args.lr}; replay_capacity={args.replay_capacity}")
    print(
        f"[CAPE-AV1M] num_workers={args.num_workers}; "
        f"pin_memory={not args.no_pin_memory}; amp={args.amp}"
    )
    if args.max_train_batches or args.max_eval_batches:
        print(f"[CAPE-AV1M] debug limits: train={args.max_train_batches}; eval={args.max_eval_batches}")
    eval_every_epoch = not args.no_eval_every_epoch
    save_best = not args.no_save_best

    if eval_every_epoch:
        print(
            f"[CAPE-AV1M] eval_every_epoch=True; "
            f"save_best={save_best}; best_metric={args.best_metric}"
        )
    print(f"[CAPE-AV1M] backbone_ckpt={args.backbone_ckpt or 'none'}")
    if args.device == "cpu":
        print("[CAPE-AV1M] WARNING: CUDA is not available in this Python environment; full AV1M training will be slow.")

    trainer = CAPEContinualTrainer(
        metadata_csv=args.metadata,
        task_sequence=[task.strip() for task in args.tasks.split(",") if task.strip()],
        max_length=args.max_length,
        d_model=args.d_model,
        nhead=args.nhead,
        d_hid=args.d_hid,
        nlayers=args.nlayers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        replay_capacity=args.replay_capacity,
        expert_bottleneck=args.expert_bottleneck,
        top_k=args.top_k,
        dropout=args.dropout,
        prototype_momentum=args.prototype_momentum,
        bandwidth_init=args.bandwidth_init,
        bandwidth_momentum=args.bandwidth_momentum,
        min_bandwidth=args.min_bandwidth,
        max_bandwidth=args.max_bandwidth,
        device=args.device,
        output_dir=args.output_dir,
        eval_split=args.eval_split,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=not args.no_persistent_workers,
        amp=args.amp,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        eval_every_epoch=eval_every_epoch,
        save_best=save_best,
        best_metric=args.best_metric,
        show_progress=not args.no_progress,
        freeze_backbone=not args.train_backbone,
        backbone_ckpt=args.backbone_ckpt or None,
        use_confidence_density=not args.no_confidence_density,
        unknown_alpha=args.unknown_alpha,
        unknown_normalization_fraction=args.unknown_normalization_fraction,
        calibration_capacity=args.calibration_capacity,
        lambda_distill=args.lambda_distill,
        backbone_mode=args.backbone_mode,
    )
    if trainer.model.backbone_load_report:
        print(f"[CAPE-AV1M] Backbone load: {trainer.model.backbone_load_report}")

    history = trainer.run()
    final_seen = history[-1]["seen"] if history else {}
    if final_seen:
        avg_auc = sum(metrics["auc"] for metrics in final_seen.values()) / len(final_seen)
        avg_ap = sum(metrics["ap"] for metrics in final_seen.values()) / len(final_seen)
        print("\n[CAPE-AV1M] Final results")
        for task_id, metrics in final_seen.items():
            print(
                f"  {task_id}: "
                f"AUC={metrics['auc']:.4f}, "
                f"AP={metrics['ap']:.4f}, "
                f"unknown={metrics['unknown_score_mean']:.4f}"
            )
        print(f"  Final Avg: AUC={avg_auc:.4f}, AP={avg_ap:.4f}")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"final_seen": final_seen}, f, indent=2)
    print(f"[CAPE-AV1M] History saved to {os.path.join(args.output_dir, 'history.json')}")
    print(f"[CAPE-AV1M] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
