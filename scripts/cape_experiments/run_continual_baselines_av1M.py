import argparse
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cape_baselines import ContinualBaselineTrainer

AV1M_TASKS = "1,2,3"
DEFAULT_METHODS = "seq_ft,er,lwf,der,derpp,ewc"
DEFAULT_BACKBONE_CKPT = ""


def default_output_root(epochs):
    return f"results/continual_baselines_av1m_e{epochs}"


def main():
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description="Run AV-Deepfake1M continual-learning baselines.")
    parser.add_argument("--metadata", default="data/cape_metadata.csv")
    parser.add_argument("--tasks", default=AV1M_TASKS, help="Default AV1M pattern tasks: 1=visual, 2=audio, 3=audio-visual.")
    parser.add_argument("--methods", default=DEFAULT_METHODS, help="Comma-separated methods: seq_ft,er,lwf,der,derpp,ewc,joint,single_task.")
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
    parser.add_argument("--replay-capacity", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--backbone-ckpt", default=DEFAULT_BACKBONE_CKPT)
    parser.add_argument("--backbone-mode", choices=["precomputed_avhubert", "legacy_dimodif"], default="precomputed_avhubert")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if args.output_root is None:
        args.output_root = default_output_root(args.epochs)

    task_sequence = [task.strip() for task in args.tasks.split(",") if task.strip()]
    methods = [method.strip().lower() for method in args.methods.split(",") if method.strip()]
    print(f"[Baseline-AV1M] project_root={PROJECT_ROOT}")
    print(f"[Baseline-AV1M] metadata={args.metadata}")
    print(f"[Baseline-AV1M] tasks={task_sequence}")
    print(f"[Baseline-AV1M] methods={methods}")
    print(f"[Baseline-AV1M] output_root={args.output_root}")
    print(f"[Baseline-AV1M] eval_split={args.eval_split}")
    print(f"[Baseline-AV1M] device={args.device}; epochs={args.epochs}; batch_size={args.batch_size}")
    print(
        f"[Baseline-AV1M] num_workers={args.num_workers}; "
        f"pin_memory={not args.no_pin_memory}; amp={args.amp}"
    )
    print(f"[Baseline-AV1M] backbone_ckpt={args.backbone_ckpt or 'none'}")
    if args.device == "cpu":
        print("[Baseline-AV1M] WARNING: CUDA is not available in this Python environment; full AV1M baselines will be slow.")

    for method in methods:
        output_dir = os.path.join(args.output_root, method)
        print(f"\n[Baseline-AV1M] Running {method} -> {output_dir}")
        trainer = ContinualBaselineTrainer(
            metadata_csv=args.metadata,
            task_sequence=task_sequence,
            method=method,
            max_length=args.max_length,
            d_model=args.d_model,
            nhead=args.nhead,
            d_hid=args.d_hid,
            nlayers=args.nlayers,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            replay_capacity=args.replay_capacity,
            seed=args.seed,
            device=args.device,
            output_dir=output_dir,
            eval_split=args.eval_split,
            num_workers=args.num_workers,
            pin_memory=not args.no_pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=not args.no_persistent_workers,
            amp=args.amp,
            freeze_backbone=not args.train_backbone,
            backbone_ckpt=args.backbone_ckpt or None,
            backbone_mode=args.backbone_mode,
            show_progress=not args.no_progress,
        )
        if trainer.model.backbone_load_report:
            print(f"[Baseline-AV1M] Backbone load: {trainer.model.backbone_load_report}")
        trainer.run()
        print(f"[Baseline-AV1M] Finished {method}")


if __name__ == "__main__":
    main()
