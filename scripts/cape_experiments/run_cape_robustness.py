import argparse
import json
import os
import sys

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cape_continual import CAPEContinualTrainer


DEFAULT_TASKS = "generator:kling2.5,generator:veo3.1,generator:wan2.5,generator:seedance1.0"
DEFAULT_BACKBONE_CKPT = ""


def main():
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description="Train CAPE and evaluate audio-visual temporal-shift robustness.")
    parser.add_argument("--metadata", default="data/cape_metadata.csv")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--shift-steps", default="-8,-4,-2,0,2,4,8")
    parser.add_argument("--seeds", default="0,1,2,3,4")
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
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", default="results/cape_robustness")
    parser.add_argument("--backbone-ckpt", default=DEFAULT_BACKBONE_CKPT)
    parser.add_argument("--backbone-mode", choices=["precomputed_avhubert", "legacy_dimodif"], default="precomputed_avhubert")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    shifts = [int(value.strip()) for value in args.shift_steps.split(",") if value.strip()]
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    for seed in seeds:
        output_dir = os.path.join(args.output_root, f"seed_{seed}")
        trainer = CAPEContinualTrainer(
            metadata_csv=args.metadata,
            task_sequence=tasks,
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
            seed=seed,
            device=args.device,
            output_dir=output_dir,
            eval_split=args.eval_split,
            show_progress=not args.no_progress,
            backbone_ckpt=args.backbone_ckpt or None,
            backbone_mode=args.backbone_mode,
        )
        trainer.run()
        results = {}
        for shift in shifts:
            results[str(shift)] = {
                task: trainer.evaluate_task(
                    args.eval_split,
                    task_id=task,
                    audio_shift_steps=shift,
                )
                for task in tasks
            }
        with open(os.path.join(output_dir, "temporal_shift_robustness.json"), "w") as f:
            json.dump({"seed": seed, "shift_steps": shifts, "results": results}, f, indent=2)
        print(f"[Robustness] seed={seed} -> {output_dir}")


if __name__ == "__main__":
    main()
