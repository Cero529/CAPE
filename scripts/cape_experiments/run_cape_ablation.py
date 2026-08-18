import argparse
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cape_continual import CAPEContinualTrainer
from src.cape_unknown import GaussianTailUnknownDetector


DEFAULT_GENERATOR_TASKS = "generator:kling2.5,generator:veo3.1,generator:wan2.5,generator:seedance1.0"
DEFAULT_ABLATIONS = (
    "full,no_discrepancy,no_pattern_guidance,no_confidence_density,"
    "no_continual_retention,no_composite_unknownness,no_conformal_calibration,"
    "no_dynamic_expansion"
)
class NullReplay:
    def sample(self, n):
        return None

    def add_batch(self, batch):
        return None


def configure_ablation(trainer, ablation):
    ablation = ablation.lower()
    if ablation == "full":
        return
    if ablation in {"no_discrepancy", "no_discrepancy_encoder"}:
        trainer.model.use_discrepancy = False
        trainer.run_config["use_discrepancy"] = False
    elif ablation in {"no_pattern", "no_pattern_guidance"}:
        trainer.model.use_pattern_guidance = False
        trainer.criterion.lambda_pattern = 0.0
        trainer.criterion.lambda_logic = 0.0
        trainer.run_config.update(
            {"use_pattern_guidance": False, "lambda_pattern": 0.0, "lambda_logic": 0.0}
        )
    elif ablation in {"no_replay_distill", "no_continual_retention"}:
        trainer.criterion.lambda_distill = 0.0
        trainer.replay = NullReplay()
        trainer.run_config.update({"lambda_distill": 0.0, "replay_enabled": False})
    elif ablation == "no_logic":
        trainer.criterion.lambda_logic = 0.0
        trainer.run_config["lambda_logic"] = 0.0
    elif ablation == "weak_router":
        trainer.criterion.lambda_router = 0.0
        trainer.run_config["lambda_router"] = 0.0
    elif ablation == "no_confidence_density":
        trainer.model.use_confidence_density = False
        trainer.run_config["use_confidence_density"] = False
    elif ablation == "no_composite_unknownness":
        trainer.model.unknown_component_indices = (1,)
        trainer.run_config["unknown_component_indices"] = [1]
    elif ablation == "no_conformal_calibration":
        trainer.unknown_detector = GaussianTailUnknownDetector()
        trainer.run_config["unknown_detector"] = "gaussian_tail"
    elif ablation == "no_dynamic_expansion":
        trainer.allow_expert_expansion = False
        trainer.run_config["allow_expert_expansion"] = False
    else:
        raise ValueError(f"Unsupported CAPE ablation: {ablation}")


def main():
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description="Run focused CAPE ablation experiments.")
    parser.add_argument("--metadata", default="data/cape_metadata.csv")
    parser.add_argument("--tasks", default=DEFAULT_GENERATOR_TASKS)
    parser.add_argument("--ablations", default=DEFAULT_ABLATIONS)
    parser.add_argument("--max-length", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--d-hid", type=int, default=3072)
    parser.add_argument("--nlayers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--expert-bottleneck", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", default="results/cape_ablations")
    parser.add_argument("--backbone-ckpt", default="")
    parser.add_argument("--backbone-mode", choices=["precomputed_avhubert", "legacy_dimodif"], default="precomputed_avhubert")
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--unknown-alpha", type=float, default=0.05)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    task_sequence = [task.strip() for task in args.tasks.split(",") if task.strip()]
    ablations = [name.strip() for name in args.ablations.split(",") if name.strip()]
    for ablation in ablations:
        output_dir = os.path.join(args.output_root, f"seed_{args.seed}", ablation)
        print(f"\n[CAPE Ablation] Running {ablation} -> {output_dir}")
        trainer = CAPEContinualTrainer(
            metadata_csv=args.metadata,
            task_sequence=task_sequence,
            max_length=args.max_length,
            d_model=args.d_model,
            nhead=args.nhead,
            d_hid=args.d_hid,
            nlayers=args.nlayers,
            batch_size=args.batch_size,
            expert_bottleneck=args.expert_bottleneck,
            top_k=args.top_k,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            output_dir=output_dir,
            show_progress=not args.no_progress,
            eval_every_epoch=True,
            save_best=True,
            freeze_backbone=not args.train_backbone,
            backbone_ckpt=args.backbone_ckpt or None,
            backbone_mode=args.backbone_mode,
            calibration_capacity=1024,
            lambda_distill=1.0,
            unknown_alpha=args.unknown_alpha,
        )
        configure_ablation(trainer, ablation)
        trainer.run_config["ablation"] = ablation
        with open(os.path.join(output_dir, "run_config.json"), "w") as handle:
            json.dump(trainer.run_config, handle, indent=2)
        trainer.run()
        print(f"[CAPE Ablation] Finished {ablation}")


if __name__ == "__main__":
    main()
