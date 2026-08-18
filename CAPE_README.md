# CAPE: Continual Audio-Visual Pattern Experts with Unknown Forgery Discovery

This directory adds CAPE as a side-by-side research extension. The original
audio-visual detector code is left intact and can still be used as a baseline.

## What Was Added

- `src/cape_models.py`: CAPE model wrapper and multi-term loss.
- `src/cape_experts.py`: dynamic pattern expert bank with add/freeze/merge/prune hooks.
- `src/cape_router.py`: audio-visual discrepancy encoder and task-free router.
- `src/cape_unknown.py`: conformal unknown-pattern detector and unknown queue.
- `src/cape_logic.py`: weak neuro-symbolic pattern targets and logic loss.
- `src/cape_datasets.py`: unified CAPE metadata dataset.
- `src/cape_continual.py`: continual training loop with replay and distillation.
- `src/cape_metrics.py`: continual and unknown-detection metrics.
- `src/cape_memory.py`: replay buffer and fixed-capacity calibration reservoirs.
- `scripts/cape_build_metadata.py`: build a unified metadata CSV.
- `scripts/cape_pattern_incremental.py`: run pattern-incremental training.
- `scripts/cape_experiments/run_open_world_protocol.py`: complete four-fold discovery and adaptation protocol.
- `scripts/cape_smoke_test.py`: synthetic self-check for model, loss, expansion, and unknown calibration.

## Unified Metadata Format

`scripts/cape_pattern_incremental.py` expects a CSV with:

```text
sample_id,dataset,feature_path,split,
video_target,audio_target,is_fake,
pattern_id,task_id,generator,fake_periods
```

For the manuscript profile, every `feature_path` NPZ must contain three
arrays: `video_features`, `audio_features`, and `backbone_features`. The last
array is the frozen AV-HuBERT final multimodal sequence (or its pooled
768-dimensional vector). Each file must also contain either scalar
`valid_length` or a contiguous-prefix `pair_valid_mask`, so zero padding is
excluded from pooling and discrepancy statistics. Missing paper-profile keys
are hard errors; the code no longer silently substitutes the legacy DiMoDif
detector for the manuscript backbone.

Recommended pattern tasks:

```text
1: visual-only fake
2: audio-only fake
3: audio-visual fake
4: temporal-local fake
generator:<name>: generator-incremental task
```

Real samples should not be a separate task. They are sampled as negative
references inside each task.

## Quick Checks

Run the smoke test:

```bash
python scripts/cape_smoke_test.py
```

Expected output includes:

```text
CAPE smoke test passed
```

Run syntax checks:

```powershell
$files = @(Get-ChildItem -LiteralPath 'src' -Filter 'cape_*.py') + @(Get-ChildItem -LiteralPath 'scripts' -Filter 'cape_*.py')
foreach ($f in $files) { python -m py_compile $f.FullName }
```

Audit every feature file before a paper run:

```powershell
python scripts\cape_experiments\audit_avhubert_features.py --metadata data\cape_metadata.csv --report results\avhubert_feature_audit.json
```

The audit requires the three feature arrays, explicit pair-validity
information, `T=100`, `d_0=768`, finite floating-point values, and a pooled
or sequential 768-dimensional AV-HuBERT backbone representation.

## Example Training

Build metadata:

```bash
python scripts/cape_build_metadata.py --root . --output data/cape_metadata.csv
```

Run AV-Deepfake1M pattern-incremental CAPE:

```bash
python scripts/cape_pattern_incremental_av1M.py
```

The AV1M-specific script now defaults to the paper settings: tasks `1,2,3`,
`T=100`, `d=768`, batch size 64, replay capacity 1024, 20 epochs, and official
`test` evaluation. To replay historical DiMoDif-feature experiments, pass
`--backbone-mode legacy_dimodif` together with the original dimensions and
checkpoint explicitly.

```bash
python scripts/cape_pattern_incremental_av1M.py --epochs 20
```

The continual history is written to:

```text
results/cape_pattern/history.json
```

## Paper Mapping

- Unknown forgery discovery: `cape_unknown.py`
- Dynamic expert expansion: `cape_experts.py`, `CAPEModel.add_expert`
- Task-free routing: `cape_router.py`
- Weak neuro-symbolic pattern learning: `cape_logic.py`
- Continual training with replay/distillation: `cape_continual.py`
- Disjoint fixed-capacity calibration: `FeatureReservoir` and
  `CAPEContinualTrainer.update_calibration_reservoir`
- Four-fold confirmation/adaptation: `run_open_world_protocol.py`
