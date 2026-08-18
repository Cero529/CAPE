# CAPE Experiment Scripts

This folder separates CAPE experiments by purpose.

## 1. Main CAPE Training

Use the existing entry when you want to train the proposed CAPE model:

```powershell
python scripts\cape_pattern_incremental.py
```

The default is the manuscript profile: `T=100`, `d=768`, batch size 64,
20 epochs, distillation weight 1.0, and
`backbone_mode=precomputed_avhubert`. Every feature NPZ must contain
`video_features`, `audio_features`, and the frozen AV-HuBERT final-layer
sequence `backbone_features`. The runner fails rather than silently replacing
AV-HuBERT with an incompatible detector. Historical compatibility runs remain
available only through an explicit non-paper configuration.

Paper-profile NPZ files must also contain scalar `valid_length` or a
contiguous-prefix `pair_valid_mask`. Verify the complete metadata inventory
before training:

```powershell
python scripts\cape_experiments\audit_avhubert_features.py --metadata data\cape_metadata.csv --report results\avhubert_feature_audit.json
python scripts\cape_experiments\audit_paper_protocol.py
```

The protocol audit fails if official AV-Deepfake1M test rows, the five
FakeAVCeleb stages, matched HiFi splits, or manuscript held-out counts are not
actually present. A failed audit blocks paper-result aggregation rather than
silently producing `NaN` or changing the protocol; resolve it before running
the five-seed aggregation commands below.

Default task stream:

```text
generator:kling2.5 -> generator:veo3.1 -> generator:wan2.5 -> generator:seedance1.0
```

The manuscript's 12-stage heterogeneous stream is run through the same entry
with the exact ordered task IDs below (the metadata must contain the five
`fakeavceleb:*` stages):

```powershell
python scripts\cape_pattern_incremental.py --tasks "fakeavceleb:faceswap,fakeavceleb:fsgan,fakeavceleb:wav2lip,fakeavceleb:rtvc,fakeavceleb:audio_visual,1,2,3,generator:kling2.5,generator:veo3.1,generator:wan2.5,generator:seedance1.0" --output-dir results\cape_heterogeneous\seed_0
```

The checked-in metadata currently has no FakeAVCeleb rows, so this protocol
must not be reported until those frozen AV-HuBERT features and metadata rows
are supplied and pass the feature audit.

## 2. Continual-Learning Baselines

Use this for real continual-learning baselines under the same stream:

```powershell
python scripts\cape_experiments\run_continual_baselines.py --methods seq_ft,er,lwf,der,derpp,ewc
```

Implemented methods:

- `seq_ft`: sequential fine-tuning.
- `er`: exemplar replay.
- `lwf`: Learning without Forgetting-style response distillation.
- `der`: replay with stored logits.
- `derpp`: stored-logit replay plus supervised replay.
- `ewc`: diagonal Fisher regularization.
- `joint`: offline joint-training reference.
- `single_task`: per-task oracle reference.

## 3. CAPE Ablations

Use this for focused CAPE ablations:

```powershell
python scripts\cape_experiments\run_cape_ablation.py
```

Implemented ablations:

- `full`
- `no_discrepancy`
- `no_pattern_guidance`
- `no_confidence_density`
- `no_continual_retention`
- `no_composite_unknownness`
- `no_conformal_calibration`
- `no_dynamic_expansion`

The paper table should keep only a small number of core ablations.

## 4. Unknown Discovery

Regenerate the leakage-safe matched-pair metadata with the manuscript's
fixed 60/20/20 split before running HiFi-AVDF protocols:

```powershell
python scripts\cape_experiments\build_hifi_grouped_metadata.py
```

Use this for held-out generator unknown scoring:

```powershell
python scripts\cape_experiments\run_unknown_discovery.py
```

Default known generators:

```text
kling2.5, veo3.1, wan2.5
```

Default unknown generator:

```text
seedance1.0
```

This script evaluates sample-level unknown scoring and forms DBSCAN candidate clusters from rejected samples. The bounded unknown queue uses FIFO eviction: appending at capacity removes the oldest queued item. The script records detection delay by replaying DBSCAN on the current queue prefix after each rejection until the first non-noise candidate cluster appears; the sample counter includes all observed discovery samples.

For the complete manuscript protocol, including all four held-out-generator
folds, FPR95, ARI, FCR, 32-sample confirmation, unknown-to-known adaptation,
and old-source drop, run:

```powershell
python scripts\cape_experiments\run_open_world_protocol.py
```

Run that entry through `run_multiseed.py` for seeds 0--4 before exporting the
paper table. It writes one auditable JSON record per fold and a macro summary.

```powershell
python scripts\cape_experiments\run_multiseed.py scripts\cape_experiments\run_open_world_protocol.py --seeds 0,1,2,3,4 --output-root results\cape_open_world --output-mode dir
python scripts\cape_experiments\summarize_open_world_multiseed.py results\cape_open_world --paper-mode
```

The second command verifies every fold configuration and produces the
five-seed mean and sample standard deviation for all eight manuscript
open-world metrics.

## Matched multi-seed protocol

Run CAPE with the five required seeds:

```powershell
python scripts\cape_experiments\run_multiseed.py scripts\cape_pattern_incremental.py --seeds 0,1,2,3,4 --output-root results\cape_multiseed --output-mode dir -- --epochs 20
```

Aggregate performance and stage-wise router/prototype diagnostics:

```powershell
python scripts\cape_experiments\summarize_multiseed.py results\cape_multiseed --paper-mode
```

`--paper-mode` refuses to aggregate missing seeds, missing run configs, or
runs whose backbone, dimensions, batch size, epoch count, replay budget,
optimizer settings, checkpoint policy, distillation weight, or calibration
capacity differ from the manuscript. Baseline configs are checked against the
shared feature/training profile without requiring CAPE-only calibration
fields.

The closest-forensic-baseline fairness contract and source status are in
`scripts/cape_experiments/matched_baselines.json`. Both ACM MM 2025
baselines require external source code or a separately audited
reimplementation; this repository does not fabricate proxy results for
them.

## Synchronization robustness

```powershell
python scripts\cape_experiments\run_cape_robustness.py --seeds 0,1,2,3,4 --shift-steps=-8,-4,-2,0,2,4,8
```

This applies zero-filled audio shifts without circular wrap and writes one
`temporal_shift_robustness.json` file per seed.

## 5. Table Export

After multiple methods finish, export unified tables:

```powershell
python scripts\cape_experiments\export_result_tables.py --result-root results --output-dir results\tables
```

Outputs:

- `summary_rows.json`
- `summary_table.md`
- `summary_table_rows.tex`
