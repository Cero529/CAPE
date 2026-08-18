import ast
import csv
import hashlib
import os

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight envs
    pd = None


PATTERN_TO_ID = {
    "real": 0,
    "visual_only": 1,
    "audio_only": 2,
    "audio_visual": 3,
    "temporal_local": 4,
    "generator": 5,
    "unknown": 6,
}


def infer_pattern_id(video_target, audio_target, fake_periods=None, generator=None):
    if int(video_target) == 0 and int(audio_target) == 0:
        return PATTERN_TO_ID["real"]
    if fake_periods not in (None, "", "[]") and len(_parse_periods(fake_periods)) > 0:
        return PATTERN_TO_ID["temporal_local"]
    if generator not in (None, "", "unknown", "none") and str(generator).lower() != "nan":
        return PATTERN_TO_ID["generator"]
    if int(video_target) == 1 and int(audio_target) == 0:
        return PATTERN_TO_ID["visual_only"]
    if int(video_target) == 0 and int(audio_target) == 1:
        return PATTERN_TO_ID["audio_only"]
    return PATTERN_TO_ID["audio_visual"]


def _parse_periods(value):
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, float) and np.isnan(value):
        return []
    try:
        return ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []


def _is_missing(value):
    if value is None or value == "":
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return str(value).strip().lower() in {"nan", "none", "null"}


def _availability(row, key, default):
    value = row.get(key, None)
    return int(default if _is_missing(value) else float(value))


def _stable_validation_bucket(sample_id, seed=0):
    payload = f"{int(seed)}:{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / float(2**64)


def pad_features(video_features, audio_features, max_length, valid_length=None):
    stored_length = min(video_features.shape[0], audio_features.shape[0], max_length)
    if stored_length < 1:
        raise ValueError("audio-visual feature sequences must contain at least one time step")
    if valid_length is None:
        valid_length = stored_length
    valid_length = int(valid_length)
    if valid_length < 1 or valid_length > stored_length:
        raise ValueError(
            f"valid_length must lie in [1, {stored_length}], got {valid_length}"
        )
    video_padding = video_features.new_zeros(max_length - stored_length, video_features.shape[1])
    audio_padding = audio_features.new_zeros(max_length - stored_length, audio_features.shape[1])
    video = torch.cat([video_features[:stored_length], video_padding])
    audio = torch.cat([audio_features[:stored_length], audio_padding])
    return video.float(), audio.float(), valid_length


def pad_backbone_features(backbone_features, max_length):
    backbone_features = torch.as_tensor(backbone_features)
    if backbone_features.ndim == 1:
        return backbone_features.float()
    if backbone_features.ndim != 2:
        raise ValueError(
            "backbone_features must be a pooled vector [D] or sequence [T, D], "
            f"got {tuple(backbone_features.shape)}"
        )
    t = min(backbone_features.shape[0], max_length)
    padding = backbone_features.new_zeros(max_length - t, backbone_features.shape[1])
    return torch.cat([backbone_features[:t], padding]).float()


def explicit_valid_length(npz_file):
    """Read the pair-validity information emitted by the AV-HuBERT extractor."""

    for key in ("valid_length", "pair_valid_length"):
        if key in npz_file.files:
            value = np.asarray(npz_file[key]).reshape(-1)
            if value.size != 1:
                raise ValueError(f"{key} must be a scalar, got shape {np.asarray(npz_file[key]).shape}")
            return int(value[0])
    for key in ("pair_valid_mask", "valid_mask"):
        if key in npz_file.files:
            mask = np.asarray(npz_file[key]).reshape(-1)
            if mask.size == 0:
                raise ValueError(f"{key} must contain at least one entry")
            binary = mask.astype(bool)
            valid_length = int(binary.sum())
            expected = np.arange(mask.size) < valid_length
            if not np.array_equal(binary, expected):
                raise ValueError(f"{key} must be a contiguous valid prefix followed by padding")
            return valid_length
    return None


def infer_dataset_for_task(task_id):
    if task_id is None:
        return None
    task_id = str(task_id)
    if task_id.startswith("fakeavceleb:"):
        return "fakeavceleb"
    if task_id.startswith("generator:"):
        return "hifi_avdf"
    if task_id in {"1", "2", "3", "4"}:
        return "avdeepfake1m"
    return None


class CAPEContinualDataset(Dataset):
    """Dataset backed by CAPE's unified metadata CSV."""

    def __init__(
        self,
        metadata_csv,
        split,
        max_length,
        task_id=None,
        include_real=True,
        real_ratio=1.0,
        root_dir=".",
        dataset_filter=None,
        validation_role=None,
        calibration_fraction=0.5,
        partition_seed=0,
        sample_ids=None,
        exclude_sample_ids=None,
        confirmed_emerging=False,
        require_backbone_features=False,
        require_explicit_valid_length=False,
    ):
        self.metadata_csv = metadata_csv
        self.split = split
        self.max_length = max_length
        self.root_dir = root_dir
        self.confirmed_emerging = bool(confirmed_emerging)
        self.require_backbone_features = bool(require_backbone_features)
        self.require_explicit_valid_length = bool(require_explicit_valid_length)
        if validation_role not in {None, "checkpoint", "calibration"}:
            raise ValueError("validation_role must be None, 'checkpoint', or 'calibration'")
        if not 0.0 < float(calibration_fraction) < 1.0:
            raise ValueError("calibration_fraction must lie strictly between 0 and 1")
        sample_ids = None if sample_ids is None else {str(value) for value in sample_ids}
        exclude_sample_ids = set() if exclude_sample_ids is None else {str(value) for value in exclude_sample_ids}
        dataset_filter = dataset_filter or infer_dataset_for_task(task_id)
        # Filter while streaming the CSV.  Materializing the complete unified
        # table (more than 800k rows) as pandas records before selecting one
        # continual stage can consume tens of gigabytes on Windows and is also
        # copied into every DataLoader worker.  Early filtering preserves the
        # exact sample set while keeping only the rows needed by this dataset.
        table = _read_metadata(
            metadata_csv,
            split=split,
            dataset_filter=dataset_filter,
            task_id=task_id,
            include_real=include_real,
            validation_role=validation_role,
            calibration_fraction=calibration_fraction,
            partition_seed=partition_seed,
            sample_ids=sample_ids,
            exclude_sample_ids=exclude_sample_ids,
        )
        if task_id is not None:
            task_id = str(task_id)
            task_rows = [row for row in table if str(row.get("task_id", "")) == task_id]
            fake_rows = [row for row in task_rows if int(row.get("is_fake", 0)) == 1]
            if include_real:
                if task_id.startswith("generator:"):
                    real_rows = [row for row in task_rows if int(row.get("is_fake", 0)) == 0]
                else:
                    real_rows = [row for row in table if int(row.get("is_fake", 0)) == 0]
                if len(fake_rows) and len(real_rows):
                    rng = np.random.default_rng(0)
                    n_real = min(len(real_rows), max(1, int(len(fake_rows) * real_ratio)))
                    real_rows = [real_rows[i] for i in rng.choice(len(real_rows), size=n_real, replace=False)]
                table = fake_rows + real_rows
            else:
                table = fake_rows
        for row in table:
            if "pattern_id" not in row or row["pattern_id"] == "":
                row["pattern_id"] = infer_pattern_id(
                    row.get("video_target", 0),
                    row.get("audio_target", 0),
                    row.get("fake_periods", None),
                    row.get("generator", None),
                )
        self.table = table
        self.name = "cape"

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        row = self.table[idx]
        feature_path = self._resolve_feature_path(str(row["feature_path"]))
        with np.load(feature_path, allow_pickle=False) as data:
            video_features = torch.from_numpy(np.asarray(data["video_features"]))
            audio_features = torch.from_numpy(np.asarray(data["audio_features"]))
            valid_length = explicit_valid_length(data)
            if self.require_explicit_valid_length and valid_length is None:
                raise ValueError(
                    f"Paper AV-HuBERT profile requires valid_length or pair_valid_mask in {feature_path}"
                )
            if self.require_backbone_features and "backbone_features" not in data.files:
                raise ValueError(
                    f"Paper AV-HuBERT profile requires backbone_features in {feature_path}"
                )
            backbone_features = (
                pad_backbone_features(np.asarray(data["backbone_features"]), self.max_length)
                if "backbone_features" in data.files
                else None
            )
        video_features, audio_features, valid_length = pad_features(
            video_features,
            audio_features,
            self.max_length,
            valid_length=valid_length,
        )
        if backbone_features is not None and backbone_features.ndim == 2:
            if valid_length > backbone_features.shape[0]:
                raise ValueError(
                    f"valid_length={valid_length} exceeds backbone sequence length "
                    f"{backbone_features.shape[0]} in {feature_path}"
                )
        video_target = int(row.get("video_target", 0))
        audio_target = int(row.get("audio_target", 0))
        fake_period_value = row.get("fake_periods", None)
        fake_periods = _parse_periods(fake_period_value)
        has_fake_period = int(len(fake_periods) > 0)
        pattern_id = int(row.get("pattern_id", 0))
        unknown_value = row.get("unknown_flag", "")
        unknown_flag = (
            int(pattern_id == PATTERN_TO_ID["unknown"])
            if _is_missing(unknown_value)
            else int(float(unknown_value))
        )
        if self.confirmed_emerging:
            unknown_flag = 1
        pattern_availability = torch.tensor(
            [
                _availability(row, "video_available", 1),
                _availability(row, "audio_available", 1),
                _availability(row, "audio_visual_available", 1),
                _availability(row, "temporal_available", int(not _is_missing(fake_period_value))),
                _availability(row, "unknown_available", 1),
            ],
            dtype=torch.float32,
        )
        if self.confirmed_emerging:
            pattern_availability[-1] = 1.0
        return {
            "video_features": video_features,
            "audio_features": audio_features,
            "backbone_features": backbone_features,
            "valid_length": torch.tensor(valid_length).long(),
            "labels": torch.tensor([video_target, audio_target]).float(),
            "pattern_id": torch.tensor(pattern_id).long(),
            "task_id": str(row.get("task_id", "")),
            "sample_id": str(row.get("sample_id", idx)),
            "has_fake_period": torch.tensor(has_fake_period).float(),
            "unknown_flag": torch.tensor(unknown_flag).float(),
            "pattern_availability": pattern_availability,
            "is_fake": torch.tensor(int(row.get("is_fake", int(video_target or audio_target)))).float(),
        }

    def _resolve_feature_path(self, feature_path):
        if not os.path.isabs(feature_path):
            feature_path = os.path.join(self.root_dir, feature_path)
        if os.path.exists(feature_path):
            return feature_path
        # Backward compatibility for metadata generated before detector-name
        # directories were included in AV-Deepfake1M feature paths.
        dirname, basename = os.path.split(feature_path)
        fallback = os.path.join(dirname, "mediapipe", basename)
        if basename == "features.npz" and os.path.exists(fallback):
            return fallback
        return feature_path


def cape_collate(batch):
    out = {
        "video_features": torch.stack([item["video_features"] for item in batch]),
        "audio_features": torch.stack([item["audio_features"] for item in batch]),
        "valid_length": torch.stack([item["valid_length"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "pattern_id": torch.stack([item["pattern_id"] for item in batch]),
        "has_fake_period": torch.stack([item["has_fake_period"] for item in batch]),
        "unknown_flag": torch.stack([item["unknown_flag"] for item in batch]),
        "pattern_availability": torch.stack([item["pattern_availability"] for item in batch]),
        "is_fake": torch.stack([item["is_fake"] for item in batch]),
        "task_id": [item["task_id"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
    }
    backbone = [item["backbone_features"] for item in batch]
    if any(value is not None for value in backbone):
        if not all(value is not None for value in backbone):
            raise ValueError("A batch cannot mix samples with and without backbone_features")
        out["backbone_features"] = torch.stack(backbone)
    else:
        out["backbone_features"] = None
    if "stream_unknown" in batch[0]:
        out["stream_unknown"] = torch.stack(
            [torch.as_tensor(item["stream_unknown"]).long() for item in batch]
        )
    return out


def _read_metadata(
    metadata_csv,
    split=None,
    dataset_filter=None,
    task_id=None,
    include_real=True,
    validation_role=None,
    calibration_fraction=0.5,
    partition_seed=0,
    sample_ids=None,
    exclude_sample_ids=None,
):
    """Read only rows that can belong to the requested continual dataset.

    Filtering is intentionally performed during CSV iteration instead of
    after ``pandas.DataFrame.to_dict``.  For a non-generator task, the
    candidate set contains that task's fake rows plus dataset-matched real
    rows, exactly as selected by ``CAPEContinualDataset`` below.  Generator
    tasks already carry their matched real references under the same task id.
    """

    split = None if split is None else str(split)
    dataset_filter = None if dataset_filter is None else str(dataset_filter)
    task_id = None if task_id is None else str(task_id)
    generator_task = bool(task_id and task_id.startswith("generator:"))
    exclude_sample_ids = set() if exclude_sample_ids is None else exclude_sample_ids
    table = []
    with open(metadata_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_id = str(row.get("sample_id", ""))
            if sample_ids is not None and sample_id not in sample_ids:
                continue
            if sample_id in exclude_sample_ids:
                continue
            if split is not None and str(row.get("split", "")) != split:
                continue
            if validation_role is not None:
                if str(row.get("split", "")) != "val":
                    continue
                is_calibration = _stable_validation_bucket(sample_id, partition_seed) < float(
                    calibration_fraction
                )
                if validation_role == "calibration" and not is_calibration:
                    continue
                if validation_role == "checkpoint" and is_calibration:
                    continue
            if dataset_filter is not None and str(row.get("dataset", "")) != dataset_filter:
                continue
            if task_id is not None:
                row_task = str(row.get("task_id", ""))
                is_fake = int(row.get("is_fake", 0)) == 1
                if generator_task:
                    if row_task != task_id:
                        continue
                elif not (is_fake and row_task == task_id) and not (include_real and not is_fake):
                    continue
            table.append(row)
    return table
