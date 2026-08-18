import torch


def shift_temporal_features(features, shift_steps):
    """Shift a [B, T, D] sequence with zero fill and no circular wrap."""
    if features.ndim != 3:
        raise ValueError(f"features must have shape [B, T, D], got {tuple(features.shape)}")
    shift_steps = int(shift_steps)
    if shift_steps == 0:
        return features
    length = features.shape[1]
    if abs(shift_steps) >= length:
        return torch.zeros_like(features)
    shifted = torch.zeros_like(features)
    if shift_steps > 0:
        shifted[:, shift_steps:] = features[:, : length - shift_steps]
    else:
        offset = -shift_steps
        shifted[:, : length - offset] = features[:, offset:]
    return shifted
