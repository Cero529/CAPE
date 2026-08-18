import torch
import torch.nn.functional as F


PATTERN_NAMES = (
    "visual",
    "audio",
    "audio_visual",
    "temporal",
    "unknown",
)
FAKE_LOGIC_INDICES = (0, 1, 3, 4)


def build_weak_pattern_targets(video_target, audio_target, has_fake_period=None, unknown_flag=None):
    """Build weak CAPE pattern labels from dataset metadata.

    Pattern order is visual, audio, audio-visual, temporal, unknown.
    The labels are multi-hot because one sample can contain multiple cues.
    """
    video_target = video_target.float()
    audio_target = audio_target.float()
    device = video_target.device
    targets = torch.zeros((video_target.shape[0], len(PATTERN_NAMES)), device=device)

    targets[:, 0] = video_target
    targets[:, 1] = audio_target
    targets[:, 2] = video_target * audio_target

    if has_fake_period is not None:
        targets[:, 3] = has_fake_period.float().to(device)
    if unknown_flag is not None:
        targets[:, 4] = unknown_flag.float().to(device)
    return targets


def soft_logic_fake_prob(pattern_probs):
    """Noisy-or over non-redundant fake attributes.

    The joint audio-visual attribute is a conjunction derived from the
    visual and audio attributes, so including it would count the same
    evidence twice in the sample-level logic rule.
    """
    independent_evidence = pattern_probs[..., list(FAKE_LOGIC_INDICES)]
    return 1.0 - torch.prod(
        1.0 - independent_evidence.clamp(1e-6, 1.0 - 1e-6),
        dim=-1,
    )


def logic_consistency_loss(fake_logits, pattern_logits, fake_targets):
    """Align fake probability with the soft logical composition of patterns."""
    pattern_probs = torch.sigmoid(pattern_logits)
    logic_fake = soft_logic_fake_prob(pattern_probs)
    model_fake = torch.sigmoid(fake_logits).max(dim=-1).values
    target_fake = fake_targets.max(dim=-1).values.float()

    consistency = F.binary_cross_entropy(model_fake, logic_fake.detach())
    supervised_logic = F.binary_cross_entropy(logic_fake, target_fake)
    return consistency + supervised_logic


def pattern_supervision_loss(pattern_logits, pattern_targets, availability=None):
    loss = F.binary_cross_entropy_with_logits(
        pattern_logits,
        pattern_targets.float(),
        reduction="none",
    )
    if availability is None:
        return loss.mean()
    availability = availability.float().to(pattern_logits.device)
    if availability.shape != loss.shape:
        raise ValueError(
            f"pattern availability must have shape {tuple(loss.shape)}, got {tuple(availability.shape)}"
        )
    return (loss * availability).sum() / availability.sum().clamp_min(1.0)
