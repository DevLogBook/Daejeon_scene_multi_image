"""Keep reference visibility, known nonmatches, and unknown labels distinct."""
import torch


def confidence_targets(reference_valid, output_valid, error, weights, *, real, label_known=None):
    """Return correctness targets/mask/weights; GT never affects inference."""
    truth = (reference_valid & (error.detach() <= 3)).float()
    if real:
        return truth, reference_valid & output_valid, weights
    mask = output_valid if label_known is None else output_valid & label_known
    # Exact synthetic nonmatches have no positive-coordinate weight, but do
    # have a known negative correctness label. Do not multiply them by zero.
    return truth, mask, torch.ones_like(weights)
