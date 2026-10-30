"""Explicit weight-only v1 -> v2 migration; never restores optimizer state."""
from .config import GRMMatcherConfig


def v2_config(config):
    values = config.to_dict()
    values.update(architecture_version="v2", classification_mode="none", candidate_peak_radius=1,
                  max_source_queries=max(512, config.max_source_queries),
                  sampling_mode="overlap_aware", min_quota_per_valid_region=0,
                  support_min_matchability=.2, support_relative_score=.1,
                  use_dynamic_query_budget=False, allow_underfilled_budget=True)
    return GRMMatcherConfig.from_dict(values)


def load_v1_weights(model, state):
    """Reject unrelated/partial checkpoints instead of silently skipping weights."""
    if model.config.architecture_version != "v2":
        raise ValueError("Migration requires a v2 destination model")
    expected = set(model.state_dict())
    added = {name for name in expected if name.startswith(("context_bridge.", "inlier_head.")) or name == "geometry_gain"}
    source = dict(state)
    if model.config.classification_mode in ("inlier", "none"):
        from .role_head import GRMGeometryRoleHead
        old_head = {"role_head." + key: value for key, value in GRMGeometryRoleHead(model.config).state_dict().items()}
        if any(key not in source or source[key].shape != value.shape for key, value in old_head.items()):
            raise ValueError("Expected a complete v1 role head in source checkpoint")
        for key in old_head:
            source.pop(key)
    if set(source) != expected - added:
        raise ValueError("Expected a complete v1 state dict; keys do not match this architecture")
    return model.load_state_dict(source, strict=False)
