"""GRM adapter for the shared agricultural benchmark; images are the only inputs."""
import numpy as np


class Matcher:
    def __init__(self, name, root, args):
        import torch
        from grm_matcher import GRMMatcher, GRMMatcherConfig

        if name != "GRM":
            raise ValueError(f"This adapter supports GRM, received {name}")
        self.torch = torch
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; check GPU allocation and PyTorch/driver compatibility")
        if args.checkpoint is None or not args.checkpoint.is_file():
            raise FileNotFoundError("Supply --checkpoint pointing to a trusted GRM training checkpoint")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "config" not in checkpoint or "model" not in checkpoint:
            raise ValueError("Expected a GRM checkpoint containing config and model (e.g. best_agri.pt)")
        self.config = GRMMatcherConfig.from_dict(checkpoint["config"])
        saved_config = self.config.to_dict()
        rerank_mode = getattr(args, "grm_rerank_mode", "checkpoint")
        if rerank_mode != "checkpoint":
            self.config.candidate_rerank_mode = rerank_mode
        sampling_mode = getattr(args, "grm_sampling_mode", "checkpoint")
        if self.config.architecture_version == "v3" and (rerank_mode != "checkpoint" or sampling_mode != "checkpoint"):
            raise ValueError("V3 uses its own Top-1 sampler; legacy reranker/sampling overrides do not apply")
        if sampling_mode != "checkpoint":
            self.config.sampling_mode = sampling_mode
        for field_name in ("support_min_matchability", "support_relative_score"):
            override = getattr(args, "grm_" + field_name, None)
            if override is not None:
                if self.config.sampling_mode != "overlap_aware":
                    raise ValueError("Support threshold overrides require overlap_aware sampling")
                setattr(self.config, field_name, override)
        preset = getattr(args, "grm_sampling_preset", "checkpoint")
        topk = getattr(args, "grm_topk_candidates", None)
        if preset != "checkpoint" or topk is not None:
            if self.config.architecture_version != "v1":
                raise ValueError("These component ablation overrides require V1")
        if preset == "uniform":
            self.config.sampling_mode = "adaptive_density"
            self.config.use_adaptive_density_sampling = False
            self.config.density_power = 0.0
            self.config.use_dynamic_query_budget = False
            self.config.use_quantile_threshold = False
            self.config.use_local_nms = False
            self.config.min_quota_per_valid_region = 2
        if topk is not None:
            self.config.topk_candidates = topk
        self.config.validate()
        # Runtime image dimensions are supported by GRM; preserve the saved sampler
        # and reranker configuration instead of applying training-only overrides.
        if args.size % self.config.coarse_stride:
            raise ValueError("--size must be divisible by the checkpoint coarse_stride")
        self.model = GRMMatcher(self.config)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model = self.model.float().to(self.device).eval()
        self.metadata = {
            "model": name, "torch": torch.__version__, "device": str(self.device),
            "precision": "FP32", "checkpoint_stage": checkpoint.get("stage"),
            "gpu_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "geometry_heads_trained": checkpoint.get("geometry_heads_trained"),
            "confidence_head_trained": checkpoint.get("confidence_head_trained"),
            "checkpoint_epoch_raw": checkpoint.get("epoch"),
            "config": self.config.to_dict(), "runtime_input_size": [args.size, args.size],
            "sampling_overrides": {k: v for k, v in self.config.to_dict().items() if saved_config[k] != v},
            "selection": "valid_mask; confidence=p_match; no role-based postfilter",
            "ground_truth_used_for_estimation": False,
        }

    def sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _tensor(self, rgb):
        tensor = self.torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
        tensor = tensor.to(device=self.device, dtype=self.torch.float32).unsqueeze(0) / 255.0
        if self.config.input_range == "imagenet_normalized":
            mean = tensor.new_tensor(self.config.normalization_mean).view(1, 3, 1, 1)
            std = tensor.new_tensor(self.config.normalization_std).view(1, 3, 1, 1)
            tensor = (tensor - mean) / std
        return tensor

    def __call__(self, image_a, image_b, *, valid_a=None, valid_b=None):
        mask_args = {}
        for key, value, rgb in (("valid_a", valid_a, image_a), ("valid_b", valid_b, image_b)):
            if value is not None:
                value = np.asarray(value, dtype=bool)
                if value.shape != rgb.shape[:2]:
                    raise ValueError(f"{key} must have the same spatial dimensions as its image")
                mask_args[key] = self.torch.from_numpy(np.ascontiguousarray(value)).to(self.device)[None, None]
        with self.torch.inference_mode(), self.torch.autocast(self.device.type, enabled=False):
            output = self.model(self._tensor(image_a), self._tensor(image_b), **mask_args)
        keep = output.valid_mask[0].bool()
        return tuple(x[0][keep].detach().float().cpu().numpy()
                     for x in (output.src_xy, output.dst_xy, output.p_match))
