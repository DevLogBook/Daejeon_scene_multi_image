"""Opt-in, scene-balanced IMC curriculum; covisibility is not pixel overlap GT."""
from collections import Counter, defaultdict
import math

import torch
from torch.utils.data import Sampler


class IMCCurriculumSampler(Sampler):
    """Equal scene mass, then difficulty mass, then uniform pairs per bucket.

    Draw with replacement, keeping the existing number of epoch microbatches.
    Missing buckets are renormalized within each scene. Seed+epoch makes the
    sequence reproducible at epoch-boundary resume, independently of workers.
    """

    def __init__(self, records, seed=2026, curriculum_epochs=10):
        if not records or curriculum_epochs < 1:
            raise ValueError("Nonempty records and positive curriculum_epochs required")
        self.seed, self.curriculum_epochs, self.epoch = seed, curriculum_epochs, 0
        self.groups = defaultdict(lambda: defaultdict(list))
        self.labels = []
        for index, row in enumerate(records):
            value = float(row["covisibility"])
            if not math.isfinite(value) or not .1 <= value <= 1:
                raise ValueError("Curriculum requires covisibility in [0.1, 1]")
            bucket = 0 if value < .3 else 1 if value < .6 else 2
            scene = str(row["scene"])
            self.groups[scene][bucket].append(index)
            self.labels.append((scene, bucket))
        self.last_summary = {}

    def __len__(self):
        return len(self.labels)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def weights(self):
        # Low/mid/high: 10/30/60 -> 30/35/35; retain hard pairs from epoch one.
        progress = min(1., self.epoch / max(1, self.curriculum_epochs - 1))
        mass = [a + progress * (b - a) for a, b in zip((.1, .3, .6), (.3, .35, .35))]
        result = torch.zeros(len(self), dtype=torch.double)
        for buckets in self.groups.values():
            available_mass = sum(mass[b] for b in buckets)
            for bucket, indices in buckets.items():
                result[indices] = mass[bucket] / available_mass / len(indices) / len(self.groups)
        return result, mass

    def __iter__(self):
        weights, mass = self.weights()
        indices = torch.multinomial(weights, len(self), replacement=True,
            generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        scenes = Counter(self.labels[i][0] for i in indices)
        buckets = Counter(self.labels[i][1] for i in indices)
        self.last_summary = {"epoch": self.epoch + 1, "nominal_low_mid_high": mass,
            "sampled_scene_counts": dict(scenes),
            "sampled_low_mid_high": [buckets[i] for i in range(3)],
            "draws": len(indices), "unique_pairs": len(set(indices))}
        return iter(indices)
