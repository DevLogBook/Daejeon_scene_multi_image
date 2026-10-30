"""Executable safeguards for the new phase trainer (no HPC data required)."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Dataset, Subset

from grm_matcher.config import GRMMatcherConfig
from grm_matcher.matcher import GRMMatcher
from grm_matcher.coordinates import flat_indices_to_image_xy
from grm_matcher.depth_curriculum import (PHASES, CurriculumObjective, StepMixture, local_terms,
    parameter_group, phase_mode, make_optimizer, reference_metrics)
from grm_matcher.scratch_transfer import lr_multiplier
from grm_matcher.train_depth_curriculum import main, gradient_audit, selection_scores
from tools.audit_grm_curriculum_data import audit


def tiny_config():
    return GRMMatcherConfig(input_height=64, input_width=64, architecture_version='v3', v3_revision=1,
        topk_candidates=1, classification_mode='none', coarse_dim=32, mid_dim=32, fine_dim=32,
        transformer_layers=4, transformer_heads=4, v3_mid_hidden=8, v3_fine_hidden=8,
        max_source_queries=16, v3_min_matchability=0, v3_relative_reliability=0)


class IdentityData(Dataset):
    def __init__(self, domain, scene='scene', rows=2):
        self.domain = domain
        self.signature = domain+scene
        self.records = [dict(scene=scene, source_image=f'{scene}.jpg', pair_id=f'{domain}/{scene}/{i}') for i in range(rows)]
        self.image = torch.rand((3, 64, 64), generator=torch.Generator().manual_seed(99))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        grid = flat_indices_to_image_xy(torch.arange(16)[None], 4, 16)[0]
        mask = torch.ones(1, 64, 64, dtype=torch.bool)
        sample = dict(image_a=self.image.clone(), image_b=self.image.clone(), valid_a=mask, valid_b=mask,
                      domain=self.domain, scene_id=self.records[index]['scene'], pair_id=self.records[index]['pair_id'])
        if self.domain == 'synthetic':
            y, x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
            sample.update(H_ab=torch.eye(3), gt_dense_mapping=torch.stack((x, y)).float(),
                          gt_overlap_a=mask, gt_overlap_b=mask)
        else:
            for direction in ('ab', 'ba'):
                sample.update({direction+'_target': grid.clone(), direction+'_valid': torch.ones(16, dtype=torch.bool),
                               direction+'_weight': torch.ones(16)})
        return sample


def batch(domain='megadepth'):
    from torch.utils.data._utils.collate import default_collate
    return default_collate([IdentityData(domain)[0]])


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gradient_scopes_and_frozen_weights(self):
        for phase in 'ABC':
            torch.manual_seed(7)
            model = GRMMatcher(tiny_config())
            optimizer = make_optimizer(model, phase, 1e-4)
            before = {n: p.detach().clone() for n, p in model.named_parameters() if parameter_group(n) not in PHASES[phase]['lr']}
            objective = CurriculumObjective(model.config, phase, anchor_points=8)
            loss, metrics = objective.agri(model, batch(), True)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            report = gradient_audit(model, phase)
            for name, stats in report.items():
                self.assertEqual(stats['squared_norm'] > 0, name in PHASES[phase]['lr'])
            optimizer.step()
            for name, previous in before.items():
                torch.testing.assert_close(model.get_parameter(name), previous, rtol=0, atol=0)
            if phase == 'A':
                self.assertEqual(metrics['anchor'], 0)
                self.assertEqual(metrics['refinement'], 0)
                self.assertEqual(metrics['confidence'], 0)
            else:
                self.assertGreater(metrics['anchor_fine_nll_supervised_count'], 0)

    def test_frozen_stateful_module_eval(self):
        model = GRMMatcher(tiny_config())
        model.backbone.probe = torch.nn.BatchNorm1d(3)
        phase_mode(model, 'B', True)
        self.assertFalse(model.backbone.probe.training)
        self.assertTrue(model.mid_refiner.training)
        phase_mode(model, 'C', True)
        self.assertTrue(model.backbone.probe.training)

    def test_fine_ramp_and_quality_mask(self):
        obj = CurriculumObjective(tiny_config(), 'B')
        self.assertEqual(obj.fine_weight(True), .1)
        obj.update = 1000
        self.assertEqual(obj.fine_weight(True), .25)
        self.assertEqual(obj.fine_weight(False), 1.)
        initial = torch.tensor([[[7.5, 7.5]]]); target = initial+torch.tensor([[[28., 0.]]])
        valid = torch.ones(1, 1, dtype=torch.bool)
        mid = SimpleNamespace(dst_xy=initial+torch.tensor([[[24., 0.]]]), valid_mask=valid,
                              local_probability=torch.full((1, 1, 7, 7), 1/49, requires_grad=True))
        fine = SimpleNamespace(dst_xy=target.clone().requires_grad_(), valid_mask=valid,
                               local_probability=torch.full((1, 1, 5, 5), 1/25, requires_grad=True))
        loss, metrics = local_terms(mid, fine, target, initial, valid, torch.ones(1, 1), tiny_config(), .25, ~valid)
        self.assertEqual(metrics['local_supervised_count'], 1)
        self.assertEqual(metrics['fine_nll_supervised_count'], 0)
        loss.backward()
        self.assertGreater(float(mid.local_probability.grad.abs().sum()), 0)
        _, disabled = local_terms(mid, fine, target, initial, valid, torch.ones(1, 1), tiny_config(), 0., valid)
        self.assertEqual(disabled['fine_coordinate_supervised_count'], 0)
        self.assertEqual(disabled['fine_nll_supervised_count'], 0)

    def test_unknown_labels_and_no_output_do_not_look_good(self):
        model = GRMMatcher(tiny_config()); sample = batch()
        for direction in ('ab', 'ba'):
            sample[direction+'_valid'][:, 0] = False
            sample[direction+'_target'][:, 0] = float('nan')
            sample[direction+'_weight'][:, 0] = 0
        loss, metrics = CurriculumObjective(model.config, 'C', anchor_points=8).agri(model, sample, True)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics['reference_count'], 15)
        good = reference_metrics(dict(reference_count=100, reference_scored_matches=50, output_matches=50,
                                      correct_at_3px=40))
        fewer = reference_metrics(dict(reference_count=100, reference_scored_matches=1, output_matches=1,
                                       correct_at_3px=1))
        empty = reference_metrics(dict(reference_count=100, output_matches=0))
        self.assertGreater(fewer['pck3'], good['pck3'])
        self.assertLess(fewer['reference_f1_3'], good['reference_f1_3'])
        self.assertEqual(empty['reference_f1_3'], 0)

    def test_exact_quota_and_prefetch_independent_resume(self):
        sources = {d: IdentityData(d, rows=4) for d in PHASES['A']['quota']}
        for phase in 'ABC':
            whole = StepMixture(sources, phase, 2026, 0, 8)
            part = StepMixture(sources, phase, 2026, 3, 5)
            self.assertEqual(whole.plan[30:], part.plan)
            for i in range(8):
                domains = [d for d, _ in whole.plan[i*10:(i+1)*10]]
                self.assertEqual({d: domains.count(d) for d in sources}, PHASES[phase]['quota'])

    def test_schedule_endpoints(self):
        self.assertEqual(lr_multiplier(0, 10000, 500, .1), .1)
        self.assertEqual(lr_multiplier(499, 10000, 500, .1), 1.)
        self.assertEqual(lr_multiplier(9999, 10000, 500, .1), .1)

    def test_metadata_benchmark_scene_rejected(self):
        metadata = dict(recipe='v3_depth_cache_v1', size=512, inventory=[], group_split={})
        for domain, scene, group, split in (
            ('megadepth', '0001', 'megadepth/0001', 'train'),
            ('megadepth', '0005', 'megadepth/0005', 'val'),
            ('uav', 'field1', 'uav/field1', 'train'),
            ('uav', 'field2', 'uav/field2', 'val'),
        ):
            metadata['inventory'].append(dict(domain=domain, scene=scene, group=group, frames=1))
            metadata['group_split'][group] = split
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'preparation.json'
            path.write_text(json.dumps(metadata))
            self.assertEqual(audit(path)['status'], 'metadata_only_pass')
            metadata['inventory'].append(dict(domain='megadepth', scene='0015', group='megadepth/0015', frames=1))
            metadata['group_split']['megadepth/0015'] = 'train'
            changed = Path(folder)/'preparation.json'; changed.write_text(json.dumps(metadata))
            self.assertEqual(audit(changed)['status'], 'failed')

    def test_main_resume_equivalence_and_phase_handoff(self):
        """Run the real optimizer/validation/checkpoint loop with tiny controlled data."""
        def fixture(args, config):
            sources = {d: IdentityData(d) for d in PHASES['A']['quota']}
            validation = {d: Subset(IdentityData(d, 'val', rows=1), [0]) for d in sources}
            info = dict(audit=dict(status='test_fixture_only'), validation_records={d: ds.dataset.records for d, ds in validation.items()})
            return sources, validation, info
        with tempfile.TemporaryDirectory(prefix='grm_curriculum_test_') as folder:
            root = Path(folder); init = root/'init.pt'
            torch.manual_seed(11); cfg = tiny_config()
            torch.save(dict(config=cfg.to_dict(), model=GRMMatcher(cfg).state_dict()), init)
            common = ['--geometry-cache', str(root/'geometry'), '--data-dir', str(root/'synthetic'),
                      '--steps', '2', '--warmup-updates', '0', '--eval-every', '1', '--save-every', '1',
                      '--num-workers', '0', '--device', 'cpu', '--anchor-points', '8', '--log-every', '0']
            with patch('grm_matcher.train_depth_curriculum.build_data', fixture), patch('grm_matcher.train_depth_curriculum.synthetic_signature', return_value='controlled_fixture'):
                main(common+['--phase', 'A', '--init-checkpoint', str(init), '--output-dir', str(root/'full')])
                main(common+['--phase', 'A', '--init-checkpoint', str(init), '--output-dir', str(root/'split'), '--stop-after-updates', '1'])
                main(common+['--phase', 'A', '--resume', str(root/'split/latest.pt'), '--output-dir', str(root/'split')])
                full = torch.load(root/'full/latest.pt', weights_only=False)
                resumed = torch.load(root/'split/latest.pt', weights_only=False)
                for name in full['model']:
                    torch.testing.assert_close(full['model'][name], resumed['model'][name], rtol=0, atol=0)
                self.assertEqual(full['phase_step'], resumed['phase_step'])
                self.assertEqual(full['best_scores'], resumed['best_scores'])
                self.assertEqual(full['optimizer']['param_groups'], resumed['optimizer']['param_groups'])
                for name, state in full['optimizer']['state'].items():
                    for key, value in state.items():
                        torch.testing.assert_close(value, resumed['optimizer']['state'][name][key], rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, 'Resume contract changed'):
                    main(common+['--phase', 'A', '--resume', str(root/'split/latest.pt'), '--output-dir', str(root/'split'), '--anchor-jitter', '0'])
                with self.assertRaisesRegex(ValueError, 'exact same curriculum phase'):
                    main(common+['--phase', 'B', '--resume', str(root/'split/latest.pt'), '--output-dir', str(root/'bad_resume')])
                parent = root/'full/latest.pt'
                for phase in ('B', 'C'):
                    main(common+['--phase', phase, '--init-checkpoint', str(parent), '--output-dir', str(root/phase), '--stop-after-updates', '1'])
                    parent = root/phase/'latest.pt'
                    result = torch.load(parent, weights_only=False)
                    self.assertEqual(result['phase_step'], 1)
                    self.assertEqual({g['name'] for g in result['optimizer']['param_groups']}, set(PHASES[phase]['lr']))


if __name__ == '__main__':
    unittest.main()
