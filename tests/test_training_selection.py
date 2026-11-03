"""Regression tests for warm-up, baseline selection and checkpoint resume."""
import tempfile
import random
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
import yaml

from test_feature_graph import StagedBackbone
from models.feature_graph import FeatureGraphRefiner
from utils.scheduler import CosineDecay, PolynomialDecayLR
from utils.training_tools import Trainer
from train import train


class TrainingSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        with open('config/wildfire_graph.yaml') as stream:
            cls.config = yaml.safe_load(stream)

    def test_first_update_uses_warmup_and_resume_preserves_next_rate(self):
        for kind in ('COS', 'POLY'):
            param = torch.nn.Parameter(torch.ones(1))
            opt = torch.optim.SGD([param], lr=1e-4)
            make = lambda optimizer: (CosineDecay(optimizer, 1e-4, 10, 4, 2)
                                      if kind == 'COS' else
                                      PolynomialDecayLR(optimizer, 1e-4, 40, 4, warmup_epochs=2))
            scheduler = make(opt)
            rates = []
            for _ in range(8):
                rates.append(opt.param_groups[0]['lr'])
                opt.step()
                scheduler.step()
            self.assertAlmostEqual(rates[0], 1e-4 / 8)
            self.assertAlmostEqual(rates[-1], 1e-4)
            self.assertEqual(rates, sorted(rates))
            restored_opt = torch.optim.SGD([param], lr=1e-4)
            restored = make(restored_opt)
            restored_opt.load_state_dict(opt.state_dict())
            restored.load_state_dict(scheduler.state_dict())
            self.assertEqual(restored.get_last_lr(), scheduler.get_last_lr())
            restored.step()
            scheduler.step()
            self.assertEqual(restored.get_last_lr(), scheduler.get_last_lr())

    def make_trainer(self):
        args = dict(self.config, DEVICE='cpu', NUM_WORKERS=0, BATCHSIZE=2,
                    EPOCHS=3, WARMUP=1, STOPCOUNTER=2)
        model = FeatureGraphRefiner(StagedBackbone(), teacher=None, hidden_dim=16,
                                    heads=4, layers=1, patch_size=4, nonlocal_k=1,
                                    backbone_mode='frozen')
        return args, model, Trainer(args, model, 5)

    def test_comparison_identity_ignores_void_pixels(self):
        args, model, trainer = self.make_trainer()
        images = torch.randn(1, 4, 16, 16)
        labels = torch.zeros(1, 16, 16, dtype=torch.long)
        labels[:, :2] = 255
        result = trainer.valid_step((images, labels, torch.zeros_like(labels), ['x']), compare=True)
        _, refined, coarse, fixed, broken = result
        np.testing.assert_array_equal(refined, coarse)
        self.assertEqual(refined.sum(), 14 * 16)
        self.assertEqual((fixed, broken), (0, 0))
        self.assertEqual(trainer.scheduler.num_batches, 2)  # 5 // 2, drop_last

    def test_resume_restores_random_generators(self):
        args, model, trainer = self.make_trainer()
        with tempfile.TemporaryDirectory() as folder:
            trainer.save_checkpoint(folder, 0, filename='latest.pth')
            expected=(random.random(), np.random.rand(), torch.rand(4))
            random.random(); np.random.rand(); torch.rand(4)
            args['CHECKPOINT']=str(Path(folder)/'latest.pth')
            Trainer(args,model,5)
            actual=(random.random(), np.random.rand(), torch.rand(4))
            self.assertEqual(actual[0],expected[0])
            self.assertEqual(actual[1],expected[1])
            torch.testing.assert_close(actual[2],expected[2],rtol=0,atol=0)

    def run_selection(self, scores, expected_epoch):
        args, model, trainer = self.make_trainer()
        trainer.training_step = Mock(return_value=(torch.tensor(0.2), np.eye(3)))
        data = [0] * 5
        logger = Mock(best_miou=float('-inf'), best_epoch=None)

        def log(metrics, epoch, scheduler, step=None):
            if metrics.get('val miou', -1) > logger.best_miou:
                logger.best_miou, logger.best_epoch = metrics['val miou'], epoch
        logger.log.side_effect = log
        validation = [({'val miou': score, 'val coarse miou': .8}, np.ones(3)) for score in scores]
        with tempfile.TemporaryDirectory() as folder:
            args['PROJECTNAME'] = folder  # absolute join keeps all checkpoints temporary
            with patch('train.Trainer', return_value=trainer), patch('train.valid', side_effect=validation):
                # train assigns per-class IoU on the dataset after validation.
                class Dataset(list):
                    pass
                train(model, Dataset(data), data, args, logger)
            path = Path(folder) / args['SESSIONAME']
            baseline = torch.load(path / 'baseline.pth', weights_only=False)
            best = torch.load(path / 'best.pth', weights_only=False)
            self.assertEqual(baseline['epoch'], 0)
            self.assertEqual(best['best_epoch'], expected_epoch)
            self.assertEqual(best['best_validation_metric'], max(scores))
            self.assertEqual(best['baseline_miou'], .8)
            self.assertEqual(logger.log.call_args_list[0].kwargs['step'], 0)
            self.assertEqual(logger.log.call_args_list[1].kwargs['step'], 1)
            last = path / f'checkpoint_epoch_{len(scores) - 2}.pth'
            args['CHECKPOINT'] = str(last)
            resumed = Trainer(args, model, len(data))
            self.assertEqual(resumed.best_epoch, expected_epoch)
            self.assertEqual(resumed.baseline_miou, .8)
            self.assertEqual(resumed.stop_cur_counter, trainer.stop_cur_counter)
            self.assertEqual(resumed.start_epoch, len(scores) - 1)

    def test_degraded_training_keeps_baseline_as_best(self):
        self.run_selection([.8, .7, .79], -1)

    def test_improvement_replaces_baseline_and_survives_resume(self):
        self.run_selection([.8, .8001, .799, .799], 0)


if __name__ == '__main__':
    unittest.main()
