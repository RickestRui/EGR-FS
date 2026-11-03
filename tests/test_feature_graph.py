import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn
from models.feature_graph import FeatureGraphRefiner
from utils.total_loss import FeatureGraphLoss
from utils.training_tools import Trainer


def loss_args():
    return dict(DEVICE='cpu', NUM_CLASSES=3, NUM_OUTPUTS=2, ALIGN_CORNERS=True,
                IGNORE_LABEL=255, CLASS_WEIGHTS=[1., 1., 1.], T_THRESH_BDLOSS=.8,
                BD_WEIGHT=1., USE_OHEM=False, BALANCE_WEIGHTS=[.4, 1.], SB_WEIGHTS=1.)


class StagedBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.planes, self.augment = 8, True
        self.encoder = nn.Sequential(nn.Conv2d(4, 32, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.layer5 = nn.Conv2d(32, 32, 1)
        self.final_layer = nn.Conv2d(32, 3, 1)
        self.seghead_d = nn.Conv2d(32, 1, 1)

    def forward(self, x, return_features=False):
        x = self.layer5(self.encoder(x))
        logits = self.final_layer(x)
        outputs = [logits, logits, self.seghead_d(x)]
        return dict(outputs=outputs, fusion=x, semantic=x, rgb=x[:, :8], ir=x[:, 8:16]) if return_features else outputs


class FeatureGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make(self):
        torch.manual_seed(200)
        base, teacher = StagedBackbone(), StagedBackbone()
        teacher.load_state_dict(base.state_dict())
        return FeatureGraphRefiner(base, teacher=teacher, unfreeze_epoch=1,
                                   num_classes=3, hidden_dim=16, heads=4, layers=2, patch_size=4)

    def test_initial_identity_first_step_gnn_gradient_and_staged_unfreezing(self):
        model = self.make().train()
        x = torch.rand(1, 4, 16, 16)
        labels = torch.randint(0, 3, (1, 16, 16))
        criterion = FeatureGraphLoss(loss_args())
        output = model(x)
        torch.testing.assert_close(output['refined_logits'], output['coarse_logits'], rtol=0, atol=0)
        criterion.get_loss(output, labels, torch.zeros_like(labels))[0].mean().backward()
        self.assertGreater(model.refiner.layers[0].attention.source.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model.backbone.layer5.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
        model.zero_grad()
        model.set_epoch(1)
        criterion.get_loss(model(x), labels, torch.zeros_like(labels))[0].mean().backward()
        self.assertGreater(model.backbone.layer5.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model.backbone.encoder[0].weight.grad)
        self.assertFalse(model.backbone.encoder[1].training)

    def test_feature_path_has_no_old_logit_ceiling(self):
        model = self.make().eval()
        with torch.no_grad():
            model.refiner.adapter[-1].bias.fill_(100)
            result = model(torch.rand(1, 4, 16, 16))
        self.assertGreater(result['correction'].abs().max().item(), 2.)
        self.assertNotIn('gate_logits', result)

    def test_teacher_is_fixed_and_ignore_loss_is_zero(self):
        model = self.make().train()
        x = torch.rand(1, 4, 16, 16)
        before = model(x)['teacher_logits']
        with torch.no_grad():
            model.backbone.layer5.weight.add_(1)
        output = model(x)
        torch.testing.assert_close(before, output['teacher_logits'], rtol=0, atol=0)
        labels = torch.full((1, 16, 16), 255)
        loss = FeatureGraphLoss(loss_args()).get_loss(output, labels, torch.zeros_like(labels))[0]
        self.assertEqual(loss.item(), 0.)
        loss.mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_feature_classifier_depends_on_graph_messages(self):
        model = self.make().eval()
        x = torch.rand(1, 4, 16, 16)
        with torch.no_grad():
            model.refiner.adapter[-1].weight.normal_(std=.1)
            normal = model(x)['refined_logits']
            hooks = [layer.attention.register_forward_hook(lambda module, inputs, output: torch.zeros_like(output))
                     for layer in model.refiner.layers]
            try:
                ablated = model(x)['refined_logits']
            finally:
                for hook in hooks:
                    hook.remove()
        self.assertGreater((normal - ablated).abs().max().item(), 1e-5)

    def test_resume_crosses_unfreeze_boundary_with_same_optimizer_groups(self):
        args = dict(loss_args(), MODEL='robofire_graph', OPTIM='ADAMW', LR=1e-4,
                    WD=1e-4, SCHED='COS', EPOCHS=10, WARMUP=1, BATCHSIZE=2,
                    STOPCOUNTER=3, CHECKPOINT=None, BACKBONE_MODE='staged', UNFREEZE_EPOCH=1,
                    BACKBONE_LR_FACTOR=.1, FEATURE_LR_FACTOR=.1, LATE_LR_FACTOR=.02, USE_TEACHER=True)
        trainer = Trainer(args, self.make(), 8)
        self.assertEqual(len(trainer.optimizer.param_groups), 4)
        self.assertFalse(trainer.model.backbone.layer5.weight.requires_grad)
        with tempfile.TemporaryDirectory() as folder:
            trainer.save_checkpoint(folder, 0)
            args['CHECKPOINT'] = str(Path(folder) / 'checkpoint_epoch_0.pth')
            resumed = Trainer(args, self.make(), 8)
            self.assertTrue(resumed.model.backbone.layer5.weight.requires_grad)
            self.assertFalse(any(p.requires_grad for p in resumed.model.teacher.parameters()))
            for _ in range(10):
                rates = resumed.scheduler.get_last_lr()
                self.assertAlmostEqual(rates[1] / rates[0], .1)
                self.assertAlmostEqual(rates[2] / rates[0], .1)
                self.assertAlmostEqual(rates[3] / rates[0], .02)
                resumed.scheduler.step()


if __name__ == '__main__':
    unittest.main()
