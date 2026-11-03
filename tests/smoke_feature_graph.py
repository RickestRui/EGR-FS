"""Real-data CUDA joint-gradient checks across EGR-FS semantic unfreezing."""
from pathlib import Path
import sys
import tempfile
import argparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from utils.training_tools import get_dataset, get_model, Trainer
from utils.tools import set_reproducibility


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', default='wildfire_graph.yaml')
    options = parser.parse_args()
    with open('config/' + options.yaml_file) as f:
        args = yaml.safe_load(f)
    args.update(NUM_WORKERS=0, BATCHSIZE=2, UNFREEZE_EPOCH=1)
    set_reproducibility(200)
    data, val, _ = get_dataset(args)
    model = get_model(args)
    trainer = Trainer(args, model, len(data))
    batch = next(iter(DataLoader(Subset(data, [500, 600]), batch_size=2)))
    x = batch[0].float().cuda()
    model.eval()
    with torch.no_grad():
        teacher_initial = model.teacher(x)[1].clone()
        output = model(x)
        torch.testing.assert_close(output['refined_logits'], output['coarse_logits'], rtol=0, atol=0)
    for epoch in (0, 1):
        model.set_epoch(epoch)
        loss, _ = trainer.training_step(batch)
        assert torch.isfinite(loss)
        semantic = next(model.backbone.stage4.parameters())
        assert (semantic.grad is not None) == (epoch == 1)
        assert all(p.grad is None for p in model.teacher.parameters())
        assert model.refiner.layers[0].attention.source.weight.grad.abs().sum() > 0
        print('Epoch', epoch, 'loss', loss.item(), 'late grad enabled', semantic.requires_grad)
    with torch.no_grad():
        torch.testing.assert_close(teacher_initial, model.teacher(x)[1], rtol=0, atol=0)
    validation = next(iter(DataLoader(val, batch_size=1)))
    trainer.valid_step(validation, compare=True)
    expected = trainer.inference(validation[0])
    with tempfile.TemporaryDirectory(prefix='feature_graph_') as folder:
        trainer.save_checkpoint(folder, 1)
        args['CHECKPOINT'] = str(Path(folder) / 'checkpoint_epoch_1.pth')
        restored = Trainer(args, get_model(args), len(data))
        torch.testing.assert_close(expected, restored.inference(validation[0]), rtol=0, atol=0)
        assert next(restored.model.backbone.stage4.parameters()).requires_grad
    print('Graph parameters:', sum(p.numel() for p in model.refiner.parameters()))
    print('Active backbone parameters:', sum(p.numel() for p in model.backbone.parameters() if p.requires_grad))
    print(options.yaml_file, 'identity, staged gradients, independent teacher and resume: OK')


if __name__ == '__main__':
    main()
