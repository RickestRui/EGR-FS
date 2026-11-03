"""Compare coarse/refined segmentation, boundaries, small fires and latency."""
import argparse
import json
import time

import numpy as np
from scipy.ndimage import binary_dilation, label
from scipy.optimize import linear_sum_assignment
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import yaml

from utils.tools import get_confusion_matrix, set_reproducibility
from utils.training_tools import Trainer, get_dataset, get_model


def segmentation_boundary(mask, valid):
    boundary = np.zeros_like(valid)
    difference = (mask[1:] != mask[:-1]) & valid[1:] & valid[:-1]
    boundary[1:] |= difference
    boundary[:-1] |= difference
    difference = (mask[:, 1:] != mask[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    boundary[:, 1:] |= difference
    boundary[:, :-1] |= difference
    return boundary


def boundary_counts(prediction, truth, valid, tolerance=2):
    pred = segmentation_boundary(prediction, valid)
    gt = segmentation_boundary(truth, valid)
    if tolerance:
        pred_near = binary_dilation(pred, iterations=tolerance) & valid
        gt_near = binary_dilation(gt, iterations=tolerance) & valid
    else:
        pred_near, gt_near = pred, gt
    return np.array([(pred & gt_near).sum(), pred.sum(),
                     (gt & pred_near).sum(), gt.sum()], dtype=np.int64)


def small_fire_counts(prediction, truth, valid, max_area=64, match_iou=0.1, fire_class=2):
    """One-to-one IoU matching of 8-connected small GT fire components."""
    connectivity = np.ones((3, 3), dtype=np.uint8)
    gt, ng = label((truth == fire_class) & valid, connectivity)
    pred, npred = label((prediction == fire_class) & valid, connectivity)
    areas = np.bincount(gt.ravel(), minlength=ng + 1)
    selected = np.flatnonzero((areas[1:] > 0) & (areas[1:] <= max_area)) + 1
    if not len(selected) or not npred:
        return np.array([0, len(selected)], dtype=np.int64)
    intersection = np.bincount((gt * (npred + 1) + pred).ravel(),
                               minlength=(ng + 1) * (npred + 1)).reshape(ng + 1, npred + 1)
    pred_areas = np.bincount(pred.ravel(), minlength=npred + 1)[1:]
    overlap = intersection[selected, 1:]
    union = areas[selected, None] + pred_areas[None] - overlap
    iou = overlap / np.maximum(union, 1)
    eligible = iou >= match_iou
    # Prioritize the number of valid matches, then use IoU to break ties.
    rows, cols = linear_sum_assignment(-eligible.astype(float) - 1e-3 * iou)
    matched = eligible[rows, cols].sum()
    return np.array([matched, len(selected)], dtype=np.int64)


def summarize(matrix, boundaries, small, class_names):
    true_positive = np.diag(matrix)
    actual = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    union = actual + predicted - true_positive
    divide = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    iou = divide(true_positive, union)
    precision = divide(true_positive, predicted)
    recall = divide(true_positive, actual)
    bp = float(boundaries[0] / boundaries[1]) if boundaries[1] else 0.
    br = float(boundaries[2] / boundaries[3]) if boundaries[3] else 0.
    return {
        'miou': float(iou.mean()),
        'per_class': {name: {'iou': float(iou[i]), 'precision': float(precision[i]),
                             'recall': float(recall[i])} for i, name in enumerate(class_names)},
        'boundary_f1': 2 * bp * br / (bp + br) if bp + br else 0.,
        'small_fire_recall': float(small[0] / small[1]) if small[1] else None,
        'small_fire_matched': int(small[0]), 'small_fire_total': int(small[1]),
    }


@torch.no_grad()
def evaluate(args, max_samples=None, max_area=64, match_iou=0.1, tolerance=2, split='test'):
    if not args['MODEL'].startswith('robofire_graph') or not args.get('CHECKPOINT'):
        raise ValueError('Evaluation requires robofire_graph and a trained CHECKPOINT')
    if max_area <= 0 or not 0 < match_iou <= 1 or tolerance < 0:
        raise ValueError('Invalid small-fire or boundary metric settings')
    if max_samples is not None and max_samples <= 0:
        raise ValueError('MAX_SAMPLES must be positive')
    if 'fire' not in args['CLS_NAMES']:
        raise ValueError('Small-fire evaluation requires a fire class')
    set_reproducibility(args['SEED'])
    _, validation, testing = get_dataset(args, test=True)
    dataset = testing if split == 'test' else validation
    full_split_size = len(dataset)
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args['NUM_WORKERS'])
    if not len(loader):
        raise ValueError('Evaluation dataset is empty')
    model = get_model(args)
    trainer = Trainer(args, model, len(dataset), test=True)
    model.eval()
    # Inference intervention only; this does not replace a separately trained
    # capacity-matched no-message baseline.
    if args.get('ABLATE_MESSAGES', False):
        for layer in model.refiner.layers:
            layer.attention.register_forward_hook(lambda module, inputs, result: torch.zeros_like(result))
    classes = args['NUM_CLASSES']
    matrices = {name: np.zeros((classes, classes), dtype=np.int64) for name in ('coarse', 'refined')}
    boundaries = {name: np.zeros(4, dtype=np.int64) for name in matrices}
    small = {name: np.zeros(2, dtype=np.int64) for name in matrices}
    repaired = damaged = valid_pixels = 0
    timings = []
    device = torch.device(args['DEVICE'])

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    for index, batch in enumerate(tqdm(loader, desc=f'Graph comparison ({split})')):
        images = batch[0].to(device=device, dtype=torch.float)
        truth = batch[1].to(device=device, dtype=torch.long)
        if index == 0:
            for _ in range(3):
                model(images)
            synchronize()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
        synchronize()
        start = time.perf_counter()
        output = model(images)
        synchronize()
        timings.append(time.perf_counter() - start)
        gt = truth[0].cpu().numpy()
        valid = gt != args['IGNORE_LABEL']
        predictions = {}
        for name, key in (('coarse', 'coarse_logits'), ('refined', 'refined_logits')):
            logits = output[key]
            matrices[name] += get_confusion_matrix(truth, logits, classes, args['IGNORE_LABEL'])
            predictions[name] = logits.argmax(1)[0].cpu().numpy()
            boundaries[name] += boundary_counts(predictions[name], gt, valid, tolerance)
            small[name] += small_fire_counts(predictions[name], gt, valid, max_area,
                                             match_iou, args['CLS_NAMES'].index('fire'))
        before, after = predictions['coarse'] == gt, predictions['refined'] == gt
        repaired += int((valid & ~before & after).sum())
        damaged += int((valid & before & ~after).sum())
        valid_pixels += int(valid.sum())
    summary = {name: summarize(matrices[name], boundaries[name], small[name], args['CLS_NAMES'])
               for name in matrices}
    summary.update(samples=len(timings), split=split, repaired_pixels=repaired,
                   damaged_pixels=damaged, valid_pixels=valid_pixels,
                   graph_pipeline_ms=float(np.mean(timings) * 1000),
                   graph_pipeline_fps=float(1 / np.mean(timings)),
                   timing_scope='RFFNet + graph construction + GNN + pixel head; excludes I/O and metrics',
                   small_fire_max_area=max_area, small_fire_match_iou=match_iou,
                   boundary_tolerance=tolerance)
    summary['coarse_scope'] = ('jointly tuned RFFNet, not the initial frozen baseline'
                              if args.get('BACKBONE_MODE', 'frozen') != 'frozen'
                              else 'frozen RFFNet')
    summary['message_ablation'] = bool(args.get('ABLATE_MESSAGES', False))
    if split == 'val' and len(dataset) == full_split_size and trainer.baseline_miou is not None:
        summary['initial_validation_miou'] = trainer.baseline_miou
        summary['refined_minus_initial_validation_miou'] = summary['refined']['miou'] - trainer.baseline_miou
    if device.type == 'cuda':
        summary['peak_allocated_cuda_mb'] = torch.cuda.max_memory_allocated(device) / 2**20
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', default='wildfire_graph.yaml')
    parser.add_argument('--CHECKPOINT', required=True)
    parser.add_argument('--DEVICE')
    parser.add_argument('--NUM_WORKERS', type=int, default=0)
    parser.add_argument('--ABLATE_MESSAGES', action='store_true', help='Diagnostic: zero GNN attention messages at inference')
    parser.add_argument('--MAX_SAMPLES', type=int)
    parser.add_argument('--SPLIT', choices=('val', 'test'), default='val')
    parser.add_argument('--SMALL_FIRE_AREA', type=int, default=64)
    parser.add_argument('--MATCH_IOU', type=float, default=0.1)
    parser.add_argument('--BOUNDARY_TOLERANCE', type=int, default=2)
    options = parser.parse_args()
    with open('config/' + options.yaml_file) as stream:
        args = yaml.safe_load(stream)
    args.update(CHECKPOINT=options.CHECKPOINT, NUM_WORKERS=options.NUM_WORKERS)
    args['ABLATE_MESSAGES'] = options.ABLATE_MESSAGES
    if options.DEVICE:
        args['DEVICE'] = options.DEVICE
    summary = evaluate(args, options.MAX_SAMPLES, options.SMALL_FIRE_AREA,
                       options.MATCH_IOU, options.BOUNDARY_TOLERANCE, options.SPLIT)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
