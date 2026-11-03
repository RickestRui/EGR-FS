import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.tools import *
from utils.training_tools import *
from utils.logger import Logger


def train(model, train_data, val_data, args, logger):
    worker_options = {'prefetch_factor': 1} if args['NUM_WORKERS'] > 0 else {}
    train_loader = DataLoader(train_data, batch_size=args['BATCHSIZE'], shuffle=True,
                             num_workers=args['NUM_WORKERS'], drop_last=True,
                             pin_memory=args['DEVICE'] != 'cpu', **worker_options)
    trainer = Trainer(args, model, len(train_data))
    if len(train_loader) == 0:
        raise ValueError('Training split must contain at least BATCHSIZE samples.')
    graph_run = args['MODEL'].startswith('robofire_graph')
    checkpoint_dir = os.path.join('weights', args['PROJECTNAME'], args['SESSIONAME'])
    logger.best_miou, logger.best_epoch = trainer.best_validation_metric, trainer.best_epoch
    if graph_run and trainer.start_epoch == 0 and trainer.baseline_miou is None:
        baseline, _ = valid(trainer, val_data, 'baseline', args)
        trainer.baseline_miou = baseline['val coarse miou']
        trainer.best_validation_metric = baseline['val miou']
        trainer.best_metric = baseline['val miou']
        trainer.best_epoch = -1
        logger.log(baseline, -1, trainer.scheduler, step=0)
        if args.get('SAVE_BASELINE_CHECKPOINT', True):
            trainer.save_checkpoint(checkpoint_dir, -1, filename='baseline.pth')
        trainer.save_checkpoint(checkpoint_dir, -1, filename='best.pth')
        print('Baseline evaluated before any optimizer update (best epoch -1 = initialization).')
    for epoch in range(trainer.start_epoch, args['EPOCHS']):
        if hasattr(model, 'set_epoch'):
            model.set_epoch(epoch)
        running_loss, conf_mat = 0, np.zeros((args['NUM_CLASSES'], args['NUM_CLASSES']))
        # train loop
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
            loss, curr_conf_mat = trainer.training_step(batch)
            conf_mat += curr_conf_mat
            running_loss += loss.item() / len(train_loader)
        metrics, _ = calculate_metrics(conf_mat, running_loss, cls_names=args['CLS_NAMES'], cls_weights=args['CLASS_WEIGHTS'])
        if hasattr(model, 'set_epoch'):
            metrics['trainable backbone params'] = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
        # validation, saving model and examining stop criterion
        if epoch % args['VALID_FREQ'] == 0:
            val_metrics, iou = valid(trainer, val_data, epoch, args)
            metrics = {**metrics, **val_metrics}
            is_best = val_metrics['val miou'] > trainer.best_validation_metric
            if is_best:
                trainer.best_validation_metric = val_metrics['val miou']
                trainer.best_epoch = epoch
            should_stop = trainer.stop_sign(metrics)
            save_every = args.get('SAVE_EVERY', 1)
            if save_every > 0 and ((epoch + 1) % save_every == 0 or should_stop or epoch == args['EPOCHS'] - 1):
                trainer.save_checkpoint(checkpoint_dir, epoch)
            if args.get('SAVE_LATEST', False):
                trainer.save_checkpoint(checkpoint_dir, epoch, filename='latest.pth')
            if is_best:
                trainer.save_checkpoint(checkpoint_dir, epoch, filename='best.pth')
            train_data.mious = iou
        logger.log(metrics, epoch, trainer.scheduler, step=epoch + 1 if graph_run else epoch)
        if epoch % args['VALID_FREQ'] == 0:
            logger.print_metrics(val_metrics, args['CLS_NAMES'], args['NUM_CLASSES'], cls_weights=args['CLS_NAMES'], val=True)
        logger.print_metrics(metrics, cls_names=args['CLS_NAMES'], num_classes=args['NUM_CLASSES'], cls_weights=args['CLS_NAMES'], val=False)
        if epoch % args['VALID_FREQ'] == 0 and should_stop:
            break
    best_label = 'baseline (before training)' if trainer.best_epoch == -1 else trainer.best_epoch
    print(f"Training Finished! Best Epoch {best_label}: MIOU {trainer.best_validation_metric}")
    if graph_run and trainer.baseline_miou is not None:
        print(f"Best minus initial RFFNet mIoU: {trainer.best_validation_metric - trainer.baseline_miou:+.6f}")
    return logger.best_miou

def valid(trainer, val_data, cur_epoch, args):
    valid_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args['NUM_WORKERS'])
    running_loss, conf_mat = 0.0, np.zeros((args['NUM_CLASSES'], args['NUM_CLASSES']))
    graph_run = args['MODEL'].startswith('robofire_graph')
    coarse_mat = np.zeros_like(conf_mat)
    repaired, damaged = 0, 0
    for batch in tqdm(valid_loader, desc='Validation'):
        if graph_run:
            loss, curr_conf_mat, coarse, fixed, broken = trainer.valid_step(batch, compare=True)
            coarse_mat += coarse
            repaired += fixed
            damaged += broken
        else:
            loss, curr_conf_mat = trainer.valid_step(batch)
        conf_mat += curr_conf_mat
        running_loss += loss.item() / len(valid_loader)
    metrics, iou = calculate_metrics(conf_mat, running_loss, cls_names=args['CLS_NAMES'], cls_weights=args['CLASS_WEIGHTS'], val=True)
    if graph_run:
        coarse_metrics, _ = calculate_metrics(coarse_mat, 0., cls_names=args['CLS_NAMES'],
                                              cls_weights=args['CLASS_WEIGHTS'])
        metrics.update({f'val coarse {key}': value for key, value in coarse_metrics.items() if key != 'avgloss'})
        for key in ['miou'] + [f'iou {name}' for name in args['CLS_NAMES']]:
            metrics[f'val delta {key}'] = metrics[f'val {key}'] - coarse_metrics[key]
        metrics.update({'val repaired pixels': repaired, 'val damaged pixels': damaged,
                        'val net repaired pixels': repaired - damaged})
        if trainer.baseline_miou is not None:
            metrics['val delta initial miou'] = metrics['val miou'] - trainer.baseline_miou
        print(f"Validation {cur_epoch}: coarse={coarse_metrics['miou']:.6f}, "
              f"refined={metrics['val miou']:.6f}, delta={metrics['val delta miou']:+.6f}, "
              f"repaired={repaired}, damaged={damaged}")
    if args.get('SAVE_VISUALIZATIONS', True):
        qualitive_eval(lambda data: trainer.inference(data), val_data,
                   ex_path=f'./outputs/{args["PROJECTNAME"]}/{args["SESSIONAME"]}/visualizations', name=f'Epoch_{cur_epoch}.png')
    return metrics, iou

def main(args):
    checkpoint_dir = os.path.join('weights', args['PROJECTNAME'], args['SESSIONAME'])
    if not args['CHECKPOINT'] and os.path.isdir(checkpoint_dir) and os.listdir(checkpoint_dir):
        raise ValueError('Session already has checkpoints. Choose a new --SESSIONAME or resume with --CHECKPOINT.')
    set_reproducibility(args['SEED'])
    train_dataset, val_dataset, _ = get_dataset(args)
    model = get_model(args)
    logger = Logger(args)
    train(model, train_dataset, val_dataset, args, logger)
    logger.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
