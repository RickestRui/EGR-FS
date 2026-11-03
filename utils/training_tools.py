import torch
import numpy as np
import torch.nn.functional as F
import os
import random
from .tools import get_confusion_matrix
from .total_loss import TotalLoss, RefinementLoss, FeatureGraphLoss
from .scheduler import CosineDecay, PolynomialDecayLR
import torch.optim as optim
from models.pidnet import PIDNet
from datasets.wildfire import WildFire
from models.robofirefusenet import RoboFireFuseNet
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF


class Trainer:
    def __init__(self, args, model, len_data, test=False):
        self.model_name = args['MODEL']
        self.use_amp = False if args['DEVICE'] == 'cpu' else False
        self.grad_clip = args.get('GRAD_CLIP', 0.)
        self.model = model
        self.optimizer = self.get_optimizer(args, self.model)
        self.criterion = self.get_loss_criterion(args)
        # Training uses drop_last=True; count actual optimizer updates per epoch.
        num_batches = max(1, len_data // args['BATCHSIZE'])
        self.scheduler = self.get_scheduler(args['SCHED'], args['LR'], args['EPOCHS'], num_batches, args['WARMUP'])
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.device = args['DEVICE']
        self.num_classes = args['NUM_CLASSES']
        self.ingore_label = args['IGNORE_LABEL']
        self.stop_counter = args['STOPCOUNTER']
        self.start_epoch = 0
        self.best_metric = 0
        self.best_validation_metric = float('-inf')
        self.best_epoch = None
        self.baseline_miou = None
        self.stop_cur_counter = 0
        self.early_stop_min_delta = args.get('EARLY_STOP_MIN_DELTA', 0.001)
        self.training_config = dict(args)
        config_keys = ('MODEL', 'NUM_CLASSES', 'CROP_SIZE', 'WINDOW_SIZE', 'TF_CONFIG',
                       'GRAPH_PATCH_SIZE', 'GRAPH_HIDDEN_DIM', 'GRAPH_HEADS',
                       'GRAPH_LAYERS', 'GRAPH_NONLOCAL_K', 'GRAPH_DROPOUT',
                       'BACKBONE_MODE', 'LABEL_ENCODING', 'UNFREEZE_EPOCH', 'USE_TEACHER')
        self.model_config = {key: args[key] for key in config_keys if key in args}
        for key in ('REFINER_KIND', 'GRAPH_USE_EVIDENCE', 'GRAPH_USE_PREDICTIONS',
                    'GRAPH_USE_PRIOR', 'GRAPH_USE_EDGE_ATTR', 'GRAPH_SPATIAL', 'ZERO_INIT_ADAPTER'):
            if key in args:
                self.model_config[key] = args[key]
        if args['CHECKPOINT'] != None:
            self.load_checkpoint(os.path.join(args['CHECKPOINT']), test)
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(self.start_epoch)

    def training_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        images, labels, edges, names = batch[0].to(dtype=torch.float, device=self.device), \
            batch[1].to(dtype=torch.long, device=self.device), \
        batch[2].to(dtype=torch.float, device=self.device), batch[3]
        with torch.autocast(device_type=torch.device(self.device).type, dtype=torch.float16, enabled=self.use_amp):
            output = self.model(images)
            output_mask = F.interpolate(
                                self.segmentation_logits(output),
                                size=[images.shape[-2], images.shape[-1]],
                                mode='bilinear', align_corners=True)
            losses, _, acc, loss_list = self.criterion.get_loss(output, labels, edges)
        conf_mat = get_confusion_matrix(labels, output_mask, self.num_classes, ignore=self.ingore_label)
        loss = losses.mean()
        self.scaler.scale(loss).backward()
        if self.grad_clip:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        return loss.detach(), conf_mat

    @staticmethod
    def segmentation_logits(output):
        if isinstance(output, dict):
            return output['refined_logits']
        return output[1] if isinstance(output, (list, tuple)) else output

    @torch.no_grad()
    def valid_step(self, batch, compare=False):
        self.model.eval()
        images, labels, edges, names = batch[0].to(dtype=torch.float, device=self.device), \
            batch[1].to(dtype=torch.long, device=self.device), batch[2].to(dtype=torch.float, device=self.device), batch[3]
        output = self.model(images)
        output_mask = F.interpolate(
                            self.segmentation_logits(output),
                            size=[images.shape[-2], images.shape[-1]],
                            mode='bilinear', align_corners=True)
        conf_mat = get_confusion_matrix(labels, output_mask, self.num_classes, ignore=self.ingore_label)
        losses, _, acc, loss_list = self.criterion.get_loss(output, labels, edges)
        loss = losses.mean()
        if compare:
            coarse = F.interpolate(output['coarse_logits'], size=labels.shape[-2:],
                                   mode='bilinear', align_corners=True)
            coarse_mat = get_confusion_matrix(labels, coarse, self.num_classes, ignore=self.ingore_label)
            valid = labels != self.ingore_label
            coarse_correct = coarse.argmax(1) == labels
            refined_correct = output_mask.argmax(1) == labels
            repaired = (valid & ~coarse_correct & refined_correct).sum().item()
            damaged = (valid & coarse_correct & ~refined_correct).sum().item()
            return loss.detach(), conf_mat, coarse_mat, repaired, damaged
        return loss.detach(), conf_mat

    
    @torch.no_grad()
    def test_step(self, batch):
        self.model.eval()
        images, labels, edges, names = batch[0].to(dtype=torch.float, device=self.device), \
            batch[1].to(dtype=torch.long, device=self.device), batch[2].to(dtype=torch.float, device=self.device), batch[3]
        output = self.model(images)
        output_mask = F.interpolate(
                            self.segmentation_logits(output),
                            size=[images.shape[-2], images.shape[-1]],
                            mode='bilinear', align_corners=True)
        conf_mat = get_confusion_matrix(labels, output_mask, self.num_classes, ignore=self.ingore_label)
        losses, _, acc, loss_list = self.criterion.get_loss(output, labels, edges)
        loss = losses.mean()
        return loss.detach(), conf_mat

    @torch.no_grad()
    def inference(self, data, proc_output=True):
        self.model.eval()
        images = data.to(dtype=torch.float, device=self.device)
        output = self.model(images)
        if proc_output:
            output = self.segmentation_logits(output)
            output = F.interpolate(
                                output,
                                size=[images.shape[-2], images.shape[-1]],
                                mode='bilinear', align_corners=True)
            output = torch.argmax(output, dim=1)
        return output

    def get_optimizer(self, args, model):
        parameters = [p for p in model.parameters() if p.requires_grad]
        if args['MODEL'] == 'robofire_graph':
            parameters = model.parameter_groups(args.get('BACKBONE_LR_FACTOR', .1),
                                                 args.get('LATE_LR_FACTOR', .02),
                                                 args.get('FEATURE_LR_FACTOR', 1.))
        if args['OPTIM'] == 'SGD':
            optimizer = optim.SGD(parameters, lr=args['LR'], momentum=args['MOMENTUM'], weight_decay=args['WD']) 
        elif args['OPTIM'] == 'ADAM':
            optimizer = optim.Adam(parameters, lr=args['LR'], weight_decay=args['WD'])
        elif args['OPTIM'] == 'ADAMW':
            optimizer = optim.AdamW(parameters, lr=args['LR'], weight_decay=args['WD'])
        else:
            print('Unsupported optimizer.')
            exit()
        return optimizer

    def get_scheduler(self, sched_name, initial_lr, epochs, num_batches, warmup):
        if sched_name == 'COS':
            scheduler = CosineDecay(self.optimizer, initial_lr, epochs, num_batches, warmup)
        elif sched_name == 'POLY':
            scheduler = PolynomialDecayLR(self.optimizer, initial_lr, epochs * num_batches, num_batches, 0.9, 10, warmup_epochs=warmup)
        return scheduler

    def get_loss_criterion(self, args):
        if args['MODEL'] == 'robofire_graph':
            return FeatureGraphLoss(args)
        loss = TotalLoss(args)
        return loss

    def stop_sign(self, metrics):
        if metrics['val miou'] - self.best_metric > self.early_stop_min_delta:
            self.best_metric, self.stop_cur_counter = metrics['val miou'], 0
        else:
            self.stop_cur_counter += 1
        if self.stop_cur_counter >= self.stop_counter:
            return True
        return False
    
    def save_checkpoint(self, path, epoch, filename=None):
        os.makedirs(f'{path}', exist_ok=True)
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_metric': self.best_metric,
            'best_validation_metric': self.best_validation_metric,
            'best_epoch': self.best_epoch,
            'baseline_miou': self.baseline_miou,
            'stop_cur_counter': self.stop_cur_counter,
            'training_config': self.training_config,
            'model_config': self.model_config,
            'rng_state': {'python': random.getstate(), 'numpy': np.random.get_state(),
                          'torch': torch.get_rng_state(),
                          'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        }
        torch.save(checkpoint, os.path.join(path, filename or f'checkpoint_epoch_{epoch}.pth'))

    def load_checkpoint(self, path, test=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if not test and self.model_name == 'robofire_graph':
            previous = checkpoint.get('training_config', {})
            for key in ('LR', 'EPOCHS', 'WARMUP', 'BATCHSIZE', 'BACKBONE_LR_FACTOR',
                        'LATE_LR_FACTOR', 'FEATURE_LR_FACTOR', 'UNFREEZE_EPOCH'):
                if key in self.training_config and previous.get(key, 1. if key == 'FEATURE_LR_FACTOR' else None) != self.training_config[key]:
                    raise ValueError(f'Resume requires original training setting {key}; start a new session for a changed schedule.')
        saved_config = checkpoint.get('model_config', {})
        # Accept the legacy checkpoint label for the same feature graph architecture.
        # Its state layout is unchanged, so retain checkpoint usability after
        # the public name was simplified to robofire_graph.
        if saved_config.get('MODEL') == 'robofire_graph_v4' and self.model_name == 'robofire_graph':
            saved_config = dict(saved_config, MODEL='robofire_graph')
        if self.model_name.startswith('robofire_graph'):
            for key, value in self.model_config.items():
                if key not in ('FREEZE_BACKBONE', 'UNFREEZE_EPOCH') and key in saved_config and saved_config[key] != value:
                    raise ValueError(f'Checkpoint config mismatch for {key}: {saved_config[key]} != {value}')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if(not test):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.start_epoch = checkpoint['epoch']
            rng = checkpoint.get('rng_state')
            if rng:
                random.setstate(rng['python'])
                np.random.set_state(rng['numpy'])
                torch.set_rng_state(rng['torch'].cpu())
                if torch.cuda.is_available() and rng.get('cuda') is not None:
                    torch.cuda.set_rng_state_all([state.cpu() for state in rng['cuda']])
        self.best_metric = checkpoint['best_metric']
        self.best_validation_metric = checkpoint.get('best_validation_metric', self.best_metric)
        self.best_epoch = checkpoint.get('best_epoch')
        self.baseline_miou = checkpoint.get('baseline_miou')
        self.stop_cur_counter = checkpoint.get('stop_cur_counter', 0)
        print('Checkpoint loaded!')

def get_dataset(args, test=False):
    train_dataset = WildFire(root=args['ROOTDATASET'],
                          list_path=args['TRAINSET'],
                          num_classes=args['NUM_CLASSES'],
                          multi_scale=args['MULTISCALE'],
                          flip=args['FLIP'],
                          brightness=args['BRIGHTNESS'],
                          ignore_label=args['IGNORE_LABEL'],
                          scale_factor=args['SCALE_FACTOR'],
                          crop_size=args['CROP_SIZE'],
                          base_size=args['BASE_SIZE'],
                          bd_dilate_size=4,
                          mode=args['MODE'],
                          blend_images=args['BLEND_IMGS'],
                          comp_mask=args['COMP_MASK'],
                          single_source=args['SINGLE_SOURCE'],
                          label_encoding=args.get('LABEL_ENCODING'),
                          seed=args['SEED']
                          )
    val_dataset = WildFire(root=args['ROOTDATASET'],
                          list_path=args['VALIDSET'],
                          num_classes=args['NUM_CLASSES'],
                          multi_scale=False,
                          flip=False,
                          brightness=False,
                          ignore_label=args['IGNORE_LABEL'],
                          scale_factor=args['SCALE_FACTOR'],
                          crop_size=args['CROP_SIZE'],
                          base_size=args['BASE_SIZE'],
                          bd_dilate_size=4,
                          mode=args['MODE'],
                          blend_images=False,
                          comp_mask=False,
                          single_source=False,
                          label_encoding=args.get('LABEL_ENCODING'),
                          seed=args['SEED'], eval_resize=args.get('EVAL_RESIZE', False))
    test_dataset = WildFire(root=args['ROOTDATASET'],
                        list_path=args['TESTSET'],
                        num_classes=args['NUM_CLASSES'],
                        multi_scale=False,
                        flip=False,
                        brightness=False,
                        ignore_label=args['IGNORE_LABEL'],
                        scale_factor=args['SCALE_FACTOR'],
                        crop_size=args['CROP_SIZE'],
                        base_size=args['BASE_SIZE'],
                        bd_dilate_size=4,
                        mode=args['MODE'],
                        blend_images=False,
                        comp_mask=False,
                        single_source=False,
                        label_encoding=args.get('LABEL_ENCODING'),
                        seed=args['SEED'], eval_resize=args.get('EVAL_RESIZE', False))
    return train_dataset, val_dataset, test_dataset if test else ()

def get_model(args):
    channels = {'rgb':3, 'ir':1, 'fusion':4}
    if 'pidnet_s' == args['MODEL']:
        model = PIDNet(m=2, n=3, num_classes=args['NUM_CLASSES'], planes=32, ppm_planes=96, head_planes=128, augment=True, channels=channels[args['MODE']])
    elif 'pidnet_m' == args['MODEL']:
        model = PIDNet(m=2, n=3, num_classes=args['NUM_CLASSES'], planes=64, ppm_planes=96, head_planes=128, augment=True, channels=channels[args['MODE']])
    elif 'pidnet_l' == args['MODEL']:
        model = PIDNet(m=3, n=4, num_classes=args['NUM_CLASSES'], planes=64, ppm_planes=112, head_planes=256, augment=True, channels=channels[args['MODE']])
    elif 'robofire' == args['MODEL']:
        model = RoboFireFuseNet(m=2, n=3, num_classes=args['NUM_CLASSES'], planes=32, ppm_planes=96, head_planes=128, augment=True, channels=channels[args['MODE']], input_resolution=args['CROP_SIZE'], window_size=(args['WINDOW_SIZE'], args['WINDOW_SIZE']), tf_depths=args['TF_CONFIG'])
    elif args['MODEL'] == 'robofire_graph':
        if args['MODE'] != 'fusion':
            raise ValueError('robofire_graph requires RGB + IR (MODE: fusion)')
        base_options = dict(m=2, n=3, num_classes=args['NUM_CLASSES'], planes=32,
                              ppm_planes=96, head_planes=128, augment=True, channels=4,
                              input_resolution=args['CROP_SIZE'],
                              window_size=(args['WINDOW_SIZE'], args['WINDOW_SIZE']),
                              tf_depths=args['TF_CONFIG'])
        base = RoboFireFuseNet(**base_options)
        # Full graph checkpoints include the backbone; never silently accept
        # partial base weights when starting stage two.
        if args.get('CHECKPOINT') is None:
            if not args.get('PRETRAINED'):
                raise ValueError('Set PRETRAINED to trained RFFNet weights for stage two')
            state = torch.load(args['PRETRAINED'], map_location='cpu', weights_only=False)
            if 'model_state_dict' in state:
                state = state['model_state_dict']
            elif 'state_dict' in state:
                state = state['state_dict']
            state = {key.removeprefix('module.'): value for key, value in state.items()}
            base.load_state_dict(state, strict=True)
        from models.feature_graph import FeatureGraphRefiner
        teacher = None
        if args.get('USE_TEACHER', True):
            teacher = RoboFireFuseNet(**base_options)
            teacher.load_state_dict(base.state_dict(), strict=True)
        model = FeatureGraphRefiner(
            base, backbone_mode=args.get('BACKBONE_MODE', 'staged'),
            unfreeze_epoch=args.get('UNFREEZE_EPOCH', 3), teacher=teacher,
            num_classes=args['NUM_CLASSES'], patch_size=args.get('GRAPH_PATCH_SIZE', 8),
            hidden_dim=args.get('GRAPH_HIDDEN_DIM', 128), heads=args.get('GRAPH_HEADS', 4),
            layers=args.get('GRAPH_LAYERS', 2), nonlocal_k=args.get('GRAPH_NONLOCAL_K', 4),
            dropout=args.get('GRAPH_DROPOUT', 0.1),
            knn_chunk_size=args.get('GRAPH_KNN_CHUNK_SIZE', 256),
            refiner_kind=args.get('REFINER_KIND', 'graph'),
            use_evidence=args.get('GRAPH_USE_EVIDENCE', True),
            use_predictions=args.get('GRAPH_USE_PREDICTIONS', True),
            use_prior=args.get('GRAPH_USE_PRIOR', True),
            use_edge_attr=args.get('GRAPH_USE_EDGE_ATTR', True),
            spatial=args.get('GRAPH_SPATIAL', True),
            zero_init=args.get('ZERO_INIT_ADAPTER', True))
    else:
        raise ValueError('Unsupported MODEL: ' + args['MODEL'])
    if not args['MODEL'].startswith('robofire_graph') and args['PRETRAINED'] is not None:
        model.imgnet_pretrain(args['PRETRAINED'])
    model.to(device=args['DEVICE'])
    return model
