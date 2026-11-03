import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

class CrossEntropy(nn.Module):
    def __init__(self, args, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.Tensor(weight).to(args['DEVICE']),
            ignore_index=ignore_label
        )
        self.num_outputs = args['NUM_OUTPUTS']
        self.balance_weights = args['BALANCE_WEIGHTS']
        self.sb_weights = args['SB_WEIGHTS']

    def _forward(self, score, target):
        loss = self.criterion(score, target)
        return loss

    def forward(self, score, target):
        if not (isinstance(score, list) or isinstance(score, tuple)):
            score = [score]
        balance_weights = self.balance_weights
        if len(balance_weights) == len(score):
            return sum([w * self._forward(x, target) for (w, x) in zip(balance_weights, score)])
        elif len(score) == 1:
            return self.sb_weights * self._forward(score[0], target)
        else:
            raise ValueError("lengths of prediction and target are not identical!")

class LovaszCE:
    def __init__(self, cls_weights):
        self.cls_weights = cls_weights

    def lovasz_softmax_loss(self, outputs, labels, classes='all', per_image=False, ignore=None):
        """
        Multi-class Lovasz-Softmax loss
        probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        per_image: compute the loss per image instead of per batch
        ignore: void class labels
        """
        probas = F.softmax(outputs, dim=1)
        if per_image:
            loss = self.mean(self.lovasz_softmax_flat(*(self.flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore)), classes=classes)
                            for prob, lab in zip(probas, labels))
        else:
            loss = self.lovasz_softmax_flat(*(self.flatten_probas(probas, labels, ignore)), classes=classes)
        return loss


    def lovasz_softmax_flat(self, probas, labels, classes='all'):
        """
        Multi-class Lovasz-Softmax loss
        probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
        labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas.sum() * 0.
        C = probas.size(1)
        losses = []
        class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
        for c in class_to_sum:
            fg = (labels == c).float() # foreground for class c
            if (classes == 'present' and fg.sum() == 0):
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError('Sigmoid output possible only with 1 class')
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, self.lovasz_grad(fg_sorted)) * self.cls_weights[c])
        return self.mean(losses)

    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        if probas.dim() == 3:
            # assumes output of a sigmoid layer
            B, H, W = probas.size()
            probas = probas.view(B, 1, H, W)

        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        labels = labels.reshape(-1)
        if ignore is None:
            return probas, labels
        valid = (labels != ignore)
        vprobas = probas[valid]
        vlabels = labels[valid]
        return vprobas, vlabels

    def isnan(self, x):
        return x != x

    def mean(self, l, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """
        l = iter(l)
        if ignore_nan:
            l = ifilterfalse(self.isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n
    
    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1: # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

class OhemCrossEntropy(nn.Module):
    """
    This is an implementation of Online Hard Example Mining and 
    is alternative of CE that can propably increase convergence 
    speed.
    """
    def __init__(self, args, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.Tensor(weight).to(args['DEVICE']),
            ignore_index=ignore_label,
            reduction='none'
        )
        self.num_outputs = args['NUM_OUTPUTS']
        self.balance_weights = args['BALANCE_WEIGHTS']
        self.sb_weights = args['SB_WEIGHTS']

    def _ce_forward(self, score, target):
        loss = self.criterion(score, target)
        return loss

    def _ohem_forward(self, score, target, **kwargs):
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label
        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        if pred.numel() == 0:   return torch.tensor(0.0)
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)
        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):
        if not (isinstance(score, list) or isinstance(score, tuple)):
            score = [score]
        balance_weights = self.balance_weights
        sb_weights = self.sb_weights
        if len(balance_weights) == len(score):
            functions = [self._ce_forward] * \
                (len(balance_weights) - 1) + [self._ohem_forward]
            return sum([
                w * func(x, target)
                for (w, x, func) in zip(balance_weights, score, functions)
            ])
        elif len(score) == 1:
            return sb_weights * self._ohem_forward(score[0], target)
        else:
            raise ValueError("lengths of prediction and target are not identical!")

class BondaryLoss(nn.Module):
    """
    Binary cross-entropy, evaluating weights for positives and negatives per batch.
    """
    def __init__(self, coeff_bce = 20.0):
        super(BondaryLoss, self).__init__()
        self.coeff_bce = coeff_bce

    def weighted_bce(self, bd_pre, target):
        n, c, h, w = bd_pre.size()
        log_p = bd_pre.permute(0,2,3,1).contiguous().view(1, -1)
        target_t = target.view(1, -1)
        pos_index = (target_t == 1)
        neg_index = (target_t == 0)
        weight = torch.zeros_like(log_p).float()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num.float() + neg_num
        weight[pos_index] = neg_num.float() * 1.0 / sum_num
        weight[neg_index] = pos_num.float() * 1.0 / sum_num
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight.float(), reduction='mean')
        return loss

    def forward(self, bd_pre, bd_gt):
        bce_loss = self.coeff_bce * self.weighted_bce(bd_pre, bd_gt)
        loss = bce_loss
        return loss

class TotalLoss:
    def __init__(self, args):
        self.align_corners = args['ALIGN_CORNERS']
        self.ignore_label = args['IGNORE_LABEL']
        self.t_thresh_bd = args['T_THRESH_BDLOSS']
        self.n_classes = args['NUM_CLASSES']
        self.bd_weight = args['BD_WEIGHT']
        self.class_weights = args['CLASS_WEIGHTS']
        self.defuse_weights = [1, 0.5, 0.5]
        self.miou_ce = LovaszCE(self.class_weights)
        self.mse_loss = nn.MSELoss()
        self.loss_params = torch.nn.SmoothL1Loss()
    
        if args['USE_OHEM']:
            self.sem_criterion = OhemCrossEntropy(args, ignore_label=args['IGNORE_LABEL'],
                                        thres=args['OHEMTHRES'],
                                        min_kept=args['OHEMKEEP'],
                                        weight=args['CLASS_WEIGHTS'])
        else:
            self.sem_criterion = CrossEntropy(args, ignore_label=args['IGNORE_LABEL'],
                                    weight=args['CLASS_WEIGHTS'])
        self.bd_criterion = BondaryLoss(coeff_bce=self.bd_weight)
         
    def pixel_acc(self, pred, label):
        """
        Calculates the mean pixel accuracy for valid pixels (non-negative labels) across the batch.
        """
        _, preds = torch.max(pred, dim=1)
        valid = (label != self.ignore_label).long()
        pixel_sum = torch.sum(valid)
        acc_sum = torch.sum(valid * (preds == label).long())
        acc = (acc_sum.float() / (pixel_sum.float() + 1e-10)).detach().cpu().numpy()
        acc_per_class = [0] * len(self.class_weights)
        for i, cls in enumerate(self.class_weights):
            cls_vld = (label == i).long()
            pixel_sum = torch.sum(cls_vld)
            acc_sum = torch.sum(cls_vld * (preds == label).long())
            acc_per_class[i] = (acc_sum.float() / (pixel_sum.float() + 1e-10)).detach().cpu().numpy()
        acc = np.array([acc, *acc_per_class])
        return acc

    def get_loss(self, outputs, labels, bd_gt):
        """
        Calculates total prediction loss, including semantic loss (using OHEM or cross-entropy), 
        boundary loss (using CE), and additional semantic loss for detected boundaries.
        :return: the loss, the semantic maps (from both propotion and integral head),
        the avg pixel accuracy and the segmentation and boundary losses.
        """
        loss = torch.tensor(0).to(labels.device).to(torch.float)
        h, w = labels.size(1), labels.size(2)
        ph, pw = outputs[0].size(2), outputs[0].size(3)
        if (ph != h) or (pw != w):
            for i in range(len(outputs)):
                outputs[i] = F.interpolate(outputs[i], size=(
                    h, w), mode='bilinear', align_corners=self.align_corners)
        acc  = self.pixel_acc(outputs[-2], labels)
        loss_s = self.miou_ce.lovasz_softmax_loss(outputs[1], labels, classes='all', per_image=False) + \
            self.miou_ce.lovasz_softmax_loss(outputs[0], labels, classes='all', per_image=False) 
        loss_b = self.bd_criterion(outputs[-1], bd_gt)      # the boundary loss l1
        filler = torch.ones_like(labels) * self.ignore_label
        bd_label = torch.where(F.sigmoid(outputs[-1][:,0,:,:])>self.t_thresh_bd, labels, filler)
        loss_sb = self.sem_criterion(outputs[-2], bd_label)
        loss += (loss_s if not torch.isnan(loss_sb) else 0)  + (loss_b if not torch.isnan(loss_b) else 0) + (loss_sb if not torch.isnan(loss_sb) else 0)
        return torch.unsqueeze(loss,0), outputs[:-1], acc, [loss_s, loss_b]


class RefinementLoss(TotalLoss):
    """Pixel CE + Lovasz + supervised boundary residual for graph refinement."""
    def __init__(self, args):
        super().__init__(args)
        weights = torch.as_tensor(args['CLASS_WEIGHTS'], dtype=torch.float,
                                  device=args['DEVICE'])
        self.weights = weights / weights.mean()
        self.lovasz = LovaszCE([1.0] * self.n_classes)
        self.ce_weight = args.get('REFINE_CE_WEIGHT', 1.0)
        self.lovasz_weight = args.get('REFINE_LOVASZ_WEIGHT', 1.0)
        self.boundary_weight = args.get('REFINE_BOUNDARY_WEIGHT', 0.2)
        self.coarse_weight = args.get('REFINE_COARSE_WEIGHT', 0.4)

    def semantic_loss(self, logits, labels):
        if not (labels != self.ignore_label).any():
            return logits.sum() * 0.
        ce = F.cross_entropy(logits, labels, weight=self.weights,
                             ignore_index=self.ignore_label)
        lovasz = self.lovasz.lovasz_softmax_loss(
            logits, labels, classes='present', ignore=self.ignore_label)
        return self.ce_weight * ce + self.lovasz_weight * lovasz

    def get_loss(self, outputs, labels, bd_gt):
        logits = outputs['refined_logits']
        boundary = outputs['boundary_logits'][:, 0]
        valid = labels != self.ignore_label
        semantic = self.semantic_loss(logits, labels)
        if valid.any():
            target = bd_gt.to(boundary.dtype)[valid]
            positives = target.sum()
            negatives = target.numel() - positives
            pos_weight = (negatives / positives.clamp_min(1)).clamp(1, 20)
            boundary_loss = F.binary_cross_entropy_with_logits(
                boundary[valid], target, pos_weight=pos_weight)
        else:
            boundary_loss = boundary.sum() * 0.
        loss = semantic + self.boundary_weight * boundary_loss
        if not outputs['backbone_frozen']:
            loss = loss + self.coarse_weight * self.semantic_loss(outputs['coarse_logits'], labels)
        accuracy = self.pixel_acc(logits, labels)
        return loss.unsqueeze(0), [outputs['coarse_logits'], logits], accuracy, [semantic, boundary_loss]


class EvidenceRefinementLoss(RefinementLoss):
    """Region soft targets, error-gate supervision and correct-pixel retention."""
    def __init__(self, args):
        super().__init__(args)
        self.node_weight = args.get('GRAPH_NODE_WEIGHT', .2)
        self.gate_weight = args.get('GRAPH_GATE_WEIGHT', .05)
        self.preserve_weight = args.get('GRAPH_PRESERVE_WEIGHT', .1)
        self.preserve_margin = args.get('GRAPH_PRESERVE_MARGIN', 1.)

    def get_loss(self, outputs, labels, bd_gt):
        loss, predictions, accuracy, components = super().get_loss(outputs, labels, bd_gt)
        valid = labels != self.ignore_label
        logits = outputs['refined_logits']
        nodes = outputs['node_logits']
        # Fractions preserve minority fire pixels inside mixed nodes; ignored
        # pixels have no mass. GT is used for this loss only, never graph creation.
        safe = labels.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe.long(), self.n_classes).permute(0, 3, 1, 2).to(logits.dtype)
        target_mass = F.adaptive_avg_pool2d(one_hot * valid[:, None], nodes.shape[-2:])
        occupancy = target_mass.sum(1)
        node_loss = -(target_mass * nodes.log_softmax(1)).sum() / occupancy.sum().clamp_min(1)
        coarse = outputs['coarse_logits'].detach()
        if valid.any():
            wrong = coarse.argmax(1) != labels
            target = wrong[valid].to(logits.dtype)
            pos_weight = ((1 - target).sum() / target.sum().clamp_min(1)).clamp(1, 5)
            gate_loss = F.binary_cross_entropy_with_logits(outputs['gate_logits'][:, 0][valid], target,
                                                           pos_weight=pos_weight)
            correct = valid & ~wrong
            kl = F.kl_div(logits.log_softmax(1), coarse.softmax(1), reduction='none').sum(1)
            if correct.any():
                # Retain a positive correct-class margin, not just similar
                # probabilities. Confident coarse logits need not be copied.
                class_mask = F.one_hot(safe.long(), self.n_classes).permute(0, 3, 1, 2).bool()
                coarse_margin = coarse.gather(1, safe[:, None]).squeeze(1) - coarse.masked_fill(class_mask, float('-inf')).amax(1)
                refined_margin = logits.gather(1, safe[:, None]).squeeze(1) - logits.masked_fill(class_mask, float('-inf')).amax(1)
                required = coarse_margin.clamp(max=self.preserve_margin)
                preservation = (F.relu(required - refined_margin) + .1 * kl)[correct].mean()
            else:
                preservation = logits.sum() * 0
        else:
            gate_loss = outputs['gate_logits'].sum() * 0
            preservation = logits.sum() * 0
        loss = loss + self.node_weight * node_loss + self.gate_weight * gate_loss + self.preserve_weight * preservation
        return loss, predictions, accuracy, components + [node_loss, gate_loss, preservation]


class FeatureGraphLoss(RefinementLoss):
    """Feature-GNN supervision; fixed-teacher retention never copies known errors."""
    def __init__(self, args):
        super().__init__(args)
        self.node_weight = args.get('GRAPH_NODE_WEIGHT', .1)
        self.teacher_weight = args.get('TEACHER_WEIGHT', .03)
        self.error_weight = args.get('TEACHER_ERROR_WEIGHT', .1)
        self.teacher_selection = args.get('TEACHER_SELECTION', 'correct')
        if self.teacher_selection not in ('correct', 'all'):
            raise ValueError('TEACHER_SELECTION must be correct or all')

    def get_loss(self, outputs, labels, bd_gt):
        loss, predictions, accuracy, components = super().get_loss(outputs, labels, bd_gt)
        valid = labels != self.ignore_label
        safe = labels.masked_fill(~valid, 0)
        node_loss = outputs['refined_logits'].sum() * 0
        if self.node_weight:
            nodes = outputs['node_logits']
            target = F.one_hot(safe.long(), self.n_classes).permute(0, 3, 1, 2).to(nodes.dtype)
            mass = F.adaptive_avg_pool2d(target * valid[:, None], nodes.shape[-2:])
            node_loss = -(mass * nodes.log_softmax(1)).sum() / mass.sum().clamp_min(1)
        logits = outputs['refined_logits']
        retain = hard_error = logits.sum() * 0
        if 'teacher_logits' in outputs:
            teacher = outputs['teacher_logits'].detach()
            correct = valid & (teacher.argmax(1) == labels)
            wrong = valid & ~correct
            retain_mask = valid if self.teacher_selection == 'all' else correct
            if retain_mask.any():
                kl = F.kl_div(logits.log_softmax(1), teacher.softmax(1), reduction='none').sum(1)
                retain = kl[retain_mask].mean()
            if wrong.any():
                per_pixel = F.cross_entropy(logits, labels, ignore_index=self.ignore_label, reduction='none')
                hard_error = per_pixel[wrong].mean()
        loss = loss + self.node_weight * node_loss + self.teacher_weight * retain + self.error_weight * hard_error
        return loss, predictions, accuracy, components + [node_loss, retain, hard_error]



