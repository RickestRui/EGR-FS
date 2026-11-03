"""GNN-conditioned decoder features, classified by RFFNet's pretrained head.

No entropy gate or bounded logit residual. The zero-initialized feature adapter
preserves the initial model; region supervision trains GNN from the first step.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .evidence_graph import EvidenceGraphHead


class FeatureGraphHead(EvidenceGraphHead):
    def __init__(self, fusion_channels, modality_channels, **options):
        zero_init = options.pop('zero_init', True)
        super().__init__(fusion_channels, modality_channels, **options)
        hidden = options.get('hidden_dim', 128)
        self.context_projection = nn.Conv2d(hidden, 64, 1)
        self.adapter = nn.Sequential(
            nn.Conv2d(64 + 32 + 8, 64, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1, groups=64), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, fusion_channels, 1))
        if zero_init:
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, coarse_logits, boundary_logits, features, return_details=False):
        encoded = self.encode_graph(coarse_logits, boundary_logits, features)
        # Condition full-resolution decoder features on graph context before
        # classification. Logit corrections have no hard amplitude ceiling.
        context = F.interpolate(self.context_projection(encoded['grid_features']),
                                features['fusion'].shape[-2:], mode='bilinear', align_corners=False)
        delta = self.adapter(torch.cat((context, encoded['fused'], encoded['evidence']), 1))
        result = dict(refined_features=features['fusion'] + delta,
                      node_logits=encoded['node_logits'], feature_delta=delta)
        if return_details:
            result.update(edge_index=encoded['edges'], edge_attr=encoded['attributes'],
                          grid_size=encoded['grid'], node_features=encoded['nodes'])
        return result


class FeatureGraphRefiner(nn.Module):
    DECODER = ('tf_conv0', 'tf_conv1', 'tf_conv2', 'tf_conv3', 'dfm', 'final_layer', 'seghead_d')
    # stage4 is a separately registered Swin module invoked from a LambdaLayer.
    LATE = ('stage4', 'layer4', 'layer5', 'layer4_rgb', 'layer5_rgb', 'layer4_ir', 'layer5_ir',
            'weight_channels_post', 'spp', 'layer4_', 'layer5_', 'layer4_d', 'layer5_d',
            'compression4', 'pag4', 'diff4')

    def __init__(self, backbone, teacher=None, unfreeze_epoch=3, backbone_mode='staged', **options):
        super().__init__()
        if backbone_mode not in ('staged', 'decoder', 'frozen') or unfreeze_epoch < 0:
            raise ValueError('Graph refinement requires BACKBONE_MODE staged/decoder/frozen and UNFREEZE_EPOCH >= 0')
        self.backbone, self.teacher = backbone, teacher
        self.backbone_mode = backbone_mode
        self.unfreeze_epoch = unfreeze_epoch
        self.freeze_backbone = backbone_mode == 'frozen'
        self.refiner_kind = options.pop('refiner_kind', 'graph')
        if self.refiner_kind not in ('graph', 'none'):
            raise ValueError('REFINER_KIND must be graph or none')
        self.refiner = (FeatureGraphHead(backbone.planes * 4, backbone.planes, **options)
                        if self.refiner_kind == 'graph' else None)
        if teacher is not None:
            teacher.requires_grad_(False).eval()
        self.set_epoch(0)
        self.backbone.eval()

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        for name, param in self.backbone.named_parameters():
            prefix = name.split('.')[0]
            active = self.backbone_mode != 'frozen' and (prefix in self.DECODER or
                     (self.backbone_mode == 'staged' and epoch >= self.unfreeze_epoch and prefix in self.LATE))
            param.requires_grad_(active)

    def parameter_groups(self, decoder_factor, late_factor, feature_factor=1.):
        groups = [] if self.refiner is None else [{'params': [p for n, p in self.refiner.named_parameters() if not n.startswith('adapter.')],
                   'lr_scale': 1.},
                  {'params': list(self.refiner.adapter.parameters()), 'lr_scale': feature_factor}]
        # Include future trainable parameters now: stable optimizer layout on resume.
        if self.backbone_mode != 'frozen':
            groups.append({'params': [p for n, p in self.backbone.named_parameters() if n.split('.')[0] in self.DECODER],
                           'lr_scale': decoder_factor})
        if self.backbone_mode == 'staged':
            groups.append({'params': [p for n, p in self.backbone.named_parameters() if n.split('.')[0] in self.LATE],
                           'lr_scale': late_factor})
        return [g for g in groups if g['params']]

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # preserve BN statistics and disable path dropout
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def forward(self, images, return_details=False):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_backbone):
            features = self.backbone(images, return_features=True)
        coarse = features['outputs'][1]
        boundary = F.interpolate(features['outputs'][2], coarse.shape[-2:], mode='bilinear', align_corners=False)
        features['images'] = images
        if self.refiner is None:
            result, refined = {}, coarse
        else:
            result = self.refiner(coarse, boundary, features, return_details)
            refined = self.backbone.final_layer(result.pop('refined_features'))
        result.update(refined_logits=refined, coarse_logits=coarse, boundary_logits=boundary,
                      coarse_boundary_logits=boundary, backbone_frozen=self.freeze_backbone,
                      correction=refined - coarse)
        if self.training and self.teacher is not None:
            with torch.no_grad():
                result['teacher_logits'] = self.teacher(images)[1]
        return result
