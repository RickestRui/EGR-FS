"""Evidence-aware GNN with supervised region predictions and bounded corrections.

Raw IR is image intensity, NOT calibrated temperature. No GT is used to build
features, edges, priors or gates. Labels supervise predictions only in the loss.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .refiner_graph import GraphRefinementHead, GraphBlock


def image_evidence(images):
    """RGB/IR, red excess, IR local contrast and modality gradient magnitudes."""
    rgb, ir = images[:, :3], images[:, 3:4]
    gray = rgb.mean(1, keepdim=True)

    def gradient(x):
        dx = F.pad((x[:, :, :, 1:] - x[:, :, :, :-1]).abs(), (0, 1, 0, 0))
        dy = F.pad((x[:, :, 1:, :] - x[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        return (dx + dy).clamp(0, 1)

    red = rgb[:, :1] - (rgb[:, 1:2] + rgb[:, 2:3]) * .5
    contrast = ir - F.avg_pool2d(F.pad(ir, (4, 4, 4, 4), mode='replicate'), 9, stride=1)
    return torch.cat((rgb, ir, red, contrast, gradient(gray), gradient(ir)), 1)


class EvidenceGraphHead(GraphRefinementHead):
    """Region encoder used by the RFFNet feature graph refiner."""
    def __init__(self, fusion_channels, modality_channels, **options):
        self.use_evidence = options.pop('use_evidence', True)
        self.use_predictions = options.pop('use_predictions', True)
        self.use_prior = options.pop('use_prior', True)
        self.use_edge_attr = options.pop('use_edge_attr', True)
        spatial = options.pop('spatial', True)
        super().__init__(fusion_channels, modality_channels, **options)
        self.builder.spatial = spatial
        hidden = options.get('hidden_dim', 128)
        self.semantic_projection = nn.Conv2d(fusion_channels, 16, 1)
        node_dim = self.embedding[0].in_features + 32  # evidence avg/max + semantic
        self.embedding = nn.Sequential(nn.Linear(node_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.layers = nn.ModuleList(GraphBlock(hidden, options.get('heads', 4),
                                              self.num_classes + 16, options.get('dropout', .1))
                                    for _ in range(options.get('layers', 2)))
        self.node_classifier = nn.Linear(hidden, self.num_classes)

    def build_graph(self, descriptors, probabilities, boundaries, grid, uncertainty):
        return self.builder(descriptors, probabilities, boundaries, grid)

    def encode_graph(self, coarse_logits, boundary_logits, features):
        batch, _, height, width = coarse_logits.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError('Image dimensions must be divisible by GRAPH_PATCH_SIZE')
        grid = height // self.patch_size, width // self.patch_size

        def pool(x, maximum=False):
            fn = F.adaptive_max_pool2d if maximum else F.adaptive_avg_pool2d
            return fn(x, grid).flatten(2).transpose(1, 2)

        fused = self.fusion_projection(features['fusion'])
        semantic = self.semantic_projection(features['semantic'])
        evidence = image_evidence(features['images'])
        if not self.use_evidence:
            evidence = torch.zeros_like(evidence)
        # Detached confidence prevents the decoder raising its entropy to open gates.
        probs = coarse_logits.detach().softmax(1)
        entropy = -(probs * probs.clamp_min(1e-7).log()).sum(1, keepdim=True) / torch.log(probs.new_tensor(self.num_classes))
        boundary = boundary_logits.detach().sigmoid()
        if not self.use_predictions:
            probs, entropy, boundary = torch.zeros_like(probs), torch.zeros_like(entropy), torch.zeros_like(boundary)
        node_probs, node_boundary, node_evidence = pool(probs), pool(boundary, True), pool(evidence)
        # kNN uses pretrained deep features plus measured appearance, not an
        # untrained projection. Selected edge features still carry gradients.
        descriptors = torch.cat((F.normalize(pool(features['semantic']), dim=-1), node_evidence), -1)
        edges, attributes, coords = self.build_graph(
            descriptors, node_probs, node_boundary, grid, pool(entropy, True))
        src, dst = edges
        raw = node_evidence.flatten(0, 1)
        differences = (raw[src] - raw[dst]).abs()
        attributes = torch.cat((attributes, differences), -1)
        # Soft evidence prior: never hard-delete cross-class corrections.
        log_prior = -(differences[:, :4].mean(-1) * 2 + differences[:, 6:].mean(-1))
        if not self.use_prior:
            log_prior = torch.zeros_like(log_prior)
        if not self.use_edge_attr:
            attributes = torch.zeros_like(attributes)
        pieces = [pool(fused), pool(fused, True), pool(self.rgb_projection(features['rgb'])),
                  pool(self.ir_projection(features['ir'])), node_probs, pool(probs, True),
                  pool(entropy), pool(entropy, True), node_boundary,
                  coords[None].expand(batch, -1, -1), node_evidence, pool(evidence, True), pool(semantic)]
        nodes = self.embedding(torch.cat(pieces, -1)).flatten(0, 1)
        for layer in self.layers:
            nodes = layer(nodes, edges, attributes, log_prior)
        grid_features = nodes.reshape(batch, *grid, -1).permute(0, 3, 1, 2).contiguous()
        node_logits = self.node_classifier(nodes).reshape(batch, *grid, -1).permute(0, 3, 1, 2)
        return dict(grid_features=grid_features, node_logits=node_logits, fused=fused,
                    evidence=evidence, probs=probs, boundary=boundary, entropy=entropy,
                    edges=edges, attributes=attributes, grid=grid, nodes=nodes)



