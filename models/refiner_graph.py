"""Full-coverage region graph and pixel residual refinement for RFFNet.

The sparse GATv2 operator is implemented with native PyTorch scatter operations;
no PyG or compiled graph extensions are needed. Graph topology uses predictions
and features only, never ground-truth labels.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RegionGraphBuilder(nn.Module):
    """Eight-neighbour grid + feature kNN + self edges, with batch isolation.

    Edge features: dx, dy, distance, cosine similarity, C probability differences,
    boundary proxy and a three-way edge type (spatial, nonlocal, self).
    The boundary proxy is the maximum pooled endpoint boundary response, not a
    calibrated physical barrier. Attention learns its effect for each edge type.
    """
    def __init__(self, nonlocal_k=4, knn_chunk_size=256):
        super().__init__()
        if nonlocal_k < 0 or knn_chunk_size <= 0:
            raise ValueError('nonlocal_k must be >= 0 and knn_chunk_size > 0')
        self.nonlocal_k = nonlocal_k
        self.knn_chunk_size = knn_chunk_size
        self.spatial = True

    def forward(self, descriptors, probabilities, boundaries, grid_size):
        batch, nodes, _ = descriptors.shape
        gh, gw = grid_size
        if nodes != gh * gw:
            raise ValueError('Node count does not match grid size')
        device = descriptors.device
        ids = torch.arange(nodes, device=device).reshape(gh, gw)
        yy, xx = torch.meshgrid(torch.arange(gh, device=device),
                                torch.arange(gw, device=device), indexing='ij')
        cells = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
        coordinates = (cells.to(descriptors.dtype) + 0.5) / descriptors.new_tensor([gw, gh])
        local_source, local_target = [], []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                y0, y1 = max(0, -dy), min(gh, gh - dy)
                x0, x1 = max(0, -dx), min(gw, gw - dx)
                local_target.append(ids[y0:y1, x0:x1].flatten())
                local_source.append(ids[y0+dy:y1+dy, x0+dx:x1+dx].flatten())
        local_source = torch.cat(local_source)
        local_target = torch.cat(local_target)
        if not self.spatial:
            local_source, local_target = local_source[:0], local_target[:0]
        offsets = torch.arange(batch, device=device).view(-1, 1) * nodes
        source = [(local_source[None] + offsets).flatten()]
        target = [(local_target[None] + offsets).flatten()]
        types = [torch.zeros(source[0].numel(), device=device, dtype=torch.long)]
        normalized = F.normalize(descriptors, dim=-1, eps=1e-6)
        if self.nonlocal_k:
            # Discrete topology is detached; selected node/edge features still
            # receive gradients. Chunking bounds the N x N query memory.
            with torch.no_grad():
                search = normalized.detach()
                for start in range(0, nodes, self.knn_chunk_size):
                    end = min(start + self.knn_chunk_size, nodes)
                    similarities = torch.bmm(search[:, start:end], search.transpose(1, 2))
                    nearby = (cells[start:end, None] - cells[None]).abs().amax(-1) <= 1
                    similarities.masked_fill_(nearby[None], float('-inf'))
                    k = min(self.nonlocal_k, nodes)
                    values, neighbours = similarities.topk(k, dim=-1)
                    query = torch.arange(start, end, device=device)[None, :, None]
                    query = query.expand(batch, -1, k)
                    valid = torch.isfinite(values)
                    offset = offsets[:, :, None]
                    source.append((neighbours + offset)[valid])
                    target.append((query + offset)[valid])
                    types.append(torch.ones_like(source[-1]))
        self_ids = torch.arange(batch * nodes, device=device)
        source.append(self_ids)
        target.append(self_ids)
        types.append(torch.full_like(self_ids, 2))
        src, dst, edge_type = torch.cat(source), torch.cat(target), torch.cat(types)
        pos = coordinates.repeat(batch, 1)
        delta = pos[src] - pos[dst]
        norm = normalized.reshape(batch * nodes, -1)
        cosine = (norm[src] * norm[dst]).sum(-1, keepdim=True)
        probs = probabilities.reshape(batch * nodes, -1)
        boundary = boundaries.reshape(batch * nodes, 1)
        barrier = torch.maximum(boundary[src], boundary[dst])
        barrier = torch.where((edge_type == 2)[:, None], torch.zeros_like(barrier), barrier)
        edge_attr = torch.cat((delta, delta.norm(dim=-1, keepdim=True), cosine,
                               (probs[src] - probs[dst]).abs(), barrier,
                               F.one_hot(edge_type, 3).to(descriptors.dtype)), dim=-1)
        return torch.stack((src, dst)), edge_attr, coordinates


class SparseGATv2(nn.Module):
    """Multi-head dynamic attention, normalized over incoming edges per node."""
    def __init__(self, hidden_dim=128, heads=4, edge_dim=11, dropout=0.1):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError('hidden_dim must be divisible by heads')
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.source = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.target = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.attention = nn.Parameter(torch.empty(heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.dropout = dropout
        nn.init.xavier_uniform_(self.attention)

    def forward(self, nodes, edge_index, edge_attr, log_prior=None):
        src, dst = edge_index
        n = nodes.shape[0]
        source = self.source(nodes).reshape(n, self.heads, self.head_dim)
        target = self.target(nodes).reshape(n, self.heads, self.head_dim)
        edges = self.edge(edge_attr).reshape(-1, self.heads, self.head_dim)
        scores = (F.leaky_relu(source[src] + target[dst] + edges, 0.2)
                  * self.attention).sum(-1)
        # Float32 segment softmax also works under mixed precision.
        scores = scores.float()
        if log_prior is not None:
            scores = scores + log_prior.float()[:, None]
        index = dst[:, None].expand(-1, self.heads)
        maxima = scores.new_full((n, self.heads), float('-inf'))
        maxima.scatter_reduce_(0, index, scores.detach(), reduce='amax', include_self=True)
        exp = (scores - maxima[dst]).exp()
        sums = exp.new_zeros(n, self.heads).index_add_(0, dst, exp)
        weights = exp / sums[dst].clamp_min(1e-12)
        weights = F.dropout(weights, self.dropout, self.training).to(source.dtype)
        messages = source[src] * weights[..., None]
        result = source.new_zeros(n, self.heads, self.head_dim).index_add_(0, dst, messages)
        return result.reshape(n, -1) + self.bias


class GraphBlock(nn.Module):
    def __init__(self, hidden_dim, heads, edge_dim, dropout):
        super().__init__()
        self.attention = SparseGATv2(hidden_dim, heads, edge_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes, edges, attributes, log_prior=None):
        return self.norm(nodes + F.silu(self.attention(nodes, edges, attributes, log_prior)))


class GraphRefinementHead(nn.Module):
    def __init__(self, fusion_channels, modality_channels, num_classes=3,
                 patch_size=8, hidden_dim=128, heads=4, layers=2,
                 nonlocal_k=4, dropout=0.1, knn_chunk_size=256):
        super().__init__()
        if patch_size <= 0 or layers <= 0 or hidden_dim <= 0 or heads <= 0:
            raise ValueError('Graph sizes, heads and layer count must be positive')
        if num_classes < 2 or not 0 <= dropout < 1:
            raise ValueError('At least two classes and dropout in [0, 1) are required')
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.fusion_projection = nn.Conv2d(fusion_channels, 32, 1)
        self.rgb_projection = nn.Conv2d(modality_channels, 16, 1)
        self.ir_projection = nn.Conv2d(modality_channels, 16, 1)
        # Mean/max fused features, modality means, probability mean/max,
        # entropy mean/max, boundary max, and normalized coordinates.
        node_dim = 64 + 32 + 2 * num_classes + 2 + 1 + 2
        self.embedding = nn.Sequential(nn.Linear(node_dim, hidden_dim),
                                       nn.LayerNorm(hidden_dim), nn.SiLU())
        self.builder = RegionGraphBuilder(nonlocal_k, knn_chunk_size)
        self.layers = nn.ModuleList(GraphBlock(hidden_dim, heads, num_classes + 8, dropout)
                                    for _ in range(layers))
