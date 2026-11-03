# EGR-FS implementation map

The backbone supplies full-resolution decoder features F, shallow RGB/IR features,
semantic features S, coarse segmentation logits and boundary logits.

Eight-channel image evidence contains RGB, IR, red excess, local IR contrast, and
RGB/IR gradient magnitudes. Detached coarse logits produce class probabilities and
normalized entropy; detached boundary logits produce boundary probabilities.

Per-source projections and pooling align these inputs to an H/8 by W/8 region grid.
The 139-dimensional statistics (including normalized coordinates) are embedded into
128-dimensional nodes. Spatial eight-neighbor edges, nonlocal cosine top-4 edges
(excluding local 3x3), and self-loops form a directed sparse graph.
Each edge has 19 attributes and one evidence-derived log prior.

Two separate four-head GATv2 blocks reuse this graph structure and edge conditioning.
Their output is restored to the region grid, projected to 64 channels and upsampled.
It is concatenated with 32-channel projected F and eight-channel image evidence.
An adapter maps 104 channels to a 128-channel residual delta F; its last convolution
is zero-initialized. F + delta F is classified by the same segmentation head as F.

Code locations:
- `models/robofirefusenet.py`: backbone and feature taps.
- `models/refiner_graph.py`: sparse graph builder and attention operators.
- `models/evidence_graph.py`: image evidence, region encoding and edge prior.
- `models/feature_graph.py`: feature adapter, teacher wrapper and training scope.
- `utils/total_loss.py`: graph-refinement losses.
- `experiments/`: experiment definitions and publication evaluation workflow.

Ground-truth labels do not construct the inference graph. A teacher and node
classification supervision are training components. No hard entropy gate or bounded
logit residual is part of the default EGR-FS feature-refinement path.
