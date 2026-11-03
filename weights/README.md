# Checkpoints

Download checkpoints from [Baidu Netdisk](https://pan.baidu.com/s/1jlV0lvsCkMfYwScKsLKwTQ?pwd=5qj2) (access code: **5qj2**).
Extract the selected ZIP files into the repository root, keeping their paths intact.

## Training from the supplied initializer

Package `02_Initialization.zip` provides
`weights/pretrained_480x640_w8_2_6.pth`, the default stage-one initializer.
The protocol records it as an existing nine-class initializer. Its filename does
not establish its training data; do not describe it as ImageNet-only without
verified provenance.

The full workflow first trains RFFNet, then initializes EGR-FS with the three-class
RFFNet checkpoint from the same random seed. New protocols generate this dependency
under `weights/experiments/egr_fs_4090/rffnet_s<seed>/best.pth`.

## Released models

- `03_Main_Models.zip`: EGR-FS, RFFNet, and the fine-tuning control for seeds 200, 201, and 202.
- `04_Optional_Ablation.zip`: 33 ablation checkpoints and their configurations.
- `05_Optional_Sensitivity.zip`: eight sensitivity-study checkpoints and their configurations.

These packages contain `best.pth` files and retain the original
`weights/experiments/v4_4090/<experiment>_s<seed>/` paths. EGR-FS checkpoints are in
`v4_s200`, `v4_s201`, and `v4_s202`. Corresponding configurations are included under
`experiments/v4_4090/configs/`. The three EGR-FS seeds are all provided; none was
selected for release based on its test score. `latest.pth` files are not distributed.

Use the matching configuration and set `CHECKPOINT` to the complete model file.
See [the evaluation example](../docs/DOWNLOADS.md#evaluate-a-released-egr-fs-checkpoint).

## Optional standalone initialization

Package `06_Optional_Standalone_RFFNet.zip` supplies `weights/robo_fire_best.pth`,
which is the default `PRETRAINED` path in `config/wildfire_graph.yaml`. This is an
RFFNet initializer, not a trained EGR-FS model. The full two-stage workflow creates
its own seed-matched initializers and does not require package 06.

The optional PIDNet baseline checkpoint `PIDNet_S_ImageNet.pth.tar` is not included
in the shared folder. It is not needed for the EGR-FS main experiments.
