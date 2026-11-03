# EGR-FS: Evidence-guided graph refinement for RGB–infrared fire and smoke segmentation

EGR-FS extends RoboFireFuseNet with evidence-aware region encoding, sparse graph
construction, two edge-aware GATv2 layers, and residual decoder-feature refinement.
This is the EGR-FS extension code, not the original authors' official RoboFireFuseNet repository.
The research model key remains `robofire_graph`.

## Included

- Training, validation, and evaluation entry points.
- Backbone, region graph, attention, adapter, and loss implementations.
- FLAME2 train/validation/test split lists (image identifiers only).
- Experiment generation, ablations, sensitivity, degradation tests, timing, and CSV aggregation.
- Engineering tests. No dataset images, masks, trained weights, or experimental result claims.

Run every command from this repository root. Some inherited imports use root-relative paths.

## Environment

Use Python 3.10. The project pins PyTorch 2.4.1 and torchvision 0.19.1.
For a headless Linux GPU server, install a compatible PyTorch/CUDA build first,
then the remaining pinned dependencies:

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1
python -m pip install -r experiments/requirements_cloud.txt
```

Select the PyTorch wheel source appropriate to the server's driver if necessary.
Alternatively, `python -m pip install -r requirements.txt` installs the original
dependency set including desktop OpenCV. Do not install both OpenCV variants in
the same environment. Confirm `torch.cuda.is_available()` before GPU training.

## Data and checkpoints

The prepared dataset and trained models are available in the
[EGR-FS-Prepared folder on Baidu Netdisk](https://pan.baidu.com/s/1jlV0lvsCkMfYwScKsLKwTQ?pwd=5qj2) (access code: **5qj2**).

- For full training, download `01_FLAME2_Data.zip` and `02_Initialization.zip`.
- To evaluate the released main models, download `01_FLAME2_Data.zip` and `03_Main_Models.zip`.
- Ablation and sensitivity checkpoints are provided separately in packages 04 and 05.

Extract the selected archives into this repository's root directory, merging
`data/`, `weights/`, and `experiments/` as needed. Use a tool that supports ZIP64.
See the [download guide](docs/DOWNLOADS.md) for all seven packages, the directory
layout, and a command for evaluating a released EGR-FS checkpoint.
The [data guide](data/README.md) explains the label format, and the
[checkpoint guide](weights/README.md) describes the training dependencies.

Released weights retain their historical `v4_4090` paths. New experiments use
`egr_fs_4090`. Keep the downloaded configurations with their matching checkpoints.

## Reproduce the experiment workflow

Configure data paths and the initialization checkpoint in `config/wildfire.yaml`
and `config/wildfire_graph.yaml` before generating a new protocol.

```bash
export EXP=experiments/egr_fs_4090
python -m experiments.paper --root "$EXP" prepare
python -m experiments.paper --root "$EXP" audit
python -m experiments.cloud --root "$EXP" --phase train --family main --gpus 0 1 --resume
python -m experiments.cloud --root "$EXP" --phase val --family main --gpus 0 1 --resume
```

`prepare` creates `manifest.csv` and 50 configurations: 9 main runs, 33 ablation
runs, and 8 sensitivity runs. Seeds are 200, 201, 202; sensitivity uses 200 only.
The generated protocol uses a maximum of 500 stage-one epochs and 100 stage-two
epochs, with early stopping. The standalone `wildfire.yaml` has a different
epoch default; use the generated protocol for this experiment schedule.

Two GPUs run two independent jobs, not DDP. Each job uses one card's memory.
The main queue completes the baseline runs before their dependent stage-two runs.
After a shell restart, set `EXP` again. `--root` must remain project-relative.
Do not omit `prepare`: the launcher requires `manifest.csv`.

See [complete experiment commands](experiments/README.md) for ablation, sensitivity,
checkpoint locking, test, degradation, timing, and result aggregation.

## Standalone training and evaluation

For standalone training, package 06 supplies the three-class RFFNet checkpoint
expected by `config/wildfire_graph.yaml`. The following evaluation command refers
to the checkpoint produced by this standalone run; for downloaded model weights,
use the command in the [download guide](docs/DOWNLOADS.md).

```bash
python train.py --yaml_file wildfire_graph.yaml --DEVICE cuda:0
python evaluate_graph.py --yaml_file wildfire_graph.yaml --CHECKPOINT weights/EGR-FS/egr_fs/best.pth --SPLIT val --DEVICE cuda:0
```

`evaluate_graph.py` is a standalone evaluator. For publication CSV outputs and
test-lock checks, use `experiments.paper` / `experiments.cloud`.

## Engineering checks

```bash
python -m unittest discover -s tests -p "test_*.py"
```

These tests use synthetic data and mock backbones where appropriate; their outputs
are not segmentation benchmark results. The release was checked for Python syntax,
copied-file integrity, packaging exclusions, and portable split lists. Full model
execution still requires the declared dependencies, data, and checkpoints.

## Architecture and attribution

See [EGR-FS architecture](docs/ARCHITECTURE.md).
The original MIT license and source-file attribution headers are preserved.
SwinV2-derived files retain Apache-2.0 headers and the corresponding license text.
No EGR-FS paper DOI, publication status, benchmark scores, or GitHub URL is fabricated here.

## Upstream sources

EGR-FS builds on RoboFireFuseNet (Fotiou, Mygdalis and Pitas),
DOI: 10.1016/j.patrec.2026.04.024. The inherited MIT LICENSE retains
Copyright (c) 2026 Fotiou Dimitrios. Upstream benchmark results are not EGR-FS results.
The SwinV2 source files retain Microsoft Research / Hugging Face and timm
attribution; the Apache-2.0 license is included in `licenses/Apache-2.0.txt`.
PIDNet retains Jiacong Xu's authorship notice (https://github.com/XuJiacong/PIDNet).
The dataset base class retains its HRNet attribution.
