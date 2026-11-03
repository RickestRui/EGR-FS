# Data and pretrained models

Download the release files from [Baidu Netdisk](https://pan.baidu.com/s/1jlV0lvsCkMfYwScKsLKwTQ?pwd=5qj2).

- Shared folder: `EGR-FS-Prepared`
- Access code: `5qj2`

## Choose your downloads

| Archive | Contents | When to download |
| --- | --- | --- |
| `01_FLAME2_Data.zip` | 992 RGB/IR/mask triplets and the train/validation/test lists | Training or evaluation |
| `02_Initialization.zip` | Stage-one initialization checkpoint | Full training workflow |
| `03_Main_Models.zip` | Nine best checkpoints with configurations: EGR-FS, RFFNet, and fine-tuning control, each with seeds 200/201/202 | Evaluation of the main models |
| `04_Optional_Ablation.zip` | 33 best checkpoints and configurations | Ablation evaluation |
| `05_Optional_Sensitivity.zip` | Eight best checkpoints and configurations | Validation-based sensitivity studies |
| `06_Optional_Standalone_RFFNet.zip` | Original `robo_fire_best.pth` | Standalone stage-two training with the default `wildfire_graph.yaml` |
| `07_Experiment_Records.zip` | Original experiment manifest, protocol, test locks, and summary tables | Reference experiment records |

The full two-stage training workflow needs packages **01 and 02**. It trains the
seed-matched RFFNet checkpoints used to initialize stage two. To evaluate the
released main models, download **01 and 03**. Add **04** or **05** for the corresponding
variants. Package **06** is optional and does not contain an EGR-FS checkpoint.

## Extract into the repository root

The archives contain `data/`, `weights/`, and/or `experiments/` directly. Merge
these directories into the repository root using an extraction tool with ZIP64 support.
Keep the filenames and internal paths unchanged. For example, extracting 01 and 03 gives:

```text
EGR-FS/
  data/
    FLAME2/images/
    lists/train_flm.txt
    lists/val_flm.txt
    lists/test_flm.txt
  weights/experiments/v4_4090/
    v4_s200/best.pth
    v4_s201/best.pth
    v4_s202/best.pth
    rffnet_s200/best.pth
    ...
  experiments/v4_4090/configs/
    v4_s200.yaml
    v4_s201.yaml
    v4_s202.yaml
    ...
```

## Evaluate a released EGR-FS checkpoint

Use the configuration supplied with that checkpoint, so evaluation settings match
the original experiment. From the repository root on Linux, after extracting 01 and 03:

```bash
cp experiments/v4_4090/configs/v4_s200.yaml config/egr_fs_s200_eval.yaml
python evaluate_graph.py --yaml_file egr_fs_s200_eval.yaml \
  --CHECKPOINT weights/experiments/v4_4090/v4_s200/best.pth \
  --SPLIT test --DEVICE cuda:0
```

For seeds 201 and 202, use the matching configuration and checkpoint. On Windows,
copy the configuration into `config/` and run the Python command on a single line.
Keep the supplied architecture and preprocessing settings; adjust data paths and
the device only if needed. A complete graph checkpoint includes the backbone,
so this evaluation command does not require package 02 or 06.

This standalone evaluator reports its own metrics. It does not rerun the complete
paper protocol, audit the original source snapshot, or reproduce every metric in
the archived tables. Use the experiment workflow and its integrity checks for a
new controlled study.

## Historical identifiers and new experiments

The released checkpoints come from completed experiments whose paths contain
`v4_4090` and `v4_s200/201/202`. Their public model name is **EGR-FS**. These paths
are retained to match the original checkpoint metadata and configurations.
New experiments use `experiments/egr_fs_4090` and `egr_fs_s200/201/202`.

Do not run `prepare` over the extracted historical protocol or mix its records
with a new run. Follow [the experiment instructions](../experiments/README.md)
to create a new protocol. Changes to source files, including the project rename,
change source hashes and can fail the strict checks for historical locked runs.
Do not rewrite test locks to suppress these failures; exact reproduction of a
locked run requires its matching source snapshot.

The optional PIDNet baseline initializer `PIDNet_S_ImageNet.pth.tar` is not included
in the shared folder and is not required by the EGR-FS main workflow.
