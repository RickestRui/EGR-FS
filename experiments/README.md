# EGR-FS experiment commands

Run from the repository root after installing dependencies and following the
[data and checkpoint download guide](../docs/DOWNLOADS.md). For a new training
protocol, download packages 01 and 02. The following commands generate new runs;
they are not commands for resuming the downloaded historical checkpoints.
Start a new protocol; do not copy old audit files, test locks or completion markers.
Packaging removes optional source files, so historical run hashes are not an exact
match to this release. Keep the original project for resuming existing runs.

```bash
export EXP=experiments/egr_fs_4090
python -m experiments.paper --root "$EXP" prepare
python -m experiments.paper --root "$EXP" audit
python -m experiments.cloud --root "$EXP" --phase preflight --gpus 0 1
python -m experiments.cloud --root "$EXP" --phase train --family main --gpus 0 1 --resume
python -m experiments.cloud --root "$EXP" --phase val --family main --gpus 0 1 --resume
python -m experiments.cloud --root "$EXP" --phase train --family ablation --gpus 0 1 --resume
python -m experiments.cloud --root "$EXP" --phase val --family ablation --gpus 0 1 --resume
python -m experiments.cloud --root "$EXP" --phase train --family sensitivity --gpus 0 1 --resume
python -m experiments.cloud --root "$EXP" --phase val --family sensitivity --gpus 0 1 --resume
```

Review validation results before locking the test checkpoints. Sensitivity stays
validation-only. For all main and ablation variants:

```bash
python -m experiments.paper --root "$EXP" lock --experiments rffnet finetune egr_fs self_only local_only no_evidence no_prior no_edge_attr no_predictions no_node_loss no_teacher teacher_all random_adapter frozen --seeds 200 201 202
python -m experiments.cloud --root "$EXP" --phase test --family confirmatory --gpus 0 1 --resume
python -m experiments.cloud --root "$EXP" --phase robustness --family confirmatory --gpus 0 1 --resume
python -m experiments.cloud --root "$EXP" --phase benchmark --family main --gpus 0 --resume
python -m experiments.paper --root "$EXP" aggregate
```

Run timing without other GPU workloads. Results are written as CSV under
`$EXP/results/`, timing files under `$EXP/benchmarks/`, and aggregation CSV files
under `$EXP/`. Training logs are under `outputs/$EXP/`; checkpoints under `weights/$EXP/`.

For individual runs, use `python -m experiments.paper --help`. For a queue preview,
append `--dry-run` to a cloud train/evaluation command (this writes a plan, not results).
The `--resume` queue flag checks completed jobs and resumes eligible incomplete training;
it is not permission to overwrite incomplete evaluation output directories.

The powershell helper `run_suite.ps1` is retained for Windows sequential execution.
Generated configs and CSV result files are intentionally not shipped in this package.
