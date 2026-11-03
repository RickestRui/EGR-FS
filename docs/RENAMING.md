# EGR-FS names and existing experiments

The public project name is EGR-FS, and new experiment IDs use `egr_fs`.
New protocol roots are `experiments/egr_fs_4090` and `experiments/egr_fs_suite`.
The model implementation key `robofire_graph`, Python classes, state-dictionary
keys, graph structure, losses, and tensor shapes remain unchanged by the rename.

The legacy `v4` experiment ID is accepted when reading historical configurations
and results. Downloaded weights and original experiment records retain their
`v4_4090` paths. New protocols generate `egr_fs`; aggregation displays the historical
main method as `egr_fs` without editing raw results.

Source text changes alter provenance hashes. Historical test locks and resume
checks may reject the current release. Use the corresponding original source
snapshot for strict reproduction, without rewriting integrity records. The cloud
packages contain checkpoint configurations and experiment records; they do not
include every historical source snapshot or resume checkpoint.

See [downloads and historical paths](DOWNLOADS.md) and
[new experiment instructions](../experiments/README.md).
