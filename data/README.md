# Prepared FLAME2 data

Download `01_FLAME2_Data.zip` from [Baidu Netdisk](https://pan.baidu.com/s/1jlV0lvsCkMfYwScKsLKwTQ?pwd=5qj2) (access code: **5qj2**)
and extract it into the repository root. This code package already includes the
same split lists; the archive supplies the images and segmentation masks.

The dataset contains 992 triplets: 694 training, 149 validation, and 149 test samples.
For the list entry `FLAME2/images/img_XXX_(116).png`, the loader expects:

```text
data/FLAME2/images/img_rgb_(116).png
data/FLAME2/images/img_ir_(116).png
data/FLAME2/images/img_gt_(116).png
```

RGB has three channels and IR has one. With `LABEL_ENCODING: palette`, the mask
colors are background `(0,0,0)`, smoke `(125,125,125)`, and fire `(255,255,255)`.
Keep these values unchanged; display colormaps are not valid training labels.

`ROOTDATASET` defaults to `./data`. Split-list paths are relative to this directory.
For a new experiment protocol, run its data audit before training. If sequence or
scene identifiers are available, supply a `sample_id,group_id` file to the audit
to check group overlap as well.

The images and prepared annotations originate from FLAME2 and the RoboFireFuseNet
data preparation. Retain their attribution and applicable licenses.
[Original FLAME2 dataset](https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset).
See the [download guide](../docs/DOWNLOADS.md) for the complete package list.
