# CrystaNet: F0 vs. F1 Classification

This module contains the code used for the F0-versus-F1 crystal-image classification task. It includes internal five-fold training scripts and a training/external-evaluation workflow for the external cohort.

## Repository contents

```text
f0\_f1/
  scripts/
    cross\_validation/
      cv\_fold\_1.py
      cv\_fold\_2.py
      cv\_fold\_3.py
      cv\_fold\_4.py
      cv\_fold\_5.py
    external\_validation/
      train\_and\_validate.py
examples/
  guide\_synthetic\_example.xlsx
requirements.txt
environment.yml
```

## Software environment

The Conda environment and package versions are recorded in `environment.yml`. Create the environment with:

```bash
conda env create -f environment.yml
```

The environment file reflects the original Windows/CUDA software environment. Platform-specific dependencies may need adjustment on other operating systems.

## Data organization

By default, scripts look for data under the repository-level `data/` directory. Set `CRYSTANET\_DATA\_ROOT` to use a different data root. The following layout is expected:

```text
data/
  internal/
    f0\_images/
    f1\_images/
  zengcheng/
    f0\_images/
    f1\_images/
  guides/
    f0\_image\_list.xlsx
    liver\_fibrosis\_list.xlsx
    External validation\_liver\_fibrosis\_list.xlsx
```

The internal cross-validation scripts use the internal image folders and guide spreadsheets. The external script uses the internal data for training and the external data for external evaluation. 

Guide spreadsheet format

The guide spreadsheet is expected to contain these columns in order:

1. `filename`: image filename used to match the image in the corresponding directory.
2. `number`: participant identifier used to keep images from the same participant in the same split.
3. `fibrosis stage`: fibrosis-stage value; the F1 guide filters rows with value `1`.

`examples/guide\_synthetic\_example.xlsx` is a synthetic template and contains no real patient identifiers or image data. It demonstrates the column layout only.

## Pretrained weights

The original scripts use a ConvNeXt-Base pretrained checkpoint. Place the checkpoint at:

```text
weights/convnext\_base.fb\_in22k\_ft\_in1k/pytorch\_model.bin
```

Alternatively, set `CRYSTANET\_PRETRAINED\_WEIGHTS` to the full path of the checkpoint file. The checkpoint is not included in this repository.

## Running the scripts

Run commands from the repository root, after arranging the required data and pretrained checkpoint. For example:

```bash
python f0\_f1/scripts/cross\_validation/cv\_fold\_1.py
python f0\_f1/scripts/external\_validation/train\_and\_validate.py
```

The scripts preserve their original fold-specific split rules, training settings, and evaluation workflow. Results, checkpoints, and logs are written to the output directories specified by each script.

## Output files

The scripts save per-image prediction tables, per-epoch metrics, model checkpoints, and training logs. Generated outputs and model weights are not included in the source-code package.



