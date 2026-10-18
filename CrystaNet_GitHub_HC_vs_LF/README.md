# CrystaNet

**Deep learning for plasma-templated crystallomics and liver fibrosis detection.**

## Part 1 — Healthy controls versus liver fibrosis (HC vs. LF)

This repository contains Python source code for the binary classification of healthy controls (HC, label `0`) and individuals with liver fibrosis (LF, label `1`) using crystal microscopy images. It includes patient-grouped five-fold internal cross-validation, final model training on internal data, and external-center inference with the final trained checkpoint.

## Repository structure

```text
CrystaNet/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── examples/
│   └── guide\_synthetic\_example.xlsx
└── hc\_vs\_liver\_fibrosis/
    └── scripts/
        ├── cross\_validation/          # cv\_fold\_1.py through cv\_fold\_5.py
        └── external\_validation/       # train\_final\_model.py and infer\_external.py
```

The published scripts use English filenames, comments, log messages, and portable paths. The model architecture, fold assignments, preprocessing operations, hyperparameters, and inference calculations are preserved.

## Workflow

1. **Five-fold cross-validation:** five provided scripts run internal training and evaluation with fold assignments from the Excel guide's patient identifiers (`identifier % 5`). Each patient has a unique identifier, and all images belonging to that patient follow the same fold assignment.
2. **Final model training:** use the provided full internal training script to obtain the final checkpoint.
3. **External inference:** load the final internally trained checkpoint and predict the external-center images, exporting class labels and prediction probabilities to CSV.

The model is implemented with `timm` ConvNeXt-Base. Dataset and output paths can be configured without editing machine-specific drive letters.

## Guide spreadsheet

The first sheet (`Sheet1`) uses the following column order:

|Column|Excel heading|Meaning|
|-|-|-|
|A|`filename`|Filename of the image in its class-specific `images/` directory|
|B|`number`|Unique patient identifier, reused across images from the same patient|

The scripts read these columns by position. The second sheet may contain human-readable filename notes but is not used in model training. See [`examples/guide\_synthetic\_example.xlsx`](examples/guide_synthetic_example.xlsx) for a synthetic template (not clinical data).

## Data directory conventions

```text
data/
├── internal/
│   ├── healthy/{guide.xlsx,images/}
│   └── fibrosis/{guide.xlsx,images/}
├── external\_zengcheng/
│   ├── healthy/{guide.xlsx,images/}
│   └── fibrosis/{guide.xlsx,images/}
└── external\_shijiazhuang/
    ├── healthy/{guide.xlsx,images/}
    └── fibrosis/{guide.xlsx,images/}
weights/
├── convnext\_pretrained/pytorch\_model.bin
└── hc\_vs\_lf\_final.pth
outputs/
```

The directory listing describes default input locations; the images, original clinical guides, pretrained weights and final checkpoint are **not bundled** in this repository. Any file needed by a particular script must be provided at the corresponding path.

By default, paths are resolved relative to the GitHub repository root. Optionally configure:

* `CRYSTANET\_DATA\_ROOT`: input image/guide base directory
* `CRYSTANET\_WEIGHTS\_ROOT`: checkpoint base directory
* `CRYSTANET\_OUTPUT\_ROOT`: results directory

For example, from the repository root:

```bash
conda env create -f environment.yml
conda activate cellpose
python hc\_vs\_liver\_fibrosis/scripts/external\_validation/infer\_external\_shijiazhuang.py
```

The example inference command requires you to supply the trained checkpoint and guide/image files. For five-fold training, run `cv\_fold\_1.py` through `cv\_fold\_5.py` inside `hc\_vs\_liver\_fibrosis/scripts/cross\_validation/` after setting up the internal data.

## Software environment

The Python and package versions are taken directly from the supplied environment export (`environment.yml`). The main dependencies are pinned in [`requirements.txt`](requirements.txt), including Python 3.8.19, PyTorch 2.2.2+cu118, torchvision 0.17.2+cu118, timm 0.9.16, NumPy 1.24.4, pandas 2.0.3, and scikit-learn 1.3.2. The full exported Conda environment includes CUDA 11.8 runtime packages. It was exported on Windows; cross-platform installations may require platform-specific adjustments.

To restore the recorded environment, from the repository root:

```bash
conda env create -f environment.yml
```

Alternatively, install the core Python packages with pip (CUDA-specific PyTorch wheels use the official PyTorch CUDA 11.8 index):

```bash
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

## Code and data sharing

The repository distributes research source code and a synthetic guide template, not patient-identifiable clinical information. Version-specific environments, model checkpoints and authorized clinical datasets should be documented or shared separately as permitted. `environment.yml` contains the provided Conda environment with exact package versions and build identifiers (with the computer-specific `prefix` removed). `requirements.txt` lists the main Python dependencies pinned to the versions recorded in that environment.

## License

No license has been assigned yet.

