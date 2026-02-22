# My NAVSIM Agent

This repository contains my custom autonomous driving agents built on top of the [NAVSIM v2](https://github.com/autonomousvision/navsim) benchmark framework.

---

## 1. Environment Setup

Create the conda environment from the provided file:

```bash
conda env create -f environment.yml
conda activate navsim
```
(NOTE that this is an exact replica of the environment used by me, it expects you to run on Linux)

Then install the package in editable mode:

```bash
pip install -e .
```

---

## 2. Environment Variables

Set the following variables (e.g. in your `~/.bashrc` or a `setup.sh`):

```bash
export NAVSIM_DEVKIT_ROOT=/path/to/this/repo
export NAVSIM_EXP_ROOT=/path/to/experiment/outputs      # where checkpoints/logs are saved
export OPENSCENE_DATA_ROOT=/path/to/openscene/dataset   # root of the downloaded sensor data
```

---

## 3. Dataset Download

Use the scripts in the `download/` folder to download the dataset splits you need:

```bash
# Example: download the mini split for quick testing
bash download/download_mini_DOWNLOADED.sh

# Example: download the full navtrain split
bash download/download_navtrain_hf_DOWNLOADED.sh
```

For the full dataset download instructions and split descriptions, follow the official [NAVSIM data setup guide](https://github.com/autonomousvision/navsim/blob/main/docs/dataset.md).

---

## 4. Pretrained Model Weights

Download the necessary pretrained model weights from HuggingFace:

**[https://huggingface.co/Zzxxxxxxxx/gtrs/tree/main](https://huggingface.co/Zzxxxxxxxx/gtrs/tree/main)**

Place the downloaded `.ckpt` files under `weights/`.

---

## 5. Running Evaluation

Evaluation and training scripts are located in `scripts/evaluation/` and `scripts/training/`. Refer to the individual scripts for the required arguments and paths.
