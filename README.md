# SLICE: Stabilized LIME for Consistent Explanations for Image Classification

Welcome to the official repository for our reproducibility research on **SLICE: Stabilized LIME for Consistent Explanations for Image Classification**, an explainability algorithm that identifies positive and negative superpixels in an image with consistency. The code uses PyTorch, multi-threading, and multiprocessing to provide an efficient and fast implementation of the algorithms.

This repository contains the code used to research the reproducibility of the experiments conducted in the original 'SLICE' paper. Additionally, it includes our contributions:

- **GRID-LIME**, a new model that provides better stability than LIME and better explainability than SLICE.
- **Ground Truth Overlap (GTO)** metric to measure how well an explainability model explains the classified output in terms of region overlap.

## Table of Contents
- [Installation Guide](#installation-guide)
- [Datasets](#datasets)
- [How to Run](#how-to-run)
  - [Experiments Notebook](#experiments-notebook)
  - [Running Individual Explanation Model](#running-individual-explanation-model)
  - [Running Generate Plots Code](#running-generate-plots-code)

## Installation Guide

1. Download the repository as a `.zip` file or clone it using:
   
   ```bash
   git clone [repository-url]
   ```

2. **If running the code on a local machine**, follow these steps:
   
   a. Install the required packages from the `.yml` file:
   
   ```bash
   conda env create -f environment.yml
   ```
   
   b. Activate or deactivate the environment:
   
   ```bash
   conda activate slice_env
   conda deactivate slice_env
   ```
   
   c. To delete the environment:
   
   ```bash
   conda remove -n slice_env --all
   ```
   
   d. Additional packages can be installed using:
   
   ```bash
   pip install [package_name]
   ```

## Datasets

The Oxford-IIIT Pets dataset and Pascal VOC dataset are dynamically downloaded via the code (based on the dataset parameter passed when running `run_explainer.py`). There is no need to download them separately.

## How to Run

Once the environment is set up correctly, you can run the code.

### Experiments Notebook

A Jupyter Notebook (`run_experiments.ipynb`) runs `run_explainer.py` for all combinations of:
- Pretrained model: `InceptionV3` / `ResNet50`
- Explainer model: `LIME` / `SLICE` / `GRID-LIME`
- Dataset: `Oxford-IIIT Pets` / `Pascal VOC`

This generates the results folder and plots **CCM, GTO, and AOPC** metrics.

#### Running on Google Colab (Recommended, preferably on an A100 GPU)

1. Open `run_experiments.ipynb` in Google Colab.
2. Change the runtime to **A100 GPU** (preferred) or any available GPU.
3. Insert a new cell at the top and paste the following:
   
   ```python
   !pip install rbo
   from google.colab import files
   import os
   files.upload()
   
   !unzip [repository-name].zip
   
   os.chdir('[repository-name]')
   os.remove('run_experiments.ipynb')
   ```

4. Click **Run All** in the Runtime tab. The first cell will prompt you to upload a `.zip` file of the repository downloaded in the installation step.

#### Running Locally (Not Recommended, Slow Execution)

1. Open Jupyter Notebook with the `slice_env` environment activated.
2. Run all cells in the notebook.

For both methods, the plots will be saved in the **`Final Plots`** folder (automatically created).

### Running Individual Explanation Model

You can also run individual explainability models by executing `run_explainer.py` with the appropriate command-line arguments:

```bash
python run_explainer.py --explain_model [lime/grid_lime/slice] \
                        --pretrained_model [inceptionv3/resnet50] \
                        --dataset [oxpets/pvoc] \
                        --metrics [yes/no] \
                        --num_runs [positive_integer] \
                        --num_perturb [positive_integer] \
                        --num_images_from_dataset [positive_integer]
```

#### Available Arguments:
- `--explain_model` (default: `slice`) – Explainability model (`lime`, `grid_lime`, or `slice`).
- `--pretrained_model` (default: `inceptionv3`) – Pretrained model (`inceptionv3` or `resnet50`).
- `--dataset` (default: `oxpets`) – Dataset (`oxpets` or `pvoc`).
- `--metrics` (default: `no`) – Whether to calculate metrics (`yes` or `no`).
- `--num_runs` (default: `5`) – Number of times to run each explainer algorithm (positive integer).
- `--num_perturb` (default: `500`) – Number of perturbations used in the algorithm (positive integer).
- `--num_images_from_dataset` (default: `50`) – Number of images used from the dataset (positive integer).

### Running Generate Plots Code

If model results have already been generated, you can run `generate_results.py` directly with the following command:

```bash
python generate_results.py --generate_ccm_plots_flag [yes/no] \
                           --generate_gto_plots_flag [yes/no] \
                           --generate_aopc_plots_flag [yes/no] \
                           --aopc_sigma_constant [yes/no] \
                           --num_iter [positive_integer]
```

#### Available Arguments:
- `--generate_ccm_plots_flag` (default: `yes`) – Generate CCM metric plot (`yes` or `no`).
- `--generate_gto_plots_flag` (default: `yes`) – Generate GTO metric plot (`yes` or `no`).
- `--generate_aopc_plots_flag` (default: `yes`) – Generate AOPC metric plot (`yes` or `no`).
- `--aopc_sigma_constant` (default: `no`) – Use a constant sigma value of 0 for SLICE AOPC plots (`yes` or `no`).
- `--num_iter` (default: `10`) – Number of times the explainer algorithm was run for metric calculations (positive integer).

The plots will be saved in the **`Final Plots`** folder (automatically created).
