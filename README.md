# Adaptive-Multi-view-Neighborhood-Retrieval
This repository provides the implementation used to reproduce the experiments
for the **Adaptive Multi-view Neighborhood Retrieval** paper at S+SSPR. The
method evaluates nearest-neighbor prediction in the original feature space,
PCA-reduced space, uniformly fused multi-view spaces, and an adaptive multi-view
fusion strategy.

## Overview

The approach constructs multiple representations of a tabular dataset and
compares their usefulness for distance-based prediction. The experimental
pipeline standardizes the input features, builds projection-based views,
computes Euclidean distance matrices, and evaluates KNN predictions under four
settings:

- `original_space`: KNN in the standardized original feature space.
- `pca_only`: KNN after PCA projection.
- `uniform_multiview`: KNN after averaging normalized distance matrices across
  all views.
- `adaptive_multiview`: KNN after learning query-dependent view weights with a
  small neural network trained on validation queries.

The implementation supports both classification and regression datasets, but
the paper experiments use classification metrics.

## File Structure

```text
.
├── data/                         # Local TALENT-derived datasets
├── scripts/
│   └── run_paper_experiments.py   # Convenience script for paper datasets
├── src/
│   ├── __init__.py
│   ├── adaptive.py                # Adaptive view-weighting model and 
│   ├── dataloader.py              # Dataset loading and categorical 
│   ├── experiments.py             # Experiment orchestration
│   ├── fusion.py                  # Distances, KNN, metrics, and view 
│   └── outputs.py                 # JSON and CSV output utilities
├── .gitignore
├── LICENSE
├── main.py                        # Command-line entry point
├── requirements.txt
└── README.md
```

## Installation

We recommend using a clean Python virtual environment.

```bash
pip install -r requirements.txt
```

## Dataset Format

Each dataset should be stored under `data/{dataset}/` with the following layout:

```text
data/{dataset}/N_train.npy
data/{dataset}/N_val.npy
data/{dataset}/N_test.npy
data/{dataset}/C_train.npy
data/{dataset}/C_val.npy
data/{dataset}/C_test.npy
data/{dataset}/y_train.npy
data/{dataset}/y_val.npy
data/{dataset}/y_test.npy
data/{dataset}/info.json
```

Numerical files (`N_*`) or categorical files (`C_*`) may be absent, but at least
one feature family must be available. Labels (`y_*`) and `info.json` are
required. Categorical features are encoded with:

```python
OneHotEncoder(handle_unknown="ignore", sparse_output=False)
```

The `info.json` file should include `task_type`. Values `binclass` and
`multiclass` are treated as classification; other values are treated as
regression.

The datasets used in the experiments are extracted from the
[TALENT benchmark](https://github.com/LAMDA-Tabular/TALENT).

## Usage

Run all four experiments for one dataset:

```bash
python main.py --dataset adult
```

Run a subset of experiments:

```bash
python main.py --dataset adult --experiments original_space,uniform_multiview
```

Use a custom dataset or output root:

```bash
python main.py \
  --dataset adult \
  --data_root data \
  --output_root output
```

The main experimental arguments are:

- `--dataset`: Dataset folder name under `data_root`.
- `--k`: Number of nearest neighbors.
- `--norm`: Distance normalization mode: `mean`, `median`, `zscore`, or `none`.
- `--jl_eps`: Johnson-Lindenstrauss epsilon used for random projection size.
- `--rp_frac`: Maximum random projection size as a fraction of input features.
- `--hidden`: Hidden size of the adaptive view-weighting network.
- `--n_epochs`: Number of adaptive training epochs.
- `--batch_size`: Validation-query batch size for adaptive training.
- `--lr`: Learning rate for Adam.
- `--temperature`: Soft KNN temperature.
- `--lambda_ent`: Entropy regularization coefficient.

## Paper Datasets

The helper script runs the dataset names used in the paper experiments:

```bash
python scripts/run_paper_experiments.py
```

If local dataset folder names differ, edit `PAPER_DATASETS` in
`scripts/run_paper_experiments.py`.

## Outputs

For each dataset, outputs are written under `output/{dataset}/`:

```text
output/{dataset}/{experiment}/metrics.json
output/{dataset}/summary.json
output/{dataset}/accuracy_comparison.json
output/{dataset}/log.txt
```

Global outputs are written to:

```text
output/results.csv
output/accuracy_comparison.json
```

For classification datasets, `metrics.json` includes accuracy, macro F1, and
weighted F1. For regression datasets, it includes RMSE, MAE, and R2.

## Reproducibility Notes

The implementation follows the stochastic behavior of the experiment script:
Gaussian random projections and adaptive model initialization are not fixed by a
global seed, while TruncatedSVD uses `random_state=42`. As a result,
`uniform_multiview` and `adaptive_multiview` can vary across fresh executions.

For exact comparisons, use the same dataset splits, dependency versions, command
line arguments, and hardware setup across runs.
