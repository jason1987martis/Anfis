# GPU ANFIS Toolkit

This folder contains a PyTorch reimplementation of the custom ANFIS pipeline with GPU support and utilities tailored for the Diabetes Health Indicators (BRFSS) dataset.

## Contents

- `model.py` – differentiable ANFIS network with generalized bell membership functions, ready for CUDA.
- `dataset_loader.py` – pandas/NumPy based loader for the BRFSS CSV export with normalization and train/val/test splits.
- `trainer.py` – training loop with CLI entry point for rapid experiments.
- `explainability.py` / `explainability.md` – layer-by-layer inspection helpers for explainable AI reporting.
- `__init__.py` – convenience exports to treat the folder as a package.

## Quickstart

1. Download the Kaggle dataset locally (requires Kaggle authentication) and note the CSV path.
2. Install dependencies (PyTorch, pandas, scikit-learn). Example:
   ```bash
   pip install torch pandas scikit-learn
   ```
3. Run the trainer (CUDA is used automatically if available):
   ```bash
   python -m gpu_version.trainer --data /path/to/diabetes_binary_health_indicators_BRFSS2015.csv --epochs 20 --batch-size 2048
   ```

The script reports train/validation losses and classification accuracy per epoch. The `train_anfis` helper may also be imported and used inside notebooks for more customized pipelines.

## Explainability

Import `ANFISExplainer` to grab per-layer tensors and top rule contributions for any prediction. See `explainability.md` for a walkthrough you can adapt for presentations or audit reports.
