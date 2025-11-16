"""Runner script for `myANFIS.py`.

Loads a small example CSV from `input/`, trains the NumPy ANFIS for a few epochs,
prints RMSE and saves predictions to `output/`.

Usage (from repo root):
    D:/code_space/Anfis/.venv/Scripts/python.exe run_myanfis.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys

from myANFIS import myanfis, plot_predictions


def main():
    repo_root = Path(__file__).resolve().parents[0]
    input_dir = repo_root / 'input'
    out_dir = repo_root / 'output'
    out_dir.mkdir(exist_ok=True)

    # Choose a small dataset present in the repo
    csv_candidates = [
        'BMI-Age-Income-PhysHlth_and_target_150x5.csv',
        'Features_and_target.csv',
        'iris.csv'
    ]

    dataset_path = None
    for name in csv_candidates:
        p = input_dir / name
        if p.exists():
            dataset_path = p
            break

    if dataset_path is None:
        print('No suitable dataset found in input/. Please provide one of:', csv_candidates)
        sys.exit(1)

    print('Using dataset:', dataset_path)
    # load as numeric array (assumes last column is target)
    df = pd.read_csv(dataset_path)
    data = df.values.astype(float)
    inputs = data[:, :-1]

    # Training parameters (kept small for demo)
    epoch_n = 5
    mf = 2
    step_size = 0.1
    decrease_rate = 0.9
    increase_rate = 1.1

    print(f'Training myANFIS: epochs={epoch_n}, mf={mf}, samples={data.shape[0]}, features={inputs.shape[1]}')
    bestnet, predictions, RMSE = myanfis(data, inputs, epoch_n, mf, step_size, decrease_rate, increase_rate)

    final_rmse = float(RMSE[max(0, epoch_n-1)-1, 0]) if epoch_n > 0 else float('nan')
    # RMSE indexing in myANFIS stores at position iter-1; handle small epochs
    print('Training finished. RMSE history (last value may be at index 0 for single epoch):')
    print(RMSE.flatten())

    # Save predictions and RMSE
    out_preds = out_dir / 'myanfis_predictions.csv'
    np.savetxt(out_preds, predictions, delimiter=',')
    out_npz = out_dir / 'myanfis_results.npz'
    np.savez(out_npz, predictions=predictions, RMSE=RMSE)
    print('Saved predictions to:', out_preds)
    print('Saved results to:', out_npz)

    try:
        plot_predictions(data[:, -1], predictions)
    except Exception:
        pass


if __name__ == '__main__':
    main()
