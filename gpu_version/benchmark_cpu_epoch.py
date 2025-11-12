"""Quick CPU epoch benchmark for ANFIS training.

This script loads the same data and model setup as the full trainer but only
executes one training epoch and reports elapsed time. Use this per-epoch time
to estimate the full 20-epoch runtime on CPU.
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import ANFISNetwork


def load_dataset(csv_path: str):
    p = str(csv_path)
    if p.lower().endswith('.csv'):
        df = pd.read_csv(p)
    elif p.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)
    cols = list(df.columns)
    target_col = None
    for c in cols:
        if 'diabet' in str(c).lower():
            target_col = c
            break
    if target_col is None:
        for name in ['Diabetes_binary', 'diabetes_binary', 'Diabetes binary', 'diabetes']:
            if name in cols:
                target_col = name
                break
    if target_col is None:
        raise ValueError('Could not detect target column automatically.')
    X = df.drop(columns=[target_col]).to_numpy(dtype='float32')
    y = df[target_col].to_numpy(dtype='float32').reshape(-1, 1)
    return X, y


def main():
    repo_root = Path(__file__).resolve().parents[1]
    input2 = repo_root / 'input2'
    csv_path = input2 / 'diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
    if not csv_path.exists():
        candidates = list(input2.glob('diabetes_binary*'))
        if not candidates:
            print(f'Dataset not found in {input2}')
            return
        csv_path = candidates[0]

    device = torch.device('cpu')
    print('Benchmark device:', device)

    X, y = load_dataset(str(csv_path))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_tmp, y_train, y_tmp = train_test_split(X_scaled, y, test_size=0.30, random_state=42, stratify=y)

    # convert to tensors
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)

    from torch.utils.data import TensorDataset, DataLoader
    batch_size = 256
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    input_dim = X.shape[1]
    mf_count = 3
    num_rules = 512
    rule_indices = torch.randint(low=0, high=mf_count, size=(num_rules, input_dim))
    model = ANFISNetwork(input_dim=input_dim, mf_count=mf_count, rule_indices=rule_indices).to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Run one epoch and measure
    model.train()
    start = time.time()
    total_loss = 0.0
    n_samples = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)
    end = time.time()
    epoch_time = end - start
    avg_loss = total_loss / n_samples if n_samples > 0 else float('nan')
    print(f'One epoch time (CPU): {epoch_time:.4f} seconds | avg train loss: {avg_loss:.6f} | samples: {n_samples}')
    projected = epoch_time * 20
    print(f'Projected time for 20 epochs (CPU): {projected:.1f} seconds (~{projected/60:.1f} minutes)')

if __name__ == '__main__':
    main()
