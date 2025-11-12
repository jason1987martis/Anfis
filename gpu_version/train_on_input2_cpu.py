"""CPU-only trainer for ANFIS on the BRFSS diabetes dataset.

This is a copy of `train_on_input2.py` but forces CPU execution and writes
its model to `output/anfis_input2_cpu.pth` to avoid overwriting the GPU model.

Usage (from repo root):
    D:/code_space/Anfis/.venv/Scripts/python.exe gpu_version/train_on_input2_cpu.py
"""

from __future__ import annotations

import os
import sys
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

    X, y, target_col = detect_target_and_split(df)
    return X, y, target_col


def detect_target_and_split(df: pd.DataFrame):
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
        raise ValueError(f"Could not detect target column automatically. Dataset columns: {cols}")

    X = df.drop(columns=[target_col]).to_numpy(dtype='float32')
    y = df[target_col].to_numpy(dtype='float32').reshape(-1, 1)
    return X, y, target_col


def main():
    repo_root = Path(__file__).resolve().parents[1]
    input2 = repo_root / 'input2'
    csv_path = input2 / 'diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
    xlsx_path = input2 / 'diabetes_binary_5050split_health_indicators_BRFSS2015.xlsx'

    if csv_path.exists():
        dataset_path = csv_path
    elif xlsx_path.exists():
        dataset_path = xlsx_path
    else:
        candidates = list(input2.glob('diabetes_binary*'))
        if candidates:
            dataset_path = candidates[0]
        else:
            print(f"Dataset not found in: {input2}")
            sys.exit(1)

    # Force CPU execution
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # load CSV or XLSX
    if str(dataset_path).lower().endswith('.csv'):
        X, y, _ = load_dataset(str(dataset_path))
    else:
        df = pd.read_excel(dataset_path)
        if 'Diabetes_binary' not in df.columns:
            raise ValueError("Expected 'Diabetes_binary' column in dataset header.")
        X = df.drop(columns=['Diabetes_binary']).to_numpy(dtype='float32')
        y = df['Diabetes_binary'].to_numpy(dtype='float32').reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_tmp, y_train, y_tmp = train_test_split(X_scaled, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device)

    from torch.utils.data import TensorDataset, DataLoader

    batch_size = 256
    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    input_dim = X.shape[1]
    print(f"Dataset: {dataset_path} | samples: {len(X)} | features: {input_dim}")

    mf_count = 3
    num_rules = 512
    rule_indices = torch.randint(low=0, high=mf_count, size=(num_rules, input_dim))
    model = ANFISNetwork(input_dim=input_dim, mf_count=mf_count, rule_indices=rule_indices).to(device)
    print(f"Using {num_rules} sampled rules (mf_count={mf_count}, input_dim={input_dim})")

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    best_val = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                loss = loss_fn(out, yb)
                val_loss += loss.item() * xb.size(0)
                preds = (torch.sigmoid(out) > 0.5).float()
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict().copy()

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    import numpy as np
    y_true = []
    y_pred = []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb)
            probs = torch.sigmoid(out).cpu().numpy().ravel()
            preds = (probs > 0.5).astype(int)
            y_pred.extend(preds.tolist())
            y_true.extend(yb.cpu().numpy().ravel().astype(int).tolist())

    from sklearn.metrics import classification_report, confusion_matrix
    print("\nTest set results:")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    out_dir = repo_root / 'output'
    out_dir.mkdir(exist_ok=True)
    model_path = out_dir / 'anfis_input2_cpu.pth'
    torch.save({'model_state_dict': model.state_dict()}, str(model_path))
    print(f"Saved model to: {model_path}")


if __name__ == '__main__':
    main()
