"""Utilities for loading and preparing the BRFSS diabetes dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DatasetSplits:
    """Container that holds prepared dataset splits."""

    train: TensorDataset
    val: Optional[TensorDataset] = None
    test: Optional[TensorDataset] = None

    def to_dataloaders(
        self,
        batch_size: int,
        *,
        shuffle_train: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
    ) -> Dict[str, DataLoader]:
        """Instantiate PyTorch dataloaders for each available split."""

        loaders: Dict[str, DataLoader] = {}
        if self.train is not None:
            loaders["train"] = DataLoader(
                self.train,
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
            )
        if self.val is not None:
            loaders["val"] = DataLoader(
                self.val,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        if self.test is not None:
            loaders["test"] = DataLoader(
                self.test,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        return loaders


class BRFSSDatasetLoader:
    """
    Helper that prepares the Diabetes Health Indicators dataset for ANFIS training.
    """

    # Default column names for BRFSS dataset (BMI, Age, Income, PhysHlth, Target)
    DEFAULT_COLUMNS = ["BMI", "Age", "Income", "PhysHlth", "Target"]
    DEFAULT_TARGET = "Target"

    def __init__(
        self,
        csv_path: str | Path,
        *,
        columns: list[str] = DEFAULT_COLUMNS,
        target_column: str = DEFAULT_TARGET,
        feature_subset: Optional[Sequence[str]] = None,
        drop_columns: Optional[Iterable[str]] = None,
        replace_missing: bool = True,
        normalization: str = "standard",
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        sample_size: Optional[int] = None,
    ):
        self.csv_path = Path(csv_path)
        self.columns = columns
        self.target_column = target_column
        self.feature_subset = feature_subset
        self.drop_columns = set(drop_columns or [])
        self.replace_missing = replace_missing
        self.normalization = normalization
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.sample_size = sample_size

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.csv_path}")

    def load(self) -> DatasetSplits:
        # Read CSV without headers and assign our column names
        df = pd.read_csv(self.csv_path, header=None, names=self.columns)

        if self.sample_size is not None and self.sample_size < len(df):
            df = df.sample(self.sample_size, random_state=self.random_state)

        df = df.drop(columns=[col for col in self.drop_columns if col in df.columns])

        if self.feature_subset is not None:
            missing = set(self.feature_subset) - set(df.columns)
            if missing:
                raise ValueError(f"Missing feature columns in dataset: {missing}")
            feature_df = df[list(self.feature_subset)]
        else:
            feature_df = df.drop(columns=[self.target_column])

        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not present in dataset.")

        target_series = df[self.target_column].astype("float32")

        if self.replace_missing:
            feature_df = feature_df.fillna(feature_df.median())
        else:
            combined = pd.concat([feature_df, target_series], axis=1)
            combined = combined.dropna()
            feature_df = combined.drop(columns=[self.target_column])
            target_series = combined[self.target_column]

        features = feature_df.to_numpy(dtype="float32")
        targets = target_series.to_numpy(dtype="float32").reshape(-1, 1)

        scaler = self._create_scaler()
        if scaler is not None:
            features = scaler.fit_transform(features)

        label_vector = targets.reshape(-1)

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            targets,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=label_vector,
        )

        train_labels = y_train.reshape(-1)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=train_labels,
        )

        train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
        test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

        return DatasetSplits(train=train_ds, val=val_ds, test=test_ds)

    def _create_scaler(self):
        if self.normalization is None or self.normalization.lower() == "none":
            return None
        if self.normalization.lower() == "standard":
            return StandardScaler()
        if self.normalization.lower() == "minmax":
            return MinMaxScaler()
        raise ValueError(f"Unknown normalization strategy: {self.normalization}")
