"""GPU-accelerated ANFIS package."""

from .model import ANFISNetwork, GaussianMembershipLayer
from .dataset_loader import BRFSSDatasetLoader, DatasetSplits
from .trainer import train_anfis
from .explainability import ANFISExplainer, LayerBreakdown, RuleContribution

__all__ = [
    "ANFISNetwork",
    "GaussianMembershipLayer",
    "BRFSSDatasetLoader",
    "DatasetSplits",
    "train_anfis",
    "ANFISExplainer",
    "LayerBreakdown",
    "RuleContribution",
]
