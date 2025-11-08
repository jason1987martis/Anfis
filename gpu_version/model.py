"""PyTorch implementation of an ANFIS model that supports GPU acceleration."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _cartesian_rule_indices(num_inputs: int, mf_count: int) -> Tensor:
    """Create a full grid of rule indices."""
    if num_inputs <= 0 or mf_count <= 0:
        raise ValueError("num_inputs and mf_count must be positive integers.")
    combinations = itertools.product(range(mf_count), repeat=num_inputs)
    data = torch.tensor(list(combinations), dtype=torch.long)
    return data


@dataclass
class RuleStrengths:
    """Container for rule strength diagnostics."""

    firing: Tensor
    normalized: Tensor


class GaussianMembershipLayer(nn.Module):
    """Gaussian membership functions with learnable parameters."""

    def __init__(self, input_dim: int, mf_count: int):
        super().__init__()
        if input_dim <= 0 or mf_count <= 0:
            raise ValueError("input_dim and mf_count must be positive.")

        self.input_dim = input_dim
        self.mf_count = mf_count

        # Parameters are stored in an unconstrained space
        self._a = nn.Parameter(torch.ones(input_dim, mf_count))
        self._c = nn.Parameter(torch.zeros(input_dim, mf_count))

    @property
    def a(self) -> Tensor:
        """Width parameter (standard deviation), always positive."""
        return F.softplus(self._a)

    @property
    def c(self) -> Tensor:
        """Center parameter (mean)."""
        return self._c

    def forward(self, x: Tensor) -> Tensor:
        """
        Evaluate membership degrees using Gaussian function.

        Args:
            x: Tensor of shape (batch, input_dim)

        Returns:
            membership degrees with shape (batch, input_dim, mf_count)
        """
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"Expected input shape (batch, {self.input_dim}), got {tuple(x.shape)}.")

        x_expanded = x.unsqueeze(-1)  # (batch, input_dim, 1)
        a = self.a  # Width (standard deviation)
        c = self.c  # Center (mean)

        # Simple Gaussian function: exp(-(x-μ)²/(2σ²))
        squared_diff = (x_expanded - c).pow(2)
        memberships = torch.exp(-squared_diff / (2 * a.pow(2) + 1e-6))
        return memberships

    @torch.no_grad()
    def initialize_from_statistics(
        self,
        data: Tensor,
        spread_scale: float = 0.5,
        eps: float = 1e-3,
    ) -> None:
        """
        Initialize membership parameters from data statistics.

        Args:
            data: Tensor (n_samples, input_dim)
            spread_scale: relative spread applied to standard deviation
            eps: numerical floor for widths
        """
        if data.dim() != 2 or data.size(1) != self.input_dim:
            raise ValueError("Data must be of shape (n_samples, input_dim).")

        quantiles = torch.linspace(0.05, 0.95, self.mf_count, device=data.device)
        centers = torch.quantile(data, quantiles, dim=0)

        std = data.std(dim=0).clamp_min(eps)
        spreads = (std * spread_scale).unsqueeze(1).expand(-1, self.mf_count)

        def _softplus_inverse(value: Tensor) -> Tensor:
            return value + torch.log(-torch.expm1(-value))

        self._c.copy_(centers.T)
        self._a.copy_(_softplus_inverse(spreads))


class ANFISNetwork(nn.Module):
    """Adaptive neuro-fuzzy inference system backed by PyTorch."""

    def __init__(
        self,
        input_dim: int,
        mf_count: int,
        rule_indices: Optional[Tensor] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mf_count = mf_count

        if rule_indices is None:
            rule_indices = _cartesian_rule_indices(input_dim, mf_count)
        if rule_indices.dim() != 2 or rule_indices.size(1) != input_dim:
            raise ValueError("rule_indices must have shape (num_rules, input_dim).")

        self.register_buffer("rule_indices", rule_indices.clone())
        self.num_rules = rule_indices.size(0)

        self.membership_layer = GaussianMembershipLayer(input_dim, mf_count)
        self.consequents = nn.Parameter(torch.zeros(self.num_rules, input_dim + 1))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network."""
        memberships = self.membership_layer(x)
        strengths = self._rule_strengths(memberships)

        linear_input = torch.cat(
            (x, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)),
            dim=1,
        )

        rule_outputs = torch.matmul(linear_input, self.consequents.t())
        weighted = strengths.normalized * rule_outputs
        y = torch.sum(weighted, dim=1, keepdim=True)
        return y

    def forward_with_details(self, x: Tensor) -> Tuple[Tensor, RuleStrengths, Tensor]:
        """Forward pass with detailed outputs."""
        memberships = self.membership_layer(x)
        strengths = self._rule_strengths(memberships)

        linear_input = torch.cat(
            (x, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)),
            dim=1,
        )
        rule_outputs = torch.matmul(linear_input, self.consequents.t())
        y = torch.sum(strengths.normalized * rule_outputs, dim=1, keepdim=True)
        return y, strengths, rule_outputs

    def _rule_strengths(self, memberships: Tensor) -> RuleStrengths:
        """
        Compute raw and normalized firing strengths.

        Args:
            memberships: Tensor (batch, input_dim, mf_count)
        """
        batch_size = memberships.size(0)
        expanded = memberships.unsqueeze(1).expand(-1, self.num_rules, -1, -1)

        gather_indices = self.rule_indices.view(1, self.num_rules, self.input_dim, 1)
        gather_indices = gather_indices.expand(batch_size, -1, -1, -1)

        selected = torch.gather(expanded, dim=3, index=gather_indices)
        firing = selected.squeeze(-1).prod(dim=2)

        norm = firing / torch.clamp(firing.sum(dim=1, keepdim=True), min=1e-6)
        return RuleStrengths(firing=firing, normalized=norm)

    @torch.no_grad()
    def initialize_memberships(self, data: Tensor, **kwargs) -> None:
        """Delegate initialization to the membership layer."""
        self.membership_layer.initialize_from_statistics(data, **kwargs)

    def to_device(self, device: torch.device | str) -> "ANFISNetwork":
        """Convenience helper for moving the model."""
        return self.to(device)