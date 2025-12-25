"""Explainability utilities for the GPU-enabled ANFIS model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .model import ANFISNetwork, RuleStrengths


@dataclass
class RuleContribution:
    """Stores per-rule diagnostics for a single sample."""

    rule_index: int
    antecedent_mfs: List[int]
    firing_strength: float
    normalized_strength: float
    consequent_weights: Tensor
    rule_output: float
    contribution: float


@dataclass
class LayerBreakdown:
    """Captures tensors flowing through the ANFIS layers."""

    input: Tensor
    membership: Tensor
    rule_strengths: RuleStrengths
    rule_outputs: Tensor
    weighted_outputs: Tensor
    final_output: Tensor
    top_rules: List[RuleContribution]


class ANFISExplainer:
    """
    Generates human-readable breakdowns of ANFIS decisions layer by layer.
    """

    def __init__(
        self,
        model: ANFISNetwork,
        feature_names: Optional[Sequence[str]] = None,
    ):
        self.model = model
        self.feature_names = list(feature_names) if feature_names is not None else None

    def explain_sample(
        self,
        sample: Tensor,
        *,
        top_k: int = 5,
        device: Optional[torch.device] = None,
    ) -> LayerBreakdown:
        """
        Produce a layer-wise explanation for a single input sample.
        """
        if sample.dim() == 1:
            sample = sample.unsqueeze(0)
        if sample.dim() != 2 or sample.size(0) != 1:
            raise ValueError("sample must be a single example with shape (input_dim,) or (1, input_dim).")

        if device is None:
            device = next(self.model.parameters()).device

        sample = sample.to(device)
        self.model.eval()

        with torch.no_grad():
            membership = self.model.membership_layer(sample)
            output, strengths, rule_outputs = self.model.forward_with_details(sample)

        weighted = strengths.normalized * rule_outputs
        top_rules = self._select_top_rules(
            strengths,
            rule_outputs,
            weighted,
            top_k=top_k,
            device=device,
        )

        return LayerBreakdown(
            input=sample.cpu(),
            membership=membership.cpu(),
            rule_strengths=RuleStrengths(
                firing=strengths.firing.cpu(),
                normalized=strengths.normalized.cpu(),
            ),
            rule_outputs=rule_outputs.cpu(),
            weighted_outputs=weighted.cpu(),
            final_output=output.cpu(),
            top_rules=top_rules,
        )

    def _select_top_rules(
        self,
        strengths: RuleStrengths,
        rule_outputs: Tensor,
        weighted: Tensor,
        *,
        top_k: int,
        device: torch.device,
    ) -> List[RuleContribution]:
        contribution_scores = weighted.squeeze(0)
        abs_scores = contribution_scores.abs()
        k = min(top_k, contribution_scores.numel())
        top_values, top_indices = torch.topk(abs_scores, k)

        antecedents = self.model.rule_indices.to(device)

        contributions: List[RuleContribution] = []
        for score, rule_idx in zip(top_values, top_indices):
            idx = rule_idx.item()
            contributions.append(
                RuleContribution(
                    rule_index=idx,
                    antecedent_mfs=antecedents[idx].tolist(),
                    firing_strength=strengths.firing[0, idx].item(),
                    normalized_strength=strengths.normalized[0, idx].item(),
                    consequent_weights=self.model.consequents[idx].detach().cpu(),
                    rule_output=rule_outputs[0, idx].item(),
                    contribution=contribution_scores[idx].item(),
                )
            )

        contributions.sort(key=lambda rc: abs(rc.contribution), reverse=True)
        return contributions

    def describe_rule(self, rule: RuleContribution) -> str:
        """
        Build a human-readable rule explanation string.
        """
        antecedent_terms = []
        for dim, mf_index in enumerate(rule.antecedent_mfs):
            if self.feature_names and dim < len(self.feature_names):
                feat = self.feature_names[dim]
            else:
                feat = f"feature_{dim}"
            antecedent_terms.append(f"{feat} -> MF {mf_index}")

        antecedent_text = " AND ".join(antecedent_terms)
        return (
            f"Rule {rule.rule_index}: IF {antecedent_text} "
            f"THEN output = {rule.rule_output:.4f} (firing={rule.firing_strength:.4f}, "
            f"weight={rule.normalized_strength:.4f}, contribution={rule.contribution:.4f})"
        )
