# Explainable Pipeline

The GPU ANFIS stack includes an `ANFISExplainer` utility (`gpu_version/explainability.py`) that exposes the intermediate tensors for a single prediction. Use it when you need to walk through the network layer by layer.

## Layers at a Glance

1. **Layer 1 – Memberships**  
   Gaussian bell parameters (`a`, `b`, `c`) convert the raw input into membership degrees for each fuzzy set.

2. **Layer 2 – Rule Firing Strengths**  
   For every rule, multiply the relevant membership degrees. These are the raw firing strengths.

3. **Layer 3 – Normalisation**  
   Divide each firing strength by the sum across rules so they form a probability simplex.

4. **Layer 4 – Consequent Evaluation**  
   Append a bias term to the original input, perform a per-rule linear combination, and weight it by the normalised strengths.

5. **Layer 5 – Aggregation**  
   Sum the weighted rule outputs to produce the final network prediction.

## Quick Usage

```python
import torch
from gpu_version import ANFISNetwork, ANFISExplainer

model = ANFISNetwork(input_dim=4, mf_count=3)
explainer = ANFISExplainer(model, feature_names=["BMI", "Age", "Income", "PhysHlth"])

sample = torch.tensor([0.2, -0.1, 0.5, 0.0])
layer_view = explainer.explain_sample(sample, top_k=3)

for rule in layer_view.top_rules:
    print(explainer.describe_rule(rule))
```

`LayerBreakdown` contains membership tensors, per-rule activations, weighted contributions, and the aggregated output so you can visualise or serialize them for reports or dashboards.
