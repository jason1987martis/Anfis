# ANFIS Parameter Fitting Options

Below are complementary strategies you can mix and match when tuning the GPU-based ANFIS model. They are grouped so you can decide how aggressive or exploratory you want to be.

## 1. Membership Initialisation
- **Quantile seeding (default):** Initialise centres from input quantiles and spreads from feature standard deviations (current implementation).
- **K-means centroids:** Cluster each input dimension separately to capture multimodal distributions before training.
- **Expert priors:** Override specific `c` (centres) and `a` (spreads) for domain-critical features (e.g., BMI thresholds suggested by clinicians).

## 2. Rule Base Construction
- **Full grid (default):** All combinations of membership functions across inputs (mf_count^inputs). Best when input count is modest.
- **Sparse expert rules:** Supply a curated list of rules if you want to limit combinatorial growth or align with domain knowledge.
- **Feature grouping:** Build separate ANFIS blocks per feature subset (e.g., demographic vs. behavioural) and fuse their outputs.

## 3. Training Schedules
- **Adam with static LR (default):** Works well for quick convergence; monitor validation loss to stop early.
- **Cyclic learning rates:** Cycle between low/high LR to escape plateaus; pair with cosine annealing.
- **Two-stage optimisation:** Freeze membership parameters for initial epochs (train consequents only), then unfreeze for fine-tuning.

## 4. Regularisation & Stability
- **Rule dropout:** Randomly drop a fraction of rules per batch to reduce overfitting on redundant rules.
- **L1 penalty on consequents:** Encourages sparse linear terms, aiding interpretability of rule consequents.
- **Temperature annealing:** Scale firing strengths to sharpen or smooth rule competition during training.

## 5. Evaluation Targets
- **Regression-style MSE:** Keep current loss when targeting probabilities.
- **BCE with logits:** Switch to a binary cross-entropy loss if you interpret the output as logits.
- **Calibration checks:** Add reliability diagrams/expected calibration error to ensure probabilistic outputs are trustworthy.

You can combine these ideas based on guidance—for example, start with quantile seeding + full grid for baseline, then explore sparse rule sets or two-stage optimisation if you hit capacity or overfitting issues.
