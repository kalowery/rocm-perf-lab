# Autotune Mechanics

## Three-Phase Algorithm

1. Seed Phase
   - Profile subset
   - Fit polynomial regression

2. Prediction Phase
   - Build static feature vectors
   - Predict runtimes
   - Prune configs

3. Confirm Phase
   - Profile pruned configs
   - Select best runtime

---

## Static Features

Used for pruning only:

- ACC
- ACC²
- BLOCK_K
- num_warps
- num_stages
- threads_per_block

Runtime features are NOT used for pruning.

---

## Model Confidence

If R² < 0.75:

- Pruning reliability is low
- Warning is emitted
- Consider increasing seed fraction

---

## Best Practices

- Ensure kernel parameters actually affect runtime
- Avoid degenerate search spaces
- Validate stability before tuning
- Use JSON output for automation pipelines
