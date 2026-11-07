# Compact Residual CNN for CIFAR-10 (≤ 500k parameters)

## Design decisions
- Hard limit: ≤ 500 000 parameters.
- Goal: ≥ 85 % test accuracy from scratch so a basic sequential CNN cannot outperform it.
- Architecture: stem + residual depthwise/pointwise mixer blocks + Global Average Pooling.
  - Residuals → stable training.
  - Depthwise + 1×1 → parameter efficiency.
  - GAP → almost zero classifier parameters.
- All spatial mixing uses standard convolutional kernels (3×3 / 5×5 / 7×7).
- No PyTorch – pure NumPy definition + TensorFlow/Keras training code provided.