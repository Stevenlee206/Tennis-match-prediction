# Predictive Coding (PC) Model Report

## 1. Feature Selection Decision
**Correlation vs. Non-linear Models**
It was decided to stop filtering features based on their individual correlation (`corr`) with the target. 
- **Reasoning**: Correlation only measures independent, linear relationships. In tennis predictions, features often gain predictive power through complex, non-linear interactions (e.g., player fatigue combined with court surface). Filtering by top-K correlation risks discarding crucial features that our non-linear models (Predictive Coding, DeepForest, Random Forest) could effectively leverage.
- **Action Taken**: We now keep all non-leaky features, removing only those with zero variance across the training set.

## 2. Data Augmentation
- **Strategy**: Data augmentation (e.g., swapping Player 1 and Player 2 features) is strictly applied *only* to the training set. This effectively doubles the training data and prevents positional bias. Validation and test sets are kept entirely pure (unaugmented) to ensure realistic evaluation.

## 3. Hardware Optimization & Batch Size constraints
- **GPU Parallelization in PC**: Unlike standard neural networks utilizing Backpropagation, Predictive Coding relies on iterative local updates (Hebbian-like learning) to minimize prediction errors across hidden states. While PyTorch is used for faster tensor operations, arbitrarily increasing batch sizes on the GPU may not yield the same parallelization efficiency as standard Deep Learning due to the state-clamping and iterative loop inherent in PC inference. We will tune batch sizes carefully without forcing them to be overly large.
