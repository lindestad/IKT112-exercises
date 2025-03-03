# Theoretical Question: Why, and when do we use feature scaling?

Feature scaling is used to normalize the range of features so that no single feature dominates due to its scale. When working with distance-based algorithms or gradient descent, differences in magnitude among features can lead to biased results or slow convergence. By rescaling the features, each one contributes equally to the model's learning process. This is particularly important when features are measured in different units or have inherently different ranges.
