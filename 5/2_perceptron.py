import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# Rocks get class 0
rocks = [[random.uniform(-5, 1), random.uniform(-5, 1), 0] for i in range(100)]
# Not rocks get class 1
not_rocks = [[random.uniform(-1, 5), random.uniform(-1, 5), 1] for i in range(100)]


def train_test(data: list, split: float) -> tuple[list, list]:
    """
    Splits data into train and test at _split_.
    Example: split = 0.80. Train is the first 80%. Test is the remainder 20%.
    """
    size = len(data)
    idx = round(size * split)
    return (data[0:idx], data[idx::])


# Initialize weights and other parameters
weights = [random.random() for i in range(3)]
bias = 1
learning_rate = 0.01
epochs = 100  # Number of iterations over the training dataset


def step_activation(x: float) -> int:
    """Simple step function: returns 1 if x is >= 0, else 0."""
    return 1 if x >= 0 else 0


def train_perceptron(
    training_data: list, weights: list, bias: float, learning_rate: float, epochs: int
) -> list:
    """
    Trains a simple perceptron using a step activation function.
    Each sample in training_data is assumed to be of the form [x1, x2, label].
    """
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        for sample in training_data:
            x1, x2, label = sample
            # Compute weighted sum, weights[2] works as the weight for the bias
            weighted_sum = weights[0] * x1 + weights[1] * x2 + weights[2] * bias
            prediction = step_activation(weighted_sum)
            error = label - prediction
            # Update the weights based on the error
            weights[0] += learning_rate * error * x1
            weights[1] += learning_rate * error * x2
            weights[2] += learning_rate * error * bias
    return weights


def test_perceptron(test_data: list, weights: list, bias: float) -> float:
    """
    Tests the perceptron on the test_data and returns the accuracy.
    Each sample is of the form [x1, x2, label].
    """
    correct = 0
    for sample in test_data:
        x1, x2, label = sample
        weighted_sum = weights[0] * x1 + weights[1] * x2 + weights[2] * bias
        prediction = step_activation(weighted_sum)
        if prediction == label:
            correct += 1
    return correct / len(test_data)


def plot_decision_boundary(weights: list, bias: float, data: list) -> None:
    """
    Plots the decision boundary defined by the perceptron along with the data points.
    The decision boundary is the line where the perceptron's weighted sum equals zero.
    """
    # Determine range for x values using the data
    all_x = [sample[0] for sample in data]
    x_min = min(all_x) - 1
    x_max = max(all_x) + 1
    xs = [x_min, x_max]

    # Calculate corresponding y values from the decision boundary equation:
    # weights[0]*x + weights[1]*y + weights[2]*bias = 0  -->  y = -(weights[0]*x + weights[2]*bias) / weights[1]
    if weights[1] != 0:
        ys = [-(weights[0] * x + weights[2] * bias) / weights[1] for x in xs]
        plt.plot(xs, ys, "k--", label="Decision Boundary")
    else:
        # If weights[1] is 0, the boundary is a vertical line.
        x_boundary = -weights[2] * bias / weights[0]
        plt.axvline(x=x_boundary, linestyle="--", color="k", label="Decision Boundary")


def visualize_data(data: list, weights: list, bias: float) -> None:
    """
    Visualizes the training data and the perceptron's decision boundary.
    Data points from different classes are plotted in different colors.
    """
    # Separate points by class
    rocks_points = [sample for sample in data if sample[2] == 0]
    not_rocks_points = [sample for sample in data if sample[2] == 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        [pt[0] for pt in rocks_points],
        [pt[1] for pt in rocks_points],
        color="red",
        label="Rocks (0)",
    )
    plt.scatter(
        [pt[0] for pt in not_rocks_points],
        [pt[1] for pt in not_rocks_points],
        color="blue",
        label="Not Rocks (1)",
    )
    plot_decision_boundary(weights, bias, data)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.title("Perceptron Decision Boundary")
    plt.show()


def main():
    split = 0.80
    train_rocks, test_rocks = train_test(rocks, split)
    train_not_rocks, test_not_rocks = train_test(not_rocks, split)

    # Combine training and testing sets
    train_data = train_rocks + train_not_rocks
    test_data = test_rocks + test_not_rocks

    # Shuffle data to avoid grouping examples
    random.shuffle(train_data)
    random.shuffle(test_data)

    # Train the perceptron model on the training data
    trained_weights = train_perceptron(train_data, weights, bias, learning_rate, epochs)

    # Evaluate the model on the testing data
    accuracy = test_perceptron(test_data, trained_weights, bias)
    print(f"Task 3:\nTesting Accuracy: {accuracy}\n")

    # Visualize the decision boundary along with the training data
    visualize_data(train_data, trained_weights, bias)


if __name__ == "__main__":
    main()
