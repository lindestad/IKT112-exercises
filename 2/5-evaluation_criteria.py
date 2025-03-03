true_labels = [
    "dog",
    "cat",
    "dog",
    "cat",
    "cat",
    "dog",
    "dog",
    "cat",
    "dog",
    "cat",
    "dog",
    "dog",
    "cat",
    "dog",
    "cat",
    "cat",
    "dog",
    "dog",
    "cat",
    "dog",
]
predicted_labels = [
    "dog",
    "dog",
    "dog",
    "cat",
    "cat",
    "cat",
    "dog",
    "dog",
    "cat",
    "cat",
    "dog",
    "dog",
    "cat",
    "cat",
    "cat",
    "dog",
    "dog",
    "cat",
    "cat",
    "dog",
]


def accuracy(prediction: list, ground_truth: list) -> float:
    size = len(prediction)
    assert size == len(ground_truth) and size != 0
    correct = 0
    for p, truth in zip(prediction, ground_truth):
        correct += 1 * (p == truth)
    return correct / size


def precision(prediction: list, ground_truth: list, positive_label: str) -> float:
    assert len(prediction) == len(ground_truth)
    true_positives = 0
    false_positives = 0
    for p, truth in zip(prediction, ground_truth):
        if p == positive_label:
            if p == truth:
                true_positives += 1
            else:
                false_positives += 1
    if true_positives + false_positives == 0:
        return 0.0
    return true_positives / (true_positives + false_positives)


def recall(prediction: list, ground_truth: list, positive_label: str) -> float:
    assert len(prediction) == len(ground_truth)
    true_positives = 0
    false_negatives = 0
    for p, truth in zip(prediction, ground_truth):
        if truth == positive_label:
            if p == positive_label:
                true_positives += 1
            else:
                false_negatives += 1
    if true_positives + false_negatives == 0:
        return 0.0
    return true_positives / (true_positives + false_negatives)


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * ((precision * recall) / (precision + recall))


def calculate_f1_score(
    prediction: list, ground_truth: list, positive_label: str
) -> float:
    p = precision(prediction, ground_truth, positive_label)
    r = recall(prediction, ground_truth, positive_label)
    return f1_score(p, r)


def main():
    print("Task: Calculate the accuracy of the model:")
    print(
        f"The model accuracy is {accuracy(predicted_labels, true_labels)*100:.2f}%.\n"
    )

    print("Task: Calculate the recall for 'dog':")
    r = recall(predicted_labels, true_labels, "dog")
    l = len(predicted_labels)
    print(f"The recall for dog is {r*100:.2f}%, or {round(r*l)}/{l}.\n")

    print("Task: Calculate precision for 'cat':")
    p = precision(predicted_labels, true_labels, "cat")
    print(f"The precision for cat is {p*100:.2f}%, or {round(p*l)}/{l}.\n")

    print("Task: Calculate the F1 Score for the model for both 'cat' and 'dog'.")
    print(
        f"F1 score for 'dog': {calculate_f1_score(predicted_labels, true_labels, "dog"):.4f}"
    )
    print(
        f"F1 score for 'cat': {calculate_f1_score(predicted_labels, true_labels, "cat"):.4f}"
    )


if __name__ == "__main__":
    main()
