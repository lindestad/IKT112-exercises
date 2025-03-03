import random

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


def main():
    split = 0.80
    train_rocks, test_rocks = train_test(rocks, split)
    train_not_rocks, test_not_rocks = train_test(not_rocks, split)

    assert len(train_rocks) == 80 and len(test_rocks) == 20
    assert len(train_not_rocks) == 80 and len(test_not_rocks) == 20


if __name__ == "__main__":
    main()
