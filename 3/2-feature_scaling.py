from typing import List, Union

x = [12, 55, 74, 32, 89, 26, 37, 45, 68, 99, 22, 50, 73, 85, 17, 31, 52, 47, 65, 90]


def min_max_scale(data: List[Union[int, float]]) -> List[float]:
    min_val = min(data)
    max_val = max(data)
    scaled = [(x - min_val) / (max_val - min_val) for x in data]
    return scaled


def main():
    print("Input values:")
    print(x)
    print(f"\nScaled values:")
    print(min_max_scale(x))


if __name__ == "__main__":
    main()
