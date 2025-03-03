from feature_scaling_cpy import x


def z_score_normalize(data):
    mean_val = sum(data) / len(data)
    variance = sum((x - mean_val) ** 2 for x in data) / len(data)
    std_dev = variance**0.5
    return [(x - mean_val) / std_dev for x in data]


def main():
    data = x
    normalized_data = z_score_normalize(data)
    export_string = f"x = {normalized_data}\n"
    with open("manually_normalized_x.py", "w+") as f:
        f.write(export_string)

    print("Input:")
    print(x)

    print("\nZ-score normalized:")
    print(normalized_data)


if __name__ == "__main__":
    main()
