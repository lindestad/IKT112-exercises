from feature_scaling_cpy import x, min_max_scale
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def main():
    x_mine = min_max_scale(x)
    x_mine_np_arr = np.array(x_mine).reshape(
        -1, 1
    )  # .reshape() to comply with MinMaxScaler expected format

    x_np = np.array(x).reshape(
        -1, 1
    )  # .reshape() to comply with MinMaxScaler expected format

    scaler = MinMaxScaler()
    x_sklearn = scaler.fit_transform(x_np)

    print("Original data (as numpy array):")
    print(x_np)

    print("\nMy transformation:")
    print(x_mine_np_arr)

    print("\nTransformation with Sklearn MinMaxScaler:")
    print(x_sklearn)

    if x_mine_np_arr.all() == x_sklearn.all():
        print("\nWe got the exact same result.")
    else:
        print("\nWe did not get the exact same result.")


if __name__ == "__main__":
    main()
