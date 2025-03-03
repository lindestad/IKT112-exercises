from manually_normalized_x import x as x_manual
from feature_scaling_cpy import x
from sklearn.preprocessing import StandardScaler
import numpy as np


def main():
    global x_manual  # Python complains without this due to scope rules
    global x
    scaler = StandardScaler()
    data = np.array(x).reshape(-1, 1)
    x_lib = scaler.fit_transform(data)
    x_manual = np.array(x_manual).reshape(-1, 1)

    print("Comparison between sklearn StandardScaler and my own implementation:")
    print("My result:")
    print(x_manual)

    print("\nSklearn StandardScaler result:")
    print(x_lib)

    if x_manual.all() == x_lib.all():
        print("\nWe got the exact same result.")
    else:
        print("\nWe did not get the exact same result.")


if __name__ == "__main__":
    main()
