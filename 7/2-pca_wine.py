import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA


def main():
    # Load the Wine dataset
    wine = load_wine()
    X = wine.data
    y = wine.target
    target_names = wine.target_names

    # Brief exploration of the dataset
    print("Wine dataset shape:", X.shape)
    print("Classes:", target_names)

    # Apply PCA to reduce data to 2 dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Plot the PCA results
    plt.figure(figsize=(8, 6))
    for i, target in enumerate(target_names):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=target)
    plt.title("Wine Dataset - PCA")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
