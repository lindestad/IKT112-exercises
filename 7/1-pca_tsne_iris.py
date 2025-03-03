import os
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def main():
    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # Apply PCA to reduce data to 2 dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Apply t-SNE to reduce data to 2 dimensions
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Save the PCA plot separately for later comparison.
    os.makedirs("img", exist_ok=True)
    fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
    for i, target in enumerate(target_names):
        ax_pca.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=target)
    ax_pca.set_title("Iris Dataset - PCA")
    ax_pca.set_xlabel("Component 1")
    ax_pca.set_ylabel("Component 2")
    ax_pca.legend()
    fig_pca.savefig("img/iris_pca.png")
    plt.close(fig_pca)

    # Create subplots to display PCA and t-SNE side by side.
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Left: PCA
    for i, target in enumerate(target_names):
        axs[0].scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=target)
    axs[0].set_title("Iris Dataset - PCA")
    axs[0].set_xlabel("Component 1")
    axs[0].set_ylabel("Component 2")
    axs[0].legend()

    # Right: t-SNE
    for i, target in enumerate(target_names):
        axs[1].scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=target)
    axs[1].set_title("Iris Dataset - t-SNE")
    axs[1].set_xlabel("Dimension 1")
    axs[1].set_ylabel("Dimension 2")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
