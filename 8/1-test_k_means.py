from sklearn.datasets import make_blobs
from sklearn.cluster import k_means
import matplotlib.pyplot as plt


def main():
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)

    # Plotting
    plt.figure(figsize=(12, 9))

    # Plot true labels
    plt.subplot(2, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("True Labels")

    config = [
        {"K": 3, "plot_idx": 2},
        {"K": 4, "plot_idx": 3},
        {"K": 6, "plot_idx": 4},
    ]

    for c in config:
        centroids, labels, inertia = k_means(X, n_clusters=c["K"], random_state=0)

        # Plot k-means predicted labels
        plt.subplot(2, 2, c["plot_idx"])
        plt.scatter(X[:, 0], X[:, 1], c=labels)
        # Plot centroids as red X-es
        plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c="red", marker="X")
        plt.title(f"K-Means Predicted Labels for K = {c["K"]}.")

    # Display the plots
    plt.show()


if __name__ == "__main__":
    main()
