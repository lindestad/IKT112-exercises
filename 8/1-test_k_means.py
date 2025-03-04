from sklearn.datasets import make_blobs
from sklearn.cluster import k_means
import matplotlib.pyplot as plt


def main():
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)

    centroids, labels, inertia = k_means(X, n_clusters=3, random_state=0)

    print(X.shape)
    print(y.shape)
    # Plotting
    plt.figure(figsize=(12, 5))
    # Plot true labels
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("True Labels")

    # Plot k-means predicted labels
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    # Plot centroids as red X-es
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c="red", marker="X")
    plt.title("K-Means Predicted Labels")

    # Display the plots
    plt.show()


if __name__ == "__main__":
    main()
