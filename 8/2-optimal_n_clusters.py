import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import k_means
from sklearn.metrics import silhouette_score
import numpy as np


def find_elbow_point(k_values, wcss):
    """
    Determine the elbow point as the k with the maximum distance from
    the line joining the first and last point of the WCSS curve.
    """
    # Convert to numpy arrays for vectorized operations
    k_values = np.array(k_values)
    wcss = np.array(wcss)

    # Line between first and last points
    point1 = np.array([k_values[0], wcss[0]])
    point2 = np.array([k_values[-1], wcss[-1]])

    # Compute distances from each point to the line
    # Using the distance from a point to a line formula
    numerator = np.abs(
        (point2[1] - point1[1]) * k_values
        - (point2[0] - point1[0]) * wcss
        + point2[0] * point1[1]
        - point2[1] * point1[0]
    )
    denominator = np.sqrt((point2[1] - point1[1]) ** 2 + (point2[0] - point1[0]) ** 2)
    distances = numerator / denominator

    # The elbow point is where the distance is maximized
    elbow_idx = np.argmax(distances)
    return k_values[elbow_idx], distances


def plot_metrics(X, k_range, ax_elbow, ax_silhouette):
    """
    Computes and plots the WCSS (Elbow Method) and Silhouette scores
    for a range of k values on the provided axes.
    """
    wcss = []
    silhouette_scores = []

    for k in k_range:
        centroids, labels, inertia = k_means(X, n_clusters=k, random_state=0)
        wcss.append(inertia)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)

    ax_elbow.plot(list(k_range), wcss, "gx-")
    ax_elbow.set_xlabel("Number of clusters (k)")
    ax_elbow.set_ylabel("WCSS")
    ax_elbow.set_title("Elbow Method")

    # Determine optimal k via Elbow Method
    optimal_k_elbow, distances = find_elbow_point(list(k_range), wcss)
    ax_elbow.axvline(
        optimal_k_elbow,
        color="red",
        linestyle="--",
        label=f"Optimal k = {optimal_k_elbow}",
    )
    ax_elbow.legend()

    ax_silhouette.plot(list(k_range), silhouette_scores, "bx-")
    ax_silhouette.set_xlabel("Number of clusters (k)")
    ax_silhouette.set_ylabel("Silhouette Score")
    ax_silhouette.set_title("Silhouette Score")

    # Determine optimal k via Silhouette Score (max score)
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    ax_silhouette.axvline(
        optimal_k_silhouette,
        color="red",
        linestyle="--",
        label=f"Optimal k = {optimal_k_silhouette}",
    )
    ax_silhouette.legend()


def main():
    # Define the range of k values to test
    k_range = range(2, 11)

    # Create two datasets: one with 4 clusters, one with 5 clusters
    X_4, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.5, random_state=0)
    X_5, _ = make_blobs(n_samples=300, centers=5, cluster_std=0.5, random_state=0)

    # Create a 2x2 grid for subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Plot for the dataset with 4 clusters
    plot_metrics(X_4, k_range, axes[0, 0], axes[0, 1])
    axes[0, 0].set_title("4 Clusters - Elbow Method")
    axes[0, 1].set_title("4 Clusters - Silhouette Score")

    # Plot for the dataset with 5 clusters
    plot_metrics(X_5, k_range, axes[1, 0], axes[1, 1])
    axes[1, 0].set_title("5 Clusters - Elbow Method")
    axes[1, 1].set_title("5 Clusters - Silhouette Score")

    plt.tight_layout()
    plt.show()

    print(
        "Bonus discussion:\n",
        'When the data becomes "harder" to cluster (e.g., clusters overlap),\n',
        "the elbow in the WCSS plot may not be as distinct. In such cases, the Silhouette score—which\n",
        "measures the separation distance between clusters—might provide a more robust indication of the optimal number of clusters.\n",
    )


if __name__ == "__main__":
    main()
