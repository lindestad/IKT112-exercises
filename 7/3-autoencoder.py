import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris

# Extremely sensitive to starting conditions. # 331472
SEED = 660036
RANDOM = False

EPOCHS = 400  # Keeps improving with more, but unclear when it's overfitting on the tiny dataset

if RANDOM:
    seed = np.random.random_integers(0, 1000000)
    print(f"Seed: {seed}")
else:
    seed = SEED

torch.manual_seed(seed)
np.random.seed(seed)

# 331472


# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim=4, encoding_dim=2):
        super(Autoencoder, self).__init__()
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, encoding_dim),
        )
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def main():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data.astype(np.float32)  # shape: (150, 4)
    y = iris.target
    target_names = iris.target_names

    # Convert the features to a PyTorch tensor.
    X_tensor = torch.tensor(X)

    # Instantiate the autoencoder model, loss function, and optimizer.
    model = Autoencoder(input_dim=X.shape[1], encoding_dim=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the autoencoder.
    epochs = EPOCHS
    for epoch in range(epochs):
        optimizer.zero_grad()
        reconstructed = model(X_tensor)
        loss = criterion(reconstructed, X_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Use the encoder part of the autoencoder to reduce the data to 2 dimensions.
    with torch.no_grad():
        encoded_data = model.encoder(X_tensor).numpy()

    # Create subplots to compare the Autoencoder and PCA results side by side.
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Left subplot: Autoencoder results.
    for i, label in enumerate(target_names):
        axs[0].scatter(encoded_data[y == i, 0], encoded_data[y == i, 1], label=label)
    axs[0].set_title("Iris Dataset - Autoencoder (PyTorch)")
    axs[0].set_xlabel("Encoded Feature 1")
    axs[0].set_ylabel("Encoded Feature 2")
    axs[0].legend()

    # Right subplot: Load the saved PCA plot.
    pca_plot_path = os.path.join("img", "iris_pca.png")
    if os.path.exists(pca_plot_path):
        pca_img = plt.imread(pca_plot_path)
        axs[1].imshow(pca_img)
        axs[1].set_title("Iris Dataset - PCA (from Task 1)")
        axs[1].axis("off")
    else:
        axs[1].text(
            0.5,
            0.5,
            "PCA plot not found.\nRun the PCA script to generate it.",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[1].transAxes,
        )
        axs[1].set_title("Iris Dataset - PCA (from Task 1)")
        axs[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
