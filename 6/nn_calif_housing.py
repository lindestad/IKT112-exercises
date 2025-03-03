import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim

# Load the dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
Y = pd.DataFrame(housing.target, columns=["MedianHouseValue"])

# Split into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=6969
)

# Standardize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32)


# Define the neural network model
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super(MLP, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU())

        # Hidden layer 1
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())

        # Hidden layer 2
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_size, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Instantiate the model, loss function, and optimizer
input_dim = X_train.shape[1]
output_dim = Y_train.shape[1]
hidden_size = 64  # Size of hidden layers
model = MLP(input_dim, output_dim, hidden_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Enable batch size
batch_size = 128
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

# Train the model
best_loss = float("inf")
best_model_state = None  # Store the best model weights
epochs = 100
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass and update weights
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)

    # Compute average training loss
    avg_loss = epoch_loss / len(train_loader.dataset)

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        preds_test_t = model(X_test_tensor)
        val_loss = criterion(preds_test_t, Y_test_tensor).item()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}], Train loss: {avg_loss:.4f}   Validation Loss: {val_loss:.4f}"
        )

    model.train()  # Switch back to training mode

    # Check for improvement
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_state = model.state_dict()  # Save the best model state

    # Restore best model before evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)


# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, Y_test_tensor)
    print(f"\nNeural Network Test MSE: {test_loss.item():.4f}")

# Compare to naive linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train.values)  # Fit on scaled data

# Make predictions and compute the mean squared error
lr_predictions = lin_reg.predict(X_test)
lr_test_mse = mean_squared_error(Y_test, lr_predictions)
print(f"Linear Regression Test MSE (closed-form): {lr_test_mse:.4f}")
