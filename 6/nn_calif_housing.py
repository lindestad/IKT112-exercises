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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

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
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Instantiate the model, loss function, and optimizer
input_dim = X_train.shape[1]
output_dim = Y_train.shape[1]
hidden_size = 128  # Size of hidden layers
model = MLP(input_dim, output_dim, hidden_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor)

    # Backward pass and update weights
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

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
