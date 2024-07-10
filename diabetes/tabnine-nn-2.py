# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define the dataset
# In this example, we'll use a simple dataset for demonstration purposes
# Replace this with your own dataset
x_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Create an instance of the neural network model
model = NeuralNetwork()

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(10):
    # Forward pass
    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluate the model
with torch.no_grad():
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    accuracy = (outputs.round() == y_train).float().mean()
    print(f'Loss: {loss.item()}, Accuracy: {accuracy.item()}')