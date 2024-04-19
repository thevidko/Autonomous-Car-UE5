import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Load data from CSV file
data = pd.read_csv("received_messages.csv")
inputs = data.iloc[:, :7].values  # First 5 columns as inputs
outputs = data.iloc[:, 7:].values  # Last 2 columns as outputs

# Convert data to PyTorch tensors
inputs = torch.from_numpy(inputs).float()
outputs = torch.from_numpy(outputs).float()

# Define the model
class MyModel(nn.Module):
   def __init__(self):
       super(MyModel, self).__init__()
       self.linear1 = nn.Linear(7, 16)  # Input layer with 5 neurons, first hidden layer with 16
       self.linear2 = nn.Linear(16, 16)  # Second hidden layer with 16
       self.linear3 = nn.Linear(16, 16)  # Third hidden layer with 16
       self.linear4 = nn.Linear(16, 2)  # Output layer with 2 neurons

   def forward(self, x):
       x = torch.relu(self.linear1(x))  # ReLU activation function
       x = torch.relu(self.linear2(x))  # ReLU activation function
       x = torch.relu(self.linear3(x))  # ReLU activation function
       x = self.linear4(x)
       return x

# Create model instance
model = MyModel()

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean squared error loss
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer with learning rate 0.01

# Training loop
num_epochs = 2000  # Number of training iterations
for epoch in range(num_epochs):
   # Forward pass
   outputs_pred = model(inputs)
   loss = criterion(outputs_pred, outputs)

   # Backward pass and optimization
   optimizer.zero_grad()  # Reset gradients
   loss.backward()  # Calculate gradients
   optimizer.step()  # Update model parameters

   # Print loss every 10 epochs
   if (epoch + 1) % 10 == 0:
       print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "trained_model.pt")
