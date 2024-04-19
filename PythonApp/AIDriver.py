import threading
import torch
import torch.nn as nn
import torch.optim as optim
import socket
import pandas as pd

# Model definition (same as previous code)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = nn.Linear(7, 16)  # Input layer with 5 neurons, first hidden layer with 16
        self.linear2 = nn.Linear(16, 16)  # Second hidden layer with 16
        self.linear3 = nn.Linear(16, 16)  # Third hidden layer with 16
        self.linear4 = nn.Linear(16, 2)  # Output layer with 2 neurons

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = self.linear4(x)
        return x

# Function to load the model
def load_model(path):
    model = MyModel()
    model.load_state_dict(torch.load(path))
    model.eval()  # Set model to evaluation mode
    return model

# Function to handle client requests
def handle_client(conn, model):
    while True:
        try:
            # Receive data from client
            data = conn.recv(1024).decode()
            if not data:
                break

            # Convert data to tensor
            inputs = [float(val) for val in data.split(',')]
            print(inputs)
            metadata = pd.to_numeric(inputs)
            data = torch.from_numpy(metadata).float()

            # Predict using the model
            output = model(data.unsqueeze(0))

            # Access the desired output value
            output_value1 = output[0, 0].item()  # Selects first element from batch (0) and first output (0)
            output_value2 = output[0, 1].item()
            output_string = str(output_value1) + "," + str(output_value2)

            # Send prediction back to the client
            conn.sendall(output_string.encode())

        except Exception as e:
            print(f"Error handling client: {e}")
            break

    conn.close()

# Start the server
def start_server(model_path, server_port):
    # Load the model
    model = load_model(model_path)

    # Create server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("", server_port))
    server_socket.listen(1)

    print(f"Server listening on port {server_port}")

    while True:
        # Accept connection from client
        conn, addr = server_socket.accept()
        print(f"Connected by {addr}")

        # Handle client in a separate thread
        thread = threading.Thread(target=handle_client, args=(conn, model))
        thread.start()

# Program arguments
model_path = "trained_model.pt"  # Replace with your model path
server_port = 65432  # Replace with your desired server port

# Start the server
start_server(model_path, server_port)
