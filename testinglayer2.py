from testing import *  # type: ignore
import cupy as cp
import torch
import torch.nn as nn
import torch.optim as optim

# Define the nodes and their states
nodes = {
    'A': ['True', 'False'],
    'B': ['True', 'False'],
    'C': ['True', 'False']
}

# Initialize the CPTs
cpt_A = cp.array([0.6, 0.4])  # P(A)
cpt_B_given_A = cp.array([[0.8, 0.2], [0.3, 0.7]])  # P(B|A)
cpt_C_given_B = cp.array([[0.9, 0.1], [0.4, 0.6]])  # P(C|B)

# Sampling from the Network
def sample_from_network(cpt_A, cpt_B_given_A, cpt_C_given_B, num_samples):
    samples = []
    for _ in range(num_samples):
        # Sample A
        A = cp.random.choice([0, 1], size=1, p=cpt_A).item()
        
        # Sample B given A
        B = cp.random.choice([0, 1], size=1, p=cpt_B_given_A[A]).item()
        
        # Sample C given B
        C = cp.random.choice([0, 1], size=1, p=cpt_C_given_B[B]).item()
        
        samples.append((A, B, C))
    
    return cp.array(samples)

# Generate 100 samples
num_samples = 100
samples = sample_from_network(cpt_A, cpt_B_given_A, cpt_C_given_B, num_samples)

# Convert samples to PyTorch tensors and move to CPU
samples_torch = torch.tensor(samples.get(), dtype=torch.float32)

# Split into input and output
X = samples_torch[:-1]
Y = samples_torch[1:]

# Reshape for LSTM input (batch_size, seq_length, input_size)
X = X.unsqueeze(1)
Y = Y.unsqueeze(1)
import torch
import torch.nn as nn
import torch.optim as optim

class ModularNetwork(nn.Module):
    def __init__(self):
        super(ModularNetwork, self).__init__()
        self.modules_list = nn.ModuleList()

    def add_module(self, module):
        self.modules_list.append(module)

    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x

class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), self.hidden_size)
        c_0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h_0, c_0))
        return out

class DenseLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(DenseLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

class DropoutLayer(nn.Module):
    def __init__(self, dropout_rate):
        super(DropoutLayer, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(x)

class RecursiveModularNetwork(ModularNetwork):
    def __init__(self):
        super(RecursiveModularNetwork, self).__init__()

    def add_recursive_module(self, input_size, hidden_size, depth, num_layers=1):
        if depth > 0:
            lstm_layer = LSTMLayer(input_size, hidden_size, num_layers)
            self.add_module(lstm_layer)
            self.add_recursive_module(hidden_size, hidden_size, depth - 1, num_layers)

# Initialize the network
input_size = 3
hidden_size = 50
output_size = 3
depth = 2
num_layers = 1

model = RecursiveModularNetwork()
model.add_recursive_module(input_size, hidden_size, depth, num_layers)
model.add_module(DenseLayer(hidden_size, output_size))
model.add_module(DropoutLayer(0.5))

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

import cupy as cp

# Define the nodes and their states
nodes = {
    'A': ['True', 'False'],
    'B': ['True', 'False'],
    'C': ['True', 'False']
}

# Initialize the CPTs
cpt_A = cp.array([0.6, 0.4])  # P(A)
cpt_B_given_A = cp.array([[0.8, 0.2], [0.3, 0.7]])  # P(B|A)
cpt_C_given_B = cp.array([[0.9, 0.1], [0.4, 0.6]])  # P(C|B)

# Sampling from the Network
def sample_from_network(cpt_A, cpt_B_given_A, cpt_C_given_B, num_samples):
    samples = []
    for _ in range(num_samples):
        # Sample A
        A = cp.random.choice([0, 1], size=1, p=cpt_A).item()
        
        # Sample B given A
        B = cp.random.choice([0, 1], size=1, p=cpt_B_given_A[A]).item()
        
        # Sample C given B
        C = cp.random.choice([0, 1], size=1, p=cpt_C_given_B[B]).item()
        
        samples.append((A, B, C))
    
    return cp.array(samples)

# Generate 100 samples
num_samples = 100
samples = sample_from_network(cpt_A, cpt_B_given_A, cpt_C_given_B, num_samples)

# Convert samples to PyTorch tensors and move to CPU
samples_torch = torch.tensor(samples.get(), dtype=torch.float32)

# Split into input and output
X = samples_torch[:-1]
Y = samples_torch[1:]

# Reshape for LSTM input (batch_size, seq_length, input_size)
X = X.unsqueeze(1)
Y = Y.unsqueeze(1)

# Train the model
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    outputs = model(X)
    loss = criterion(outputs, Y.squeeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
def evaluate_model(model, X, Y):
    model.eval()
    with torch.no_grad():
        predictions = model(X)
        mse = criterion(predictions, Y.squeeze(1)).item()
    return mse

mse = evaluate_model(model, X, Y)
print(f'Model Evaluation MSE: {mse:.4f}')

# Save the model
torch.save(model.state_dict(), 'recursive_modular_model.pth')
print('Model saved to recursive_modular_model.pth')

# Load the model
def load_model(path, input_size, hidden_size, output_size, depth, num_layers):
    model = RecursiveModularNetwork()
    model.add_recursive_module(input_size, hidden_size, depth, num_layers)
    model.add_module(DenseLayer(hidden_size, output_size))
    model.add_module(DropoutLayer(0.5))
    model.load_state_dict(torch.load(path))
    return model

loaded_model = load_model('recursive_modular_model.pth', input_size, hidden_size, output_size, depth, num_layers)
print('Model loaded from recursive_modular_model.pth')
