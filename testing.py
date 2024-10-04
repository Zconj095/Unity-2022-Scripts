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

# Inference: Calculate P(C|A=True)
def calculate_probability(cpt_A, cpt_B_given_A, cpt_C_given_B, evidence):
    P_A = cpt_A if evidence['A'] is None else cp.array([1.0, 0.0]) if evidence['A'] == 'True' else cp.array([0.0, 1.0])
    P_B_given_A = cpt_B_given_A
    P_C_given_B = cpt_C_given_B
    
    # Calculate P(B|A=True)
    P_B_given_A_true = P_B_given_A[0, :]
    
    # Calculate P(C|B)
    P_C_given_B_true = P_C_given_B[0, :]
    P_C_given_B_false = P_C_given_B[1, :]
    
    # Calculate P(C|A=True)
    P_C_given_A_true = P_B_given_A_true[0] * P_C_given_B_true + P_B_given_A_true[1] * P_C_given_B_false
    
    return P_C_given_A_true

# Define the evidence
evidence = {'A': 'True', 'B': None, 'C': None}

# Calculate the probability
P_C_given_A_true = calculate_probability(cpt_A, cpt_B_given_A, cpt_C_given_B, evidence)
print(f"P(C|A=True): {P_C_given_A_true}")

import cupy as cp

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

# Generate 10 samples
num_samples = 10
samples = sample_from_network(cpt_A, cpt_B_given_A, cpt_C_given_B, num_samples)
print("Samples:\n", samples)

def update_cpts(samples):
    num_samples = samples.shape[0]
    
    # Update CPT for A
    count_A = cp.sum(samples[:, 0])
    cpt_A_updated = cp.array([count_A / num_samples, (num_samples - count_A) / num_samples])
    
    # Update CPT for B given A
    count_B_given_A_true = cp.sum(samples[samples[:, 0] == 0][:, 1])
    count_B_given_A_false = cp.sum(samples[samples[:, 0] == 1][:, 1])
    cpt_B_given_A_updated = cp.array([
        [count_B_given_A_true / cp.sum(samples[:, 0] == 0), (cp.sum(samples[:, 0] == 0) - count_B_given_A_true) / cp.sum(samples[:, 0] == 0)],
        [count_B_given_A_false / cp.sum(samples[:, 0] == 1), (cp.sum(samples[:, 0] == 1) - count_B_given_A_false) / cp.sum(samples[:, 0] == 1)]
    ])
    
    # Update CPT for C given B
    count_C_given_B_true = cp.sum(samples[samples[:, 1] == 0][:, 2])
    count_C_given_B_false = cp.sum(samples[samples[:, 1] == 1][:, 2])
    cpt_C_given_B_updated = cp.array([
        [count_C_given_B_true / cp.sum(samples[:, 1] == 0), (cp.sum(samples[:, 1] == 0) - count_C_given_B_true) / cp.sum(samples[:, 1] == 0)],
        [count_C_given_B_false / cp.sum(samples[:, 1] == 1), (cp.sum(samples[:, 1] == 1) - count_C_given_B_false) / cp.sum(samples[:, 1] == 1)]
    ])
    
    return cpt_A_updated, cpt_B_given_A_updated, cpt_C_given_B_updated

# Update CPTs with generated samples
cpt_A_updated, cpt_B_given_A_updated, cpt_C_given_B_updated = update_cpts(samples)
print("Updated CPTs:\n", cpt_A_updated, "\n", cpt_B_given_A_updated, "\n", cpt_C_given_B_updated)

def compute_marginals(cpt_A, cpt_B_given_A, cpt_C_given_B):
    # P(A)
    P_A = cpt_A
    
    # P(B)
    P_B = cp.sum(cpt_B_given_A.T * P_A, axis=1)
    
    # P(C)
    P_C = cp.sum(cpt_C_given_B.T * P_B, axis=1)
    
    return P_A, P_B, P_C

# Compute marginal probabilities
P_A, P_B, P_C = compute_marginals(cpt_A_updated, cpt_B_given_A_updated, cpt_C_given_B_updated)
print("Marginal Probabilities:\nP(A):", P_A, "\nP(B):", P_B, "\nP(C):", P_C)

def condition_on_evidence(cpt_A, cpt_B_given_A, cpt_C_given_B, evidence):
    if 'A' in evidence:
        if evidence['A'] == 'True':
            cpt_A = cp.array([1.0, 0.0])
        elif evidence['A'] == 'False':
            cpt_A = cp.array([0.0, 1.0])
    
    if 'B' in evidence:
        if evidence['B'] == 'True':
            cpt_B_given_A[:, 0] = 1.0
            cpt_B_given_A[:, 1] = 0.0
        elif evidence['B'] == 'False':
            cpt_B_given_A[:, 0] = 0.0
            cpt_B_given_A[:, 1] = 1.0
    
    if 'C' in evidence:
        if evidence['C'] == 'True':
            cpt_C_given_B[:, 0] = 1.0
            cpt_C_given_B[:, 1] = 0.0
        elif evidence['C'] == 'False':
            cpt_C_given_B[:, 0] = 0.0
            cpt_C_given_B[:, 1] = 1.0
    
    return cpt_A, cpt_B_given_A, cpt_C_given_B

# Example: Condition on B=True
evidence = {'B': 'True'}
cpt_A_conditioned, cpt_B_given_A_conditioned, cpt_C_given_B_conditioned = condition_on_evidence(cpt_A, cpt_B_given_A, cpt_C_given_B, evidence)
print("Conditioned CPTs:\n", cpt_A_conditioned, "\n", cpt_B_given_A_conditioned, "\n", cpt_C_given_B_conditioned)

def calculate_joint_probability(cpt_A, cpt_B_given_A, cpt_C_given_B, values):
    A, B, C = values
    
    P_A = cpt_A[A]
    P_B_given_A = cpt_B_given_A[A, B]
    P_C_given_B = cpt_C_given_B[B, C]
    
    joint_probability = P_A * P_B_given_A * P_C_given_B
    return joint_probability

# Example: Calculate P(A=True, B=True, C=True)
values = (0, 0, 0)
joint_probability = calculate_joint_probability(cpt_A, cpt_B_given_A, cpt_C_given_B, values)
print("Joint Probability P(A=True, B=True, C=True):", joint_probability)

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

def mrf_energy(samples, cpt_A, cpt_B_given_A, cpt_C_given_B):
    energy = 0
    for sample in samples:
        A, B, C = sample
        energy -= cp.log(cpt_A[A])
        energy -= cp.log(cpt_B_given_A[A, B])
        energy -= cp.log(cpt_C_given_B[B, C])
    return energy

energy = mrf_energy(samples, cpt_A, cpt_B_given_A, cpt_C_given_B)
print("MRF Energy:", energy)

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), hidden_size)
        c_0 = torch.zeros(1, x.size(0), hidden_size)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

input_size = 3  # Number of nodes
hidden_size = 50
output_size = 3  # Number of nodes
num_layers = 1

model = LSTMNetwork(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Convert samples to PyTorch tensors and move to CPU
samples_torch = torch.tensor(samples.get(), dtype=torch.float32)

# Split into input and output
X = samples_torch[:-1]
Y = samples_torch[1:]

# Reshape for LSTM input (batch_size, seq_length, input_size)
X = X.unsqueeze(1)
Y = Y.unsqueeze(1)

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    outputs = model(X)
    loss = criterion(outputs, Y.squeeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    test_sample = X[0].unsqueeze(0)  # Take the first sample for testing
    prediction = model(test_sample)
    print("Prediction:", prediction)

def evaluate_model(model, X, Y):
    model.eval()
    with torch.no_grad():
        predictions = model(X)
        mse = criterion(predictions, Y.squeeze(1)).item()
    return mse

# Evaluate the model
mse = evaluate_model(model, X, Y)
print(f'Model Evaluation MSE: {mse:.4f}')

torch.save(model.state_dict(), 'lstm_mrf_model.pth')
print('Model saved to lstm_mrf_model.pth')

def load_model(path, input_size, hidden_size, output_size, num_layers):
    model = LSTMNetwork(input_size, hidden_size, output_size, num_layers)
    model.load_state_dict(torch.load(path))
    return model

# Load the model
loaded_model = load_model('lstm_mrf_model.pth', input_size, hidden_size, output_size, num_layers)
print('Model loaded from lstm_mrf_model.pth')

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

# Define the LSTM-based ANN
class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), hidden_size)
        c_0 = torch.zeros(1, x.size(0), hidden_size)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

input_size = 3  # Number of nodes
hidden_size = 50
output_size = 3  # Number of nodes
num_layers = 1

model = LSTMNetwork(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert samples to PyTorch tensors and move to CPU
samples_torch = torch.tensor(samples.get(), dtype=torch.float32)

# Split into input and output
X = samples_torch[:-1]
Y = samples_torch[1:]

# Reshape for LSTM input (batch_size, seq_length, input_size)
X = X.unsqueeze(1)
Y = Y.unsqueeze(1)

# Train the Model
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    outputs = model(X)
    loss = criterion(outputs, Y.squeeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the Model
mse = evaluate_model(model, X, Y)
print(f'Model Evaluation MSE: {mse:.4f}')

# Save the Model
torch.save(model.state_dict(), 'lstm_mrf_model.pth')
print('Model saved to lstm_mrf_model.pth')

# Load the Model
loaded_model = load_model('lstm_mrf_model.pth', input_size, hidden_size, output_size, num_layers)
print('Model loaded from lstm_mrf_model.pth')

