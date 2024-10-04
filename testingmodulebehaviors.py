from testinglayer2 import * # type: ignore
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

# Define the Modular Network
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

class ActivationLayer(nn.Module):
    def __init__(self, activation_function='relu'):
        super(ActivationLayer, self).__init__()
        if activation_function == 'relu':
            self.activation = nn.ReLU()
        elif activation_function == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_function == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, x):
        return self.activation(x)

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)

class AttentionLayer(nn.Module):
    def __init__(self, embed_size, heads):
        super(AttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads)

    def forward(self, x):
        x, _ = self.attention(x, x, x)
        return x

class BatchNormLayer(nn.Module):
    def __init__(self, num_features):
        super(BatchNormLayer, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)

    def forward(self, x):
        return self.batch_norm(x)

# Define the Recursive Modular Network
class RecursiveModularNetwork(ModularNetwork):
    def __init__(self):
        super(RecursiveModularNetwork, self).__init__()

    def add_recursive_module(self, input_size, hidden_size, depth, num_layers=1):
        if depth > 0:
            lstm_layer = LSTMLayer(input_size, hidden_size, num_layers)
            self.add_module(lstm_layer)
            self.add_recursive_module(hidden_size, hidden_size, depth - 1, num_layers)

    def remove_module(self, index):
        if 0 <= index < len(self.modules_list):
            del self.modules_list[index]

    def insert_module(self, index, module):
        if 0 <= index <= len(self.modules_list):
            self.modules_list.insert(index, module)

# Instantiate the network
input_size = 3
hidden_size = 50
output_size = 3
depth = 2
num_layers = 1

model = RecursiveModularNetwork()
model.add_recursive_module(input_size, hidden_size, depth, num_layers)
model.add_module(DenseLayer(hidden_size, output_size))
model.add_module(DropoutLayer(0.5))

# Add an activation layer
model.add_module(ActivationLayer('relu'))

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Custom training loop with early stopping
def custom_train(model, X, Y, criterion, optimizer, num_epochs, early_stopping_patience=5):
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X)
        loss = criterion(outputs, Y.squeeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check for early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model

# Train the model with custom training loop
model = custom_train(model, X, Y, criterion, optimizer, num_epochs=100, early_stopping_patience=5)

def evaluate_model(model, X, Y):
    model.eval()
    with torch.no_grad():
        predictions = model(X)
        mse = criterion(predictions, Y.squeeze(1)).item()
    return mse

# Evaluate the model
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
    model.add_module(ActivationLayer('relu'))
    model.load_state_dict(torch.load(path))
    return model

loaded_model = load_model('recursive_modular_model.pth', input_size, hidden_size, output_size, depth, num_layers)
print('Model loaded from recursive_modular_model.pth')

import cupy as cp
import torch
import torch.nn as nn
import torch.optim as optim
import time

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

# Define the Modular Network
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

class ActivationLayer(nn.Module):
    def __init__(self, activation_function='relu'):
        super(ActivationLayer, self).__init__()
        if activation_function == 'relu':
            self.activation = nn.ReLU()
        elif activation_function == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_function == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, x):
        return self.activation(x)

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)

class AttentionLayer(nn.Module):
    def __init__(self, embed_size, heads):
        super(AttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads)

    def forward(self, x):
        x, _ = self.attention(x, x, x)
        return x

class BatchNormLayer(nn.Module):
    def __init__(self, num_features):
        super(BatchNormLayer, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)

    def forward(self, x):
        return self.batch_norm(x)

# Define the Recursive Modular Network
class RecursiveModularNetwork(ModularNetwork):
    def __init__(self):
        super(RecursiveModularNetwork, self).__init__()
        self.max_health = 100
        self.health = 100
        self.status_conditions = []
        self.prevent_critical_hits = False
        self.lucky_chant_duration = 0
        self.is_raining = False
        self.effects = []

    def add_recursive_module(self, input_size, hidden_size, depth, num_layers=1):
        if depth > 0:
            lstm_layer = LSTMLayer(input_size, hidden_size, num_layers)
            self.add_module(lstm_layer)
            self.add_recursive_module(hidden_size, hidden_size, depth - 1, num_layers)

    def remove_module(self, index):
        if 0 <= index < len(self.modules_list):
            del self.modules_list[index]

    def insert_module(self, index, module):
        if 0 <= index <= len(self.modules_list):
            self.modules_list.insert(index, module)

    def add_effect(self, effect):
        self.effects.append(effect)

    def apply_effects(self):
        for effect in self.effects:
            effect.apply(self)

    def update_status(self):
        if self.lucky_chant_duration > 0:
            self.lucky_chant_duration -= 1
        if self.lucky_chant_duration == 0:
            self.prevent_critical_hits = False

    def log_status(self):
        print(f"Current Health: {self.health}/{self.max_health}")
        print(f"Status Conditions: {self.status_conditions}")
        print(f"Prevent Critical Hits: {self.prevent_critical_hits}")
        print(f"Is Raining: {self.is_raining}")
        print(f"Effects: {[effect.__class__.__name__ for effect in self.effects]}")

    def forward(self, x):
        self.apply_effects()
        self.update_status()
        self.log_status()
        return super().forward(x)

# Define Effects and Abilities
class Effect:
    def apply(self, network):
        pass

class RainDish(Effect):
    def __init__(self, health_replenish_rate):
        self.health_replenish_rate = health_replenish_rate

    def apply(self, network):
        if network.is_raining:
            network.health = min(network.max_health, network.health + self.health_replenish_rate)

class AquaRing(Effect):
    def apply(self, network):
        network.health = min(network.max_health, network.health + network.max_health / 16)

class Moonlight(Effect):
    def apply(self, network):
        current_hour = time.localtime().tm_hour
        moon_strength = 1.0 if 20 <= current_hour <= 4 else 0.5
        network.health = min(network.max_health, network.health + moon_strength * network.max_health / 8)

class Refresh(Effect):
    def apply(self, network):
        network.status_conditions = []

class MorningSun(Effect):
    def apply(self, network):
        current_hour = time.localtime().tm_hour
        if 6 <= current_hour <= 9:
            network.health = min(network.max_health, network.health + network.max_health / 4)

class LuckyChant(Effect):
    def __init__(self, duration):
        self.duration = duration

    def apply(self, network):
        network.prevent_critical_hits = True
        network.lucky_chant_duration = self.duration

# Custom training loop with status updates
def custom_train(model, X, Y, criterion, optimizer, num_epochs, early_stopping_patience=5, log_interval=10):
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X)
        loss = criterion(outputs, Y.squeeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check for early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break
        
        if (epoch + 1) % log_interval == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            model.log_status()

    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model

# Initialize and apply effects to the network
model = RecursiveModularNetwork()
model.add_recursive_module(input_size, hidden_size, depth, num_layers)
model.add_module(DenseLayer(hidden_size, output_size))
model.add_module(DropoutLayer(0.5))
model.add_module(ActivationLayer('relu'))

# Example effects
rain_dish = RainDish(health_replenish_rate=5)
aqua_ring = AquaRing()
moonlight = Moonlight()
refresh = Refresh()
morning_sun = MorningSun()
lucky_chant = LuckyChant(duration=10)

model.add_effect(rain_dish)
model.add_effect(aqua_ring)
model.add_effect(moonlight)
model.add_effect(refresh)
model.add_effect(morning_sun)
model.add_effect(lucky_chant)

# Train the model with custom training loop
model = custom_train(model, X, Y, criterion, optimizer, num_epochs=100, early_stopping_patience=5)

def evaluate_model(model, X, Y):
    model.eval()
    with torch.no_grad():
        predictions = model(X)
        mse = criterion(predictions, Y.squeeze(1)).item()
    return mse

# Evaluate the model
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
    model.add_module(ActivationLayer('relu'))
    model.load_state_dict(torch.load(path))
    return model

loaded_model = load_model('recursive_modular_model.pth', input_size, hidden_size, output_size, depth, num_layers)
print('Model loaded from recursive_modular_model.pth')

