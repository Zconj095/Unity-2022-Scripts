from ClaudeProject1 import *
normal_mitochondria = Mitochondria(membrane_potential=-150)
altered_mitochondria = Mitochondria(membrane_potential=-100)

normal_atp = normal_mitochondria.produce_ATP()
altered_atp = altered_mitochondria.produce_ATP()

print(f"Normal ATP production: {normal_atp} units")
print(f"Altered ATP production: {altered_atp} units")

input_current = 5  # Arbitrary unit of current

normal_neuron = Neuron()
altered_neuron = Neuron()

# Simulate the response to the same input current under different ATP levels
normal_neuron.update(input_current * normal_atp / 100)
altered_neuron.update(input_current * altered_atp / 100)

print(f"Voltage of normal neuron: {normal_neuron.voltage} mV")
print(f"Voltage of altered neuron: {altered_neuron.voltage} mV")

sequences = [[normal_neuron.voltage for _ in range(5)], [altered_neuron.voltage for _ in range(5)]]

learning_model = ContinuityEncoder(sequence_len=5)
learning_model.train(sequences)

prediction_normal = learning_model.predict([normal_neuron.voltage])
prediction_altered = learning_model.predict([altered_neuron.voltage])

print(f"Learning prediction for normal neuron: {prediction_normal}")
print(f"Learning prediction for altered neuron: {prediction_altered}")

from keras.models import Sequential
from keras.layers import LSTM, Dense

class ContinuityEncoder:
    
    def __init__(self, sequence_len=5):
        self.model = Sequential()
        self.model.add(LSTM(32, input_shape=(sequence_len, 1)))
        self.model.add(Dense(sequence_len))
        # Compiling the model with an optimizer and loss function
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def train(self, sequences):
        # Assuming sequences is a list of lists and needs to be converted to a numpy array
        import numpy as np
        sequences = np.array(sequences).reshape((len(sequences), len(sequences[0]), 1))
        self.model.fit(sequences, sequences, epochs=10)
        
    def predict(self, seq_start):
        # Reshaping seq_start for prediction
        import numpy as np
        seq_start = np.array(seq_start).reshape((1, len(seq_start), 1))
        return self.model.predict(seq_start)
