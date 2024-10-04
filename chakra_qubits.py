from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import numpy as np

# Number of chakras and qubits assigned to each
num_chakras = 6  # Reduced to fit within limits
chakra_qubits = [3, 4, 5, 3, 4, 7]  # Adjusted qubit sizes to fit within backend limits

# Initialize quantum circuits for each chakra
circuits = []
for i in range(num_chakras):
    qc = QuantumCircuit(chakra_qubits[i])
    qc.h(range(chakra_qubits[i]))  # Simple initial state for each chakra (Hadamard on all qubits)
    circuits.append(qc)

# Combine circuits into a single circuit representing the full aura
aura_circuit = QuantumCircuit(sum(chakra_qubits))
current_index = 0
for qc in circuits:
    aura_circuit.append(qc.to_instruction(), range(current_index, current_index + qc.num_qubits))
    current_index += qc.num_qubits

# Apply interactions between chakras directly in the circuit
for i in range(num_chakras - 1):
    start_idx = sum(chakra_qubits[:i])
    aura_circuit.cz(start_idx, start_idx + chakra_qubits[i])

# Add measurements to all qubits
aura_circuit.measure_all()

# Use the qasm_simulator backend from AerSimulator
simulator = AerSimulator()

# Transpile the circuit for the simulator backend
compiled_circuit = transpile(aura_circuit, simulator)

# Assemble the circuit into a Qobj
qobj = assemble(compiled_circuit)

# Run the simulation and get the results (using shots)
result = simulator.run(qobj, shots=1024).result()

# Get the counts (measurement results)
counts = result.get_counts(compiled_circuit)

# Plot the measurement results
plot_histogram(counts)

