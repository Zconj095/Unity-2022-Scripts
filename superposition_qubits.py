import numpy as np
import math

def superposition_qubits():
    q1 = qubit()
    q2 = qubit()

    # Create a superposition of both qubits
    q1.state = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)], dtype=complex)
    q2.state = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)], dtype=complex)

    return q1, q2

# Example usage:
q1, q2 = superposition_qubits()
print(q1.print_state())  # Qubit state: 0.707106781+0.707106781|0+|1
print(q2.print_state())  # Qubit state: 0.707106781+0.707106781|0+|1