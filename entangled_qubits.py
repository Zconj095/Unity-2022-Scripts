import numpy as np
import math

def entangled_qubits():
    q1 = qubit()
    q2 = qubit()

    q1.apply_hadamard()
    q2.apply_hadamard()

    q1.apply_pauli_x()
    q2.apply_pauli_x()

    q1.apply_cnot(q2)

    return q1, q2

# Example usage:
q1, q2 = entangled_qubits()
print(q1.print_state())  # Qubit state: 0.71+0.71|0+|1
print(q2.print_state())  # Qubit state: 0.71+0.71|0+|1