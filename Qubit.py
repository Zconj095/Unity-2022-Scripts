import numpy as np
import math

def qubit():
    def __init__(self):
        self.state = np.array([1.0, 0.0], dtype=complex)

    def apply_hadamard(self):
        self.state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)

    def apply_pauli_x(self):
        self.state = np.array([self.state[1], self.state[0]], dtype=complex)

    def apply_pauli_y(self):
        self.state = np.array([complex(0, 1)*self.state[0], complex(0, 1)*self.state[1]], dtype=complex)

    def apply_pauli_z(self):
        self.state = np.array([complex(np.exp(1j*np.pi), 0)*self.state[0], complex(0, np.exp(-1j*np.pi))*self.state[1]], dtype=complex)

    def apply_cnot(self, control_qubit):
        if np.abs(control_qubit.state[0]) ** 2 > 0.5:
            self.state = np.array([self.state[1], self.state[0]], dtype=complex)
        else:
            self.state = self.state

    def measure(self):
        outcome = np.random.choice([0, 1], p=[np.abs(self.state[0]) ** 2, np.abs(self.state[1]) ** 2])
        if outcome == 0:
            self.state = np.array([1.0, 0.0], dtype=complex)
        else:
            self.state = np.array([0.0, 1.0], dtype=complex)

    def calculate_probability(self, outcome):
        if outcome == 0:
            return np.abs(self.state[0]) ** 2
        else:
            return np.abs(self.state[1]) ** 2

    def print_state(self):
        print(f"Qubit state: {self.state[0]:.2f}+i{self.state[1]:.2f}|0+|1")

    def print_probability(self, outcome):
        print(f"Probability of outcome {outcome}: {self.calculate_probability(outcome):.2f}")

# Example usage:
q1 = qubit()
q1.apply_hadamard()
print(q1.print_state())  # Qubit state: 0.71+0.71|0+|1
q1.apply_pauli_x()
print(q1.print_state())  # Qubit state: 0.71+0.71|0+|1
q1.measure()
print(q1.print_state())  # Qubit state: 1.0|0
print(q1.print_probability(0))  # Probability of outcome 0: 1.00
print(q1.print_probability(1))  # Probability of outcome 1: 0.00