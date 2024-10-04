@qml.qnode(backend, qubits=2)
def prepare_bell_state():
    qml.Hadamard(wires=0) # Superposition on the first qubit
    qml.CNOT(wires=[0, 1]) # Controlled-NOT gate between qubits

def encode(state, bit):
    if state == 0:
        qml.X(wires=0)
        return bit
    else:
        qml.X(wires=1)
        qml.Z(wires=1)
        return ~bit

@qml.qnode(backend, qubits=2)
def measurement(state, basis):
    if basis == 0: # Measure in X basis
        qml.measure(wires=[0], key="x_0")
        qml.measure(wires=[1], key="x_1")
    else: # Measure in Z basis
        qml.H(wires=[0]) # Hadamard gate to put the first qubit into superposition
        qml.measure(wires=[0], key="z_0")
        qml.H(wires=[1])
        qml.measure(wires=[1], key="z_1")

@qml.qnode(backend, qubits=2)
def prepare_entangled_state():
    qml.Hadamard(wires=[0]) # Superposition on the first qubit
    qml.CNOT(wires=[0, 1]) # Controlled-NOT gate between qubits
    return qml.expval(PauliZ, wires=1)

def teleport_qubit(a, b):
    # Alice performs a measurement on her qubit and communicates the result to Bob
    x = qml.measure(wires=0, key="x")
    classical_communication(x)

    # Based on the measurement outcome, Bob applies appropriate gates
    if x == 0:
        qml.H(wires=[1])
        qml.X(wires=[1])
    elif x == 1:
        qml.Z(wires=[1])
