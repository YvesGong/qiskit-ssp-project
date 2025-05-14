from qiskit import QuantumCircuit
import pickle

NUM_QUBITS = 3
M_values = [2, 4, 8, 16, 32, 64]
BASES = ['XXX', 'XYX', 'XZX', 'YXY', 'YYY', 'YZY', 'ZXZ', 'ZYZ', 'ZZZ', 'XXY',
         'XYY', 'XZY', 'YXX', 'YYX', 'YZX', 'XXZ', 'XYZ', 'XZZ', 'ZXX', 'ZYX',
         'ZZX', 'YXZ', 'YYZ', 'YZZ', 'ZXY', 'ZYY', 'ZZY']

def add_cnots(qc: QuantumCircuit, M: int):
    if (M < 0):
        raise ValueError("Number of CNOT gates (M) cannot be negative.")
    
    for i in range(M):
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.cx(1, 2)
        # Add a barrier after each CNOT for visual separation if M > 0
        if (M > 0):
             qc.barrier()

basis_transformations = {
    'X': {
        'initial': [('h', None)],
        'final':   [('h', None)]
    },
    'Y': {
        'initial': [('h', None), ('s', None)],
        'final':   [('sdg', None), ('h', None)]
    },
    'Z': {
        'initial': [],
        'final':   []
    }
}

def create_circuit(M: int, basis_str: str) -> QuantumCircuit:
    if not ((isinstance(basis_str, str)) and (len(basis_str) == 3) and (basis_str[0] in basis_transformations) and (basis_str[1] in basis_transformations) and (basis_str[2] in basis_transformations)):
        raise ValueError(f"Invalid basis_str '{basis_str}'. Must be 3 chars from {list(basis_transformations.keys())}")

    basis_q0 = basis_str[0]
    basis_q1 = basis_str[1]
    basis_q2 = basis_str[2]

    qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS, name=f"M={M}_basis={basis_str}")

    # Apply initial transformation gates
    for gate_name, _ in basis_transformations[basis_q0]['initial']:
        getattr(qc, gate_name)(0)
    
    for gate_name, _ in basis_transformations[basis_q1]['initial']:
        getattr(qc, gate_name)(1)

    for gate_name, _ in basis_transformations[basis_q2]['initial']:
        getattr(qc, gate_name)(2)
    
    if ((basis_transformations[basis_q0]['initial']) or (basis_transformations[basis_q1]['initial']) or (basis_transformations[basis_q2]['initial'])):
        qc.barrier()

    add_cnots(qc, M)

    for gate_name, _ in basis_transformations[basis_q0]['final']:
        getattr(qc, gate_name)(0)
    
    for gate_name, _ in basis_transformations[basis_q1]['final']:
        getattr(qc, gate_name)(1)
    
    for gate_name, _ in basis_transformations[basis_q2]['final']:
        getattr(qc, gate_name)(2)
        
    if ((basis_transformations[basis_q0]['final']) or (basis_transformations[basis_q1]['final'])):
        qc.barrier()
    
    qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))
    return qc

# test_M = 2
# test_basis_str = 'XYX'
# example_circuit = create_circuit(test_M, test_basis_str)
# print(f"\nExample circuit for M={test_M}, basis={test_basis_str}:")
# print(example_circuit.draw('text'))
# print("-" * 20 + "\n")

circuits_X, circuits_Y, circuits_Z = [], [], []
circuits = [] 

for base in BASES:
    
    for M in M_values:

        circuit = create_circuit(M, base)
        
        if (base[1] == "X"):
            circuits_X.append(circuit)
    
        elif (base[1] == "Y"):
            circuits_Y.append(circuit)
    
        elif (base[1] == "Z"):
            circuits_Z.append(circuit)
    
        else:
            pass

circuits.append(circuits_X)
circuits.append(circuits_Y)
circuits.append(circuits_Z)
with open("circuits.bin", "wb") as f:
    pickle.dump(circuits, f)