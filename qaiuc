import streamlit as st
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from qiskit.quantum_info import Statevector, entropy, partial_trace
import pandas as pd
import pickle
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel

def initialize_circuit(num_qubits):
    return QuantumCircuit(num_qubits, num_qubits)

def add_gate(circuit, gate_type, qubit, control=None, target2=None, angle=None):
    try:
        if gate_type == 'H':
            circuit.h(qubit)
        elif gate_type == 'X':
            circuit.x(qubit)
        elif gate_type == 'Y':
            circuit.y(qubit)
        elif gate_type == 'Z':
            circuit.z(qubit)
        elif gate_type == 'CNOT' and control is not None:
            circuit.cx(control, qubit)
        elif gate_type == 'CCNOT' and control is not None and target2 is not None:
            circuit.ccx(control, qubit, target2)
        elif gate_type == 'CZ' and control is not None:
            circuit.cz(control, qubit)
        elif gate_type == 'SWAP' and target2 is not None:
            circuit.swap(qubit, target2)
        elif gate_type == 'CSWAP' and control is not None and target2 is not None:
            circuit.cswap(control, qubit, target2)
        elif gate_type == 'S':
            circuit.s(qubit)
        elif gate_type == 'T':
            circuit.t(qubit)
        elif gate_type == 'RX' and angle is not None:
            circuit.rx(angle, qubit)
        elif gate_type == 'RY' and angle is not None:
            circuit.ry(angle, qubit)
        elif gate_type == 'RZ' and angle is not None:
            circuit.rz(angle, qubit)
        else:
            st.error("Invalid gate or parameters.")
    except Exception as e:
        st.error(f"Error adding gate: {e}")
    return circuit

def get_statevector(circuit):
    try:
        backend = Aer.get_backend('statevector_simulator')
        job = execute(circuit, backend)
        result = job.result()
        statevector = result.get_statevector()
        return statevector
    except Exception as e:
        st.error(f"Error getting statevector: {e}")
        return None

def plot_circuit_diagram(circuit):
    try:
        return circuit.draw(output='text')
    except Exception as e:
        st.error(f"Error plotting circuit diagram: {e}")
        return ""

def plot_probability_distribution(statevector):
    try:
        probabilities = np.abs(statevector) ** 2
        num_states = len(probabilities)
        states = [format(i, f'0{int(np.log2(num_states))}b') for i in range(num_states)]
        
        fig = go.Figure(data=[
            go.Bar(x=states, y=probabilities)
        ])
        
        fig.update_layout(
            title='Quantum State Probabilities',
            xaxis_title='Basis States',
            yaxis_title='Probability',
            yaxis_range=[0, 1]
        )
        
        return fig
    except Exception as e:
        st.error(f"Error plotting probability distribution: {e}")
        return None

def plot_bloch(statevector):
    try:
        fig = plot_bloch_multivector(statevector)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting Bloch sphere: {e}")

def save_circuit(circuit, filename):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(circuit, f)
        st.sidebar.success('Circuit Saved!')
    except Exception as e:
        st.sidebar.error(f"Error saving circuit: {e}")

def load_circuit(filename):
    try:
        with open(filename, 'rb') as f:
            circuit = pickle.load(f)
        st.sidebar.success('Circuit Loaded!')
        return circuit
    except Exception as e:
        st.sidebar.error(f"Error loading circuit: {e}")
        return None

def quantum_teleportation_circuit():
    try:
        qc = QuantumCircuit(3, 3)
        qc.h(1)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.h(0)
        qc.measure([0, 1], [0, 1])
        qc.cx(1, 2)
        qc.cz(0, 2)
        return qc
    except Exception as e:
        st.error(f"Error creating quantum teleportation circuit: {e}")
        return None

def bell_state_generator_circuit():
    try:
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        return qc
    except Exception as e:
        st.error(f"Error creating Bell state generator circuit: {e}")
        return None

def grovers_algorithm_circuit(num_qubits):
    try:
        qc = QuantumCircuit(num_qubits)
        qc.h(range(num_qubits))
        qc.z(range(num_qubits))
        qc.cz(0, range(1, num_qubits))
        qc.h(range(num_qubits))
        qc.z(range(num_qubits))
        qc.h(range(num_qubits))
        return qc
    except Exception as e:
        st.error(f"Error creating Grover's algorithm circuit: {e}")
        return None

def qft_circuit(num_qubits):
    try:
        qc = QuantumCircuit(num_qubits)
        for j in range(num_qubits):
            for k in range(j):
                qc.cp(np.pi/2**(j-k), k, j)
            qc.h(j)
        return qc
    except Exception as e:
        st.error(f"Error creating QFT circuit: {e}")
        return None

def export_circuit_as_image(circuit, filename):
    try:
        circuit.draw(output='mpl', filename=filename)
        st.sidebar.success('Circuit Exported!')
    except Exception as e:
        st.sidebar.error(f"Error exporting circuit: {e}")

def compute_entanglement_entropy(statevector, num_qubits):
    try:
        entropies = []
        for qubit in range(num_qubits):
            reduced_state = partial_trace(statevector, [qubit])
            entropies.append(entropy(reduced_state))
        return entropies
    except Exception as e:
        st.error(f"Error computing entanglement entropy: {e}")
        return []

def main():
    st.title('Quantum Circuit Simulator')
    
    # Sidebar controls
    st.sidebar.header('Circuit Controls')
    num_qubits = st.sidebar.slider('Number of Qubits', 1, 5, 2)
    
    # Initialize or get circuit from session state
    if 'circuit' not in st.session_state or st.sidebar.button('Reset Circuit'):
        st.session_state.circuit = initialize_circuit(num_qubits)
        st.session_state.gate_history = []
    
    # Predefined circuits
    st.sidebar.header('Predefined Circuits')
    predefined_circuit = st.sidebar.selectbox(
        'Select Predefined Circuit',
        ['None', 'Quantum Teleportation', 'Bell State Generator', 'Grover\'s Algorithm', 'QFT']
    )
    
    if predefined_circuit == 'Quantum Teleportation':
        st.session_state.circuit = quantum_teleportation_circuit()
        st.session_state.gate_history = [{'gate': 'Quantum Teleportation', 'target': '', 'control': '', 'angle': ''}]
    elif predefined_circuit == 'Bell State Generator':
        st.session_state.circuit = bell_state_generator_circuit()
        st.session_state.gate_history = [{'gate': 'Bell State Generator', 'target': '', 'control': '', 'angle': ''}]
    elif predefined_circuit == 'Grover\'s Algorithm':
        st.session_state.circuit = grovers_algorithm_circuit(num_qubits)
        st.session_state.gate_history = [{'gate': 'Grover\'s Algorithm', 'target': '', 'control': '', 'angle': ''}]
    elif predefined_circuit == 'QFT':
        st.session_state.circuit = qft_circuit(num_qubits)
        st.session_state.gate_history = [{'gate': 'QFT', 'target': '', 'control': '', 'angle': ''}]
    
    # Available gates
    st.sidebar.header('Quantum Gates')
    gate_type = st.sidebar.selectbox(
        'Select Gate',
        ['H', 'X', 'Y', 'Z', 'CNOT', 'CCNOT', 'CZ', 'SWAP', 'CSWAP', 'S', 'T', 'RX', 'RY', 'RZ']
    )
    
    # Gate application controls
    target_qubit = st.sidebar.selectbox(
        'Target Qubit',
        range(num_qubits)
    )
    
    control_qubit = None
    target_qubit2 = None
    angle = None
    if gate_type in ['CNOT', 'CZ']:
        control_options = [i for i in range(num_qubits) if i != target_qubit]
        if control_options:
            control_qubit = st.sidebar.selectbox(
                'Control Qubit',
                control_options
            )
    elif gate_type == 'CCNOT':
        control_options = [i for i in range(num_qubits) if i != target_qubit]
        if control_options:
            control_qubit = st.sidebar.selectbox(
                'First Control Qubit',
                control_options
            )
        target_options = [i for i in range(num_qubits) if i != target_qubit and i != control_qubit]
        if target_options:
            target_qubit2 = st.sidebar.selectbox(
                'Second Control Qubit',
                target_options
            )
    elif gate_type == 'CSWAP':
        control_options = [i for i in range(num_qubits) if i != target_qubit]
        if control_options:
            control_qubit = st.sidebar.selectbox(
                'Control Qubit',
                control_options
            )
        target_options = [i for i in range(num_qubits) if i != target_qubit and i != control_qubit]
        if target_options:
            target_qubit2 = st.sidebar.selectbox(
                'Second Qubit',
                target_options
            )
    elif gate_type == 'SWAP':
        target_options = [i for i in range(num_qubits) if i != target_qubit]
        if target_options:
            target_qubit2 = st.sidebar.selectbox(
                'Second Qubit',
                target_options
            )
    elif gate_type in ['RX', 'RY', 'RZ']:
        angle = st.sidebar.slider('Rotation Angle', 0.0, 2 * np.pi, 0.0)
    
    # Add gate button
    if st.sidebar.button('Add Gate'):
        st.session_state.circuit = add_gate(
            st.session_state.circuit,
            gate_type,
            target_qubit,
            control_qubit,
            target_qubit2,
            angle
        )
        st.session_state.gate_history.append({
            'gate': gate_type,
            'target': target_qubit,
            'control': control_qubit,
            'target2': target_qubit2,
            'angle': angle
        })
    
    # Display circuit
    st.header('Circuit Diagram')
    st.text(plot_circuit_diagram(st.session_state.circuit))
    
    # Gate History
    st.header('Gate History')
    if st.session_state.gate_history:
        df = pd.DataFrame(st.session_state.gate_history).astype(str)
        st.dataframe(df)
    
    # Save and Load Circuit
    st.sidebar.header('Save/Load Circuit')
    save_filename = st.sidebar.text_input('Save Circuit Filename', 'circuit.pkl')
    if st.sidebar.button('Save Circuit'):
        save_circuit(st.session_state.circuit, save_filename)
    
    load_filename = st.sidebar.text_input('Load Circuit Filename', 'circuit.pkl')
    if st.sidebar.button('Load Circuit'):
        st.session_state.circuit = load_circuit(load_filename)
        st.session_state.gate_history = []
    
    # Export Circuit as Image
    st.sidebar.header('Export Circuit')
    export_filename = st.sidebar.text_input('Export Circuit Filename', 'circuit.png')
    if st.sidebar.button('Export Circuit'):
        export_circuit_as_image(st.session_state.circuit, export_filename)
    
    # Simulation results
    st.header('Simulation Results')
    if st.button('Run Simulation'):
        # Get statevector
        statevector = get_statevector(st.session_state.circuit)
        
        if statevector is not None:
            # Plot probability distribution
            st.plotly_chart(plot_probability_distribution(statevector))
            
            # Plot Bloch sphere for each qubit
            st.subheader('Bloch Sphere Visualization')
            plot_bloch(statevector)
            
            # Display measurement probabilities
            st.subheader('Basis State Probabilities')
            probs = np.abs(statevector) ** 2
            states = [format(i, f'0{int(np.log2(len(probs)))}b') for i in range(len(probs))]
            prob_df = pd.DataFrame({
                'State': states,
                'Probability': probs
            })
            st.dataframe(prob_df)
            
            # Perform measurements
            st.subheader('Sample Measurements')
            num_measurements = 1000
            backend = Aer.get_backend('qasm_simulator')
            circuit_with_measure = st.session_state.circuit.copy()
            circuit_with_measure.measure_all()
            job = execute(circuit_with_measure, backend, shots=num_measurements)
            result = job.result()
            counts = result.get_counts()
            
            # Display measurement results
            measurements_df = pd.DataFrame({
                'State': list(counts.keys()),
                'Count': list(counts.values()),
                'Frequency': [count/num_measurements for count in counts.values()]
            })
            st.dataframe(measurements_df)

    # Entanglement and Correlation Analysis
    st.header('Entanglement and Correlation Analysis')
    if st.button('Compute Entanglement Entropy'):
        # Get statevector
        statevector = get_statevector(st.session_state.circuit)
        
        if statevector is not None:
            entropies = compute_entanglement_entropy(statevector, num_qubits)
            if len(entropies) == num_qubits:
                st.subheader('Entanglement Entropy')
                entropy_df = pd.DataFrame({
                    'Qubit': range(num_qubits),
                    'Entropy': entropies
                })
                st.dataframe(entropy_df)
            else:
                st.error("Error: Mismatch in number of qubits and entropies calculated.")
    
if __name__ == '__main__':
    main()
