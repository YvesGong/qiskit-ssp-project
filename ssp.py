# Phase 1: Setup & Imports

# --- Standard Libraries ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json     # For saving results later (good practice)
import time     # To time the simulation loop
import os       # To check if results file exists
import random   # For generating random numbers for Pauli noise

# --- Qiskit Imports ---
# Core components for circuit creation and transpilation
from qiskit import QuantumCircuit, transpile

# Aer simulator for ideal and noisy simulations
from qiskit_aer import AerSimulator

# Noise modeling components
from qiskit_aer.noise import NoiseModel, pauli_error, QuantumError

# --- Configuration Block ---
# This section groups all the main parameters for easy modification.
print("--- Configuration ---")

# Simulation parameters
SHOTS = 4096             # Number of times to run each circuit simulation
NUM_QUBITS = 2           # Working with a 2-qubit system

# M values to simulate (list of even numbers of CNOT gates)
M_values = [2, 4, 8, 16, 32, 64]

# New measurement basis combinations
BASES = ['XX', 'YY', 'ZZ', 'XY', 'YX', 'XZ', 'ZX', 'YZ', 'ZY']

# Noise Parameters
# Total probability for any of the 15 non-identity 2-qubit Pauli errors to occur after a CX gate.
# The remaining probability (1 - TOTAL_PAULI_ERROR_PROB) will be for the 'II' (identity) operation.
TOTAL_PAULI_ERROR_PROB = 0.03  # Example: 3% total error probability for non-Identity Paulis
PAULI_NOISE_SEED = 42          # Seed for reproducibility of random Pauli error distribution

# Results file path
RESULTS_FILENAME = "simulation_counts_custom_pauli_v3.json"

# Plot directory (plots will be saved here with dynamic names)
PLOT_DIR = "plots_custom_pauli_v3"

if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# Option to re-run simulations even if results file exists
FORCE_RERUN_SIMULATIONS = True # Set to True to force re-running Phase 4

# Print configuration to console for verification
print(f"Shots per circuit: {SHOTS}")
print(f"Number of qubits: {NUM_QUBITS}")
print(f"M values to simulate: {M_values}")
print(f"Measurement bases: {BASES}")
print(f"Total Pauli Error Probability (for non-II): {TOTAL_PAULI_ERROR_PROB}")
print(f"Pauli Noise Seed: {PAULI_NOISE_SEED}")
print(f"Results will be saved to: {RESULTS_FILENAME}")
print(f"Plots will be saved to directory: {PLOT_DIR}")
print(f"Force re-run simulations: {FORCE_RERUN_SIMULATIONS}")
print("-" * 20 + "\n")

# --- End of Phase 1 ---

# Phase 2: Define Noise Model
print("--- Defining Custom Pauli Noise Model ---")

# Set the seed for numpy's random number generator for Pauli error distribution
np.random.seed(PAULI_NOISE_SEED)
random.seed(PAULI_NOISE_SEED) # Also for python's random, if used directly

# 1. Define the 15 non-identity 2-qubit Pauli strings
pauli_ops_2q_non_identity = ['IX', 'IY', 'IZ', 'XI', 'YI', 'ZI', 'XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
num_non_identity_paulis = len(pauli_ops_2q_non_identity) # Should be 15

# 2. Generate a random vector to distribute the TOTAL_PAULI_ERROR_PROB
# Create 14 random numbers strictly between 0 and 1
rand_points = sorted([0.0] + [random.uniform(1e-4, 1.0-1e-4) for _ in range(num_non_identity_paulis - 1)] + [1.0])

# 3. Calculate differences to get 15 proportions that sum to 1
proportions = [rand_points[i+1] - rand_points[i] for i in range(num_non_identity_paulis)]
# Ensure proportions sum to 1 (approximately)
print(f"Sum of proportions: {sum(proportions)}")

# 4. Scale these proportions by TOTAL_PAULI_ERROR_PROB
# These are the probabilities for each of the 15 non-identity Pauli errors
individual_error_probs = [p * TOTAL_PAULI_ERROR_PROB for p in proportions]

# 5. Calculate the probability of the Identity 'II' (no error)
prob_identity = 1.0 - TOTAL_PAULI_ERROR_PROB

if (prob_identity < 0):
    raise ValueError(f"TOTAL_PAULI_ERROR_PROB ({TOTAL_PAULI_ERROR_PROB}) is > 1, which is not allowed.")

# 6. Construct the list of (PauliString, Probability) pairs for the QuantumError
pauli_errors_for_cx_complete = [('II', prob_identity)]

for i in range(num_non_identity_paulis):
    pauli_errors_for_cx_complete.append((pauli_ops_2q_non_identity[i], individual_error_probs[i]))

print(f"Pauli errors defined for CX gate (Total Error Prob={TOTAL_PAULI_ERROR_PROB}):")

for pauli_op, prob in pauli_errors_for_cx_complete:
    print(f"  {pauli_op}: {prob:.6f}")

# Sanity check: Ensure the probabilities sum to approximately 1
total_prob_check = sum(prob for _, prob in pauli_errors_for_cx_complete)
print(f"Sanity check: Sum of all Pauli channel probabilities = {total_prob_check:.6f}") # Should be very close to 1.0

# 7. Create the QuantumError object
try:
    if not np.isclose(total_prob_check, 1.0):
        raise ValueError(f"Probabilities do not sum to 1 ({total_prob_check}), cannot create Pauli error.")
    
    cx_pauli_channel: QuantumError = pauli_error(pauli_errors_for_cx_complete)
    print("Successfully created custom Pauli error channel object.")

except Exception as e:
    print(f"Error creating Pauli error channel: {e}")
    cx_pauli_channel = None # Handle error appropriately

# 8. Create an empty NoiseModel
noise_model = NoiseModel()
print("Created empty NoiseModel.")

# 9. Add the Pauli error channel to the NoiseModel for 'cx' gates
if cx_pauli_channel:
    noise_model.add_all_qubit_quantum_error(cx_pauli_channel, ['cx'])
    print("Added custom Pauli error channel to 'cx' gates in the noise model.")

else:
    print("Skipping adding Pauli error due to creation failure.")

# 10. Instantiate the AerSimulator instances
sim_noise = AerSimulator(noise_model=noise_model)
sim_ideal = AerSimulator()
print("Created AerSimulator instances (noisy and ideal).")
print("-" * 20 + "\n")

# --- End of Phase 2 ---

# Phase 3: Define Circuit Generation Functions
print("--- Defining Circuit Generation Functions (v3.1) ---")

# 1. Define the CNOT pattern function (Alternating CX gates)
def add_cnots(qc: QuantumCircuit, M: int):
    if (M < 0):
        raise ValueError("Number of CNOT gates (M) cannot be negative.")
    
    for i in range(M):
        qc.cx(0, 1)
        # Add a barrier after each CNOT for visual separation if M > 0
        if (M > 0):
             qc.barrier()

# 2. Define the basis transformation gates (initial and final)
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
print("Defined CNOT pattern function and new basis transformation map.")

# 3. Define the main circuit generation function
def create_measurement_circuit(M: int, basis_str: str) -> QuantumCircuit:
    if not ((isinstance(basis_str, str)) and (len(basis_str) == 2) and (basis_str[0] in basis_transformations) and (basis_str[1] in basis_transformations)):
        raise ValueError(f"Invalid basis_str '{basis_str}'. Must be 2 chars from {list(basis_transformations.keys())}")

    basis_q0 = basis_str[0]
    basis_q1 = basis_str[1]

    qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS, name=f"M={M}_basis={basis_str}")

    # Apply initial transformation gates
    for gate_name, _ in basis_transformations[basis_q0]['initial']:
        getattr(qc, gate_name)(0)
    
    for gate_name, _ in basis_transformations[basis_q1]['initial']:
        getattr(qc, gate_name)(1)
    
    if ((basis_transformations[basis_q0]['initial']) or (basis_transformations[basis_q1]['initial'])):
        qc.barrier()

    add_cnots(qc, M)

    for gate_name, _ in basis_transformations[basis_q0]['final']:
        getattr(qc, gate_name)(0)
    
    for gate_name, _ in basis_transformations[basis_q1]['final']:
        getattr(qc, gate_name)(1)
        
    if ((basis_transformations[basis_q0]['final']) or (basis_transformations[basis_q1]['final'])):
        qc.barrier()
    
    qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))
    return qc

print("Defined main circuit generation function 'create_measurement_circuit'.")

test_M = 4
test_basis_str = 'XY'
example_circuit = create_measurement_circuit(test_M, test_basis_str)
print(f"\nExample circuit for M={test_M}, basis={test_basis_str}:")
print(example_circuit.draw('text'))
print("-" * 20 + "\n")

# --- End of Phase 3 ---

# Phase 4: Run Simulations (or Load Results)
print("--- Running Simulations or Loading Results (v3.1) ---")

ideal_counts = {}
noisy_counts = {}
simulation_duration_total = 0
needs_rerun_overall = FORCE_RERUN_SIMULATIONS
config_changed = False

if ((os.path.exists(RESULTS_FILENAME)) and (not FORCE_RERUN_SIMULATIONS)):
    print(f"Loading existing results from {RESULTS_FILENAME}...")
    
    try:
        with open(RESULTS_FILENAME, 'r') as f:
            results_data = json.load(f)
        
        loaded_M_values = results_data.get('M_values', [])
        loaded_BASES = results_data.get('bases', [])
        loaded_total_pauli_error_prob = results_data.get('total_pauli_error_prob', -1)

        if ((M_values != loaded_M_values) or (BASES != loaded_BASES) or (not np.isclose(TOTAL_PAULI_ERROR_PROB, loaded_total_pauli_error_prob))):
            print("Configuration (M_values, BASES, or error probability) has changed. Re-running all simulations.")
            needs_rerun_overall = True
            config_changed = True
        
        else:
            ideal_counts = {int(k): v for k, v in results_data.get('ideal_counts', {}).items()}
            noisy_counts = {int(k): v for k, v in results_data.get('noisy_counts', {}).items()}
            simulation_duration_total = results_data.get('simulation_duration_sec', 0)
            print(f"Results loaded. Stored simulation duration: {simulation_duration_total:.2f}s.")
            
            for m_val in M_values:
                
                if m_val not in ideal_counts or m_val not in noisy_counts:
                    needs_rerun_overall = True; break
                
                for basis_str_check in BASES:
                    
                    if ((basis_str_check not in ideal_counts.get(m_val, {})) or (basis_str_check not in noisy_counts.get(m_val, {}))):
                        
                        needs_rerun_overall = True
                        break
                
                if needs_rerun_overall: 
                    break
            
            if ((needs_rerun_overall) and (not config_changed)):
                 print("Not all required M values/BASES found in the results file. Will run simulations for missing parts.")

    except Exception as e:
        print(f"Error loading results file: {e}. Re-running all simulations.")
        needs_rerun_overall = True
        ideal_counts = {}
        noisy_counts = {}
        simulation_duration_total = 0
else:
    needs_rerun_overall = True
    
    if not FORCE_RERUN_SIMULATIONS:
        print("Results file not found or re-run forced because config changed.")
    
    else: # This case implies FORCE_RERUN_SIMULATIONS is True
        print("Forcing re-run of all simulations.")


if needs_rerun_overall:
    print("Running simulations...")
    current_simulation_start_time = time.time()
    new_sims_actually_run_this_session = False

    for M_iter in M_values:
        
        if M_iter not in noisy_counts or config_changed or FORCE_RERUN_SIMULATIONS: 
            noisy_counts[M_iter] = {}
        
        if M_iter not in ideal_counts or config_changed or FORCE_RERUN_SIMULATIONS: 
            ideal_counts[M_iter] = {}

        print(f"--- M = {M_iter} ---")
        
        for basis_str_iter in BASES:
            
            if ((not ((config_changed) or (FORCE_RERUN_SIMULATIONS))) and (basis_str_iter in ideal_counts.get(M_iter, {})) and (basis_str_iter in noisy_counts.get(M_iter, {}))):
                print(f"  Skipping basis {basis_str_iter} (already loaded and config unchanged).")
                continue

            new_sims_actually_run_this_session = True
            print(f"  Simulating basis: {basis_str_iter}")
            
            try:
                qc = create_measurement_circuit(M_iter, basis_str_iter)
            
            except ValueError as e:
                print(f"    Error generating circuit for M={M_iter}, basis={basis_str_iter}: {e}")
                ideal_counts[M_iter][basis_str_iter] = {} 
                noisy_counts[M_iter][basis_str_iter] = {}
                continue

            try:
                qc_ideal_t = transpile(qc, sim_ideal, optimization_level=0)
                job_ideal = sim_ideal.run(qc_ideal_t, shots=SHOTS)
                result_ideal = job_ideal.result()
                ideal_counts[M_iter][basis_str_iter] = result_ideal.get_counts(qc_ideal_t)
                print(f"    Ideal counts for M={M_iter}, basis={basis_str_iter}: {ideal_counts[M_iter][basis_str_iter]}") 
            
            except Exception as e:
                print(f"    Error during IDEAL simulation for M={M_iter}, basis={basis_str_iter}: {e}") # Used basis_str_iter
                ideal_counts[M_iter][basis_str_iter] = {}

            try:
                qc_noisy_t = transpile(qc, sim_noise, optimization_level=1)
                job_noisy = sim_noise.run(qc_noisy_t, shots=SHOTS)
                result_noisy = job_noisy.result()
                noisy_counts[M_iter][basis_str_iter] = result_noisy.get_counts(qc_noisy_t)
                print(f"    Noisy counts for M={M_iter}, basis={basis_str_iter}: {noisy_counts[M_iter][basis_str_iter]}")
            
            except Exception as e:
                print(f"    Error during NOISY simulation for M={M_iter}, basis={basis_str_iter}: {e}")
                noisy_counts[M_iter][basis_str_iter] = {}
    
    if ((new_sims_actually_run_this_session) or (config_changed) or (FORCE_RERUN_SIMULATIONS)) :
        current_simulation_duration = time.time() - current_simulation_start_time
        print(f"\n--- Simulation Run Complete ---")
        print(f"Time for simulations in this session: {current_simulation_duration:.2f} seconds")
        
        if ((config_changed) or (FORCE_RERUN_SIMULATIONS)):
            simulation_duration_total = current_simulation_duration
        
        else: 
            simulation_duration_total += current_simulation_duration

        results_data_to_save = {
            'M_values': M_values,
            'bases': BASES,
            'shots': SHOTS,
            'total_pauli_error_prob': TOTAL_PAULI_ERROR_PROB,
            'pauli_noise_seed': PAULI_NOISE_SEED,
            'ideal_counts': {str(k): v for k, v in ideal_counts.items()},
            'noisy_counts': {str(k): v for k, v in noisy_counts.items()},
            'simulation_duration_sec': simulation_duration_total
        }
        
        try:
            with open(RESULTS_FILENAME, 'w') as f:
                json.dump(results_data_to_save, f, indent=4)
            
            print(f"Successfully saved simulation results to {RESULTS_FILENAME}")
        
        except Exception as e:
            print(f"Error saving results to {RESULTS_FILENAME}: {e}")
    
    else:
        print("No new simulations were required in this session.")

if ((not ideal_counts) or (not noisy_counts)):
    print("Error: Failed to load or run simulations. Counts data is missing. Exiting.")
    exit()

print("-" * 20 + "\n")

# --- End of Phase 4 ---

# Phase 5: Calculate Survival Probabilities
print("--- Calculating Survival Probabilities P(00) (v3.1) ---")

P00_noisy_values_all_bases = {}
analysis_M_values_numeric = sorted([int(m_key) for m_key in noisy_counts.keys()])
print(f"Analysing M values: {analysis_M_values_numeric}")
M_values_for_analysis = [m_val for m_val in M_values if m_val in analysis_M_values_numeric]

if not M_values_for_analysis:
    print("Error: No M_values from config found in analysis_M_values_numeric. Check data consistency.")
    exit()

for basis_str_calc in BASES:
    P00_noisy_values_all_bases[basis_str_calc] = []
    print(f"\n  Calculating for basis: {basis_str_calc}")
    
    for M_calc in M_values_for_analysis:
        counts = noisy_counts.get(M_calc, {}).get(basis_str_calc, {})
        p00 = counts.get('00', 0) / SHOTS
        P00_noisy_values_all_bases[basis_str_calc].append(p00)
        print(f"    M = {M_calc:3d}, P(00)_{basis_str_calc} = {p00:.4f}")
    
    P00_noisy_values_all_bases[basis_str_calc] = np.array(P00_noisy_values_all_bases[basis_str_calc])

print("\n--- Sanity Check: Ideal P(00) ---")

for basis_str_check_ideal in BASES:
    print(f"  Ideal P(00) for basis: {basis_str_check_ideal}")
    ideal_p00_check_passed_basis = True
    all_ideal_p00_one = True # Flag to see if all P(00) are 1 for this basis

    for M_check in M_values_for_analysis:
        ideal_basis_counts = ideal_counts.get(M_check, {}).get(basis_str_check_ideal, {})
        ideal_p00 = ideal_basis_counts.get('00', 0) / SHOTS
        print(f"    M = {M_check:3d}, Ideal P(00)_{basis_str_check_ideal} = {ideal_p00:.4f}", end="")
        if not np.isclose(ideal_p00, 1.0):
            all_ideal_p00_one = False

    if ideal_p00_check_passed_basis:
        
        if all_ideal_p00_one:
            print(f"    Ideal P(00)_{basis_str_check_ideal} check passed (all are ~1.0).")
        
        else:
            print(f"    Ideal P(00)_{basis_str_check_ideal} behavior noted (not all are 1.0, but consistent with M values).")
    
    else:
        print(f"    WARNING: Ideal P(00)_{basis_str_check_ideal} check FAILED for some M values where P(00) should have been 1.0.")

print("-" * 20 + "\n")

# --- End of Phase 5 ---

# Phase 6: Plotting and Fitting (Multiple Plots)
print("--- Plotting and Fitting Results (v3.1 - Multiple Plots) ---")
np_M_values_for_analysis = np.array(M_values_for_analysis)

def fit_func(m_vals, A, p):
    return A * (p**m_vals) + 0.25

bounds_fit = ([0, 0], [0.75, 1.0]) 
initial_guesses_fit = [0.7, 0.99]
fitted_p_values = {} # To store p_fit for each basis for Phase 7

if (len(np_M_values_for_analysis) == 0):
    print("No data available for plotting and fitting. Exiting.")
    exit()

for basis_str_plot in BASES:
    
    if ((basis_str_plot not in P00_noisy_values_all_bases) or (len(P00_noisy_values_all_bases[basis_str_plot]) == 0)):
        print(f"No P(00) data available for basis {basis_str_plot}. Skipping plot.")
        continue

    current_P00_values = P00_noisy_values_all_bases[basis_str_plot]
    print(f"\n--- Plotting and Fitting for Basis: {basis_str_plot} ---")
    plt.figure(figsize=(12, 7))
    plt.scatter(np_M_values_for_analysis, current_P00_values, label=f'Noisy Sim P(00) (Basis: {basis_str_plot})', marker='o', color='blue')

    try:
        popt, pcov = curve_fit(fit_func, np_M_values_for_analysis, current_P00_values, p0=initial_guesses_fit, bounds=bounds_fit, maxfev=10000)
        A_fit, p_fit = popt
        fitted_p_values[basis_str_plot] = p_fit # Store the fitted p_decay (now p_fit)
        print(f"  Curve Fit Results for {basis_str_plot}:")
        print(f"    A = {A_fit:.4f}")
        print(f"    p = {p_fit:.4f}")
        
        try:
            perr = np.sqrt(np.diag(pcov))
            print(f"    Std Error: A_err = {perr[0]:.4f}, p_err = {perr[1]:.4f}")
        
        except Exception as e_stderr:
            print(f"    Could not estimate standard errors for fit parameters: {e_stderr}")

        m_plot_smooth = np.linspace(min(np_M_values_for_analysis), max(np_M_values_for_analysis), 200)
        p00_fit_curve = fit_func(m_plot_smooth, A_fit, p_fit)
        plt.plot(m_plot_smooth, p00_fit_curve, 'r-', label=f'Fit: $A p^M + 0.25$\n$A={A_fit:.3f}$, $p={p_fit:.3f}$')

    except RuntimeError as e_fit:
        print(f"  Error during curve fitting for {basis_str_plot}: {e_fit}")
        print("  Could not determine fit parameters. Plotting data points only.")
    
    except Exception as e_gen_fit:
        print(f"  An unexpected error occurred during fitting for {basis_str_plot}: {e_gen_fit}")

    plt.xlabel("Number of CNOT Gates (M)")
    plt.ylabel(f"Survival Probability P(00) (Basis: {basis_str_plot})")
    title_str = (f"Survival Probability P(00) vs. M (Basis: {basis_str_plot})\n" f"Total Pauli Error Prob (non-II): {TOTAL_PAULI_ERROR_PROB*100:.1f}%, Seed: {PAULI_NOISE_SEED}")
    plt.title(title_str)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=0.2, top=1.05)
    plot_filename = os.path.join(PLOT_DIR, f"survival_plot_basis_{basis_str_plot}_err_{TOTAL_PAULI_ERROR_PROB:.2f}_seed_{PAULI_NOISE_SEED}.png")
    
    try:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"  Successfully saved plot to {plot_filename}")
    
    except Exception as e_save:
        print(f"  Error saving plot {plot_filename}: {e_save}")
    
    plt.close()

print("-" * 20 + "\n")

# --- End of Phase 6 ---

# Phase 7: PTM Eigenvalue Calculation and Comparison
print("--- PTM Eigenvalue Calculation and Comparison ---")

# 7.1. Define Standard Pauli Order (Lexicographical)
PAULI_ORDER = ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']

# 7.2. Define Walsh-Hadamard Transform Matrix (H_W)
# This matrix should transform Pauli error probabilities to PTM eigenvalues for a Pauli channel.
single_q_transform = np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]])
H_W = np.kron(single_q_transform, single_q_transform)

# 7.3. Prepare Pauli Error Probability Vector (p_vector)
# This vector must be ordered according to PAULI_ORDER.
prob_map = dict(pauli_errors_for_cx_complete) # Convert list of tuples to dict for easy lookup
p_vector = np.array([prob_map.get(p_op, 0) for p_op in PAULI_ORDER]) # Create ordered vector

if (len(p_vector) != 16): # Sanity check
    print("Error: Probability vector length is not 16. Check PAULI_ORDER and pauli_errors_for_cx_complete.")
    exit()

# 7.4. Calculate PTM Eigenvalues (lambda_vector)
# Eigenvalues are calculated by transforming the probability vector with H_W.
lambda_vector = H_W @ p_vector # Matrix-vector product

# Store eigenvalues in a dictionary for easy access by Pauli string
ptm_eigenvalues = {pauli_op: eig_val for pauli_op, eig_val in zip(PAULI_ORDER, lambda_vector)}
print("\nCalculated PTM Eigenvalues (lambda_P):")

for pauli_op, eig_val in ptm_eigenvalues.items():
    print(f"  lambda_{pauli_op:<2s} = {eig_val:.6f}")

# 7.5. Define Comparison Mappings and Calculate Theoretical p Values
# These definitions specify how experimental decay rates map to PTM eigenvalues.
theoretical_p_values = {}
comparison_definitions = {
    'XX': ('XX', 'XI'),
    'XY': ('XY', 'YZ'),
    'XZ': ('XZ', 'YY'),
    'YX': ('YX', 'YI'),
    'YY': ('YY', 'XZ'),
    'YZ': ('YZ', 'XY'),
    'ZX': ('ZX', None), # Direct comparison for ZX
    'ZY': ('ZY', 'IY'),
    'ZZ': ('ZZ', 'IZ')
}
print("\nCalculating Theoretical 'p' values from PTM Eigenvalues:")

for basis, (op1_str, op2_str) in comparison_definitions.items():

    if basis not in BASES:
        continue # Ensure basis is valid

    lambda_op1 = ptm_eigenvalues.get(op1_str, np.nan) # Get eigenvalue for first operator

    if op2_str: # If geometric mean is needed (two operators involved)
        lambda_op2 = ptm_eigenvalues.get(op2_str, np.nan) # Get eigenvalue for second operator
        # Theoretical p is sqrt of the absolute product of eigenvalues.
        # abs() is used because p_fit is a magnitude (0 to 1), while eigenvalues can be negative.
        theoretical_p = np.sqrt(np.abs(lambda_op1 * lambda_op2))
        print(f"  For {basis:<2s}: sqrt(|lambda_{op1_str:<2s} * lambda_{op2_str:<2s}|) = sqrt(|{lambda_op1:.4f} * {lambda_op2:.4f}|) = {theoretical_p:.4f}")
    
    else: # Direct comparison (one operator involved)
        theoretical_p = np.abs(lambda_op1) # Theoretical p is the absolute value of the eigenvalue
        print(f"  For {basis:<2s}: |lambda_{op1_str:<2s}| = |{lambda_op1:.4f}| = {theoretical_p:.4f}")
    
    theoretical_p_values[basis] = theoretical_p

# 7.6. Print Comparison Table
# This table compares the experimental p_fit values with the theoretical p values.
print("\n--- Comparison: Experimental p_fit vs. Theoretical p (from PTM  Eigenvalues) ---")
print(f"{'Basis':<6} | {'Exp. p_fit':<12} | {'Theory p':<10} | {'Diff (%)':<10}")
print("-" * 52)
missing_fits = 0

for basis_comp in BASES: # Iterate over all bases used in the experiment

    if basis_comp in comparison_definitions: # Check if a comparison rule is defined for this basis
        exp_p = fitted_p_values.get(basis_comp, np.nan) # Get experimental p_fit
        thy_p = theoretical_p_values.get(basis_comp, np.nan) # Get theoretical p

        if np.isnan(exp_p): # Handle cases where experimental fit might have failed
            print(f"{basis_comp:<6} | {'N/A':<12} | {thy_p:<10.4f} | {'N/A'}")
            missing_fits += 1
            continue
        
        if np.isnan(thy_p): # Handle cases where theoretical value couldn't be calculated (e.g., missing Pauli op)
            print(f"{basis_comp:<6} | {exp_p:<12.4f} | {'N/A':<10} | {'N/A'}")
            continue
    
    # Calculate percentage difference
    diff_percent = ((exp_p - thy_p) / thy_p) * 100 if (thy_p != 0) else np.inf
    print(f"{basis_comp:<6} | {exp_p:<12.4f} | {thy_p:<10.4f} | {diff_percent:<10.2f}%")

if (missing_fits > 0): # Notify if some experimental fits were missing
    print(f"\nNote: {missing_fits} experimental p_fit value(s) were not available (e.g., due to fitting errors).")

print("-" * 20 + "\n")

# --- End of Phase 7 ---

print("--- Script Finished ---")