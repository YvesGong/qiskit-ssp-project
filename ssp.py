# Phase 1: Setup & Imports

# --- Standard Libraries ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json # For saving results later (good practice)
import time # To time the simulation loop
import os   # To check if results file exists

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
NUM_QUBITS = 2           # Working wih a 2-qubit system (based on analysis)

# M values to simulate (list of even numbers of CNOT gates)
M_values = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]

# Measurement bases to iterate through
BASES = ['X', 'Y', 'Z']

# Noise Parameters
# Probability for the 2-qubit Pauli error applied after each CX gate
prob_cx_pauli = 0.01     # Example: 1% total Pauli error probability

# Results file path (for saving raw counts data after simulation)
RESULTS_FILENAME = "simulation_counts.json"

# Plot file path (for saving the final plot)
PLOT_FILENAME = "survival_plot_extended.png"

# Option to re-run simulations even if results file exists
# Set to True to force re-running Phase 4
FORCE_RERUN_SIMULATIONS = False

# Print configuration to console for verification
print(f"Shots per circuit: {SHOTS}")
print(f"Number of qubits: {NUM_QUBITS}")
print(f"M values to simulate: {M_values}")
print(f"Measurement bases: {BASES}")
print(f"CX Pauli Error Probability: {prob_cx_pauli}")
print(f"Results will be saved to: {RESULTS_FILENAME}")
print(f"Plot will be saved to: {PLOT_FILENAME}")
print(f"Force re-run simulations: {FORCE_RERUN_SIMULATIONS}")
print("-" * 20 + "\n")

# --- End of Phase 1 ---

# Phase 2: Define Noise Model
print("--- Defining Noise Model ---")

# 1. Define the specific Pauli errors and their probabilities for the CX gate.
"""
We'll use the symmetric model discussed: total probability 'prob_cx_pauli' is split
equally among single-qubit Pauli errors (X, Y, Z) occurring on *either* the control
or the target qubit after the CX gate. There are 6 such non-identity single-qubit
Pauli errors (IX, IY, IZ, XI, YI, ZI).
"""
num_single_pauli_errors = 6
prob_per_pauli_error = (prob_cx_pauli / num_single_pauli_errors)

# Calculate the probability of the identity (no error occurring)
# This is 1 minus the sum of probabilities of all specified error terms.
prob_identity = 1.0 - prob_cx_pauli

# List of (PauliString, Probability) pairs, now including the identity 'II'
# The sum of probabilities in this list should now be 1.0
pauli_errors_for_cx_complete = [
    ('II', prob_identity),     # Probability of no error
    ('IX', prob_per_pauli_error), ('IY', prob_per_pauli_error), ('IZ', prob_per_pauli_error),
    ('XI', prob_per_pauli_error), ('YI', prob_per_pauli_error), ('ZI', prob_per_pauli_error)
]
print(f"Pauli errors defined for CX gate (Identity prob={prob_identity:.4f}, Error prob={prob_per_pauli_error:.4f} each):")
print(pauli_errors_for_cx_complete)
# Sanity check: Ensure the probabilities sum to approximately 1
total_prob_check = sum(prob for _, prob in pauli_errors_for_cx_complete)
print(f"Sanity check: Sum of probabilities = {total_prob_check:.6f}")    # Should be very close to 1.0

# 2. Create the QuantumError object representing this Pauli channel
# Use the complete list that includes the 'II' term.
try:
    # Ensure the sum is close enough to 1 for numerical stability
    if not np.isclose(total_prob_check, 1.0):
        raise ValueError(f"Probabilities do not sum to 1 ({total_prob_check}), cannot create Pauli error.")
    
    cx_pauli_channel: QuantumError = pauli_error(pauli_errors_for_cx_complete)
    print("Successfully created Pauli error channel object.")

except Exception as e:
    print(f"Error creating Pauli error channel: {e}")
    # Handle error appropriately, maybe exit or use a default
    cx_pauli_channel = None   # Set to None or handle error

# 3. Create an empty NoiseModel
noise_model = NoiseModel()
print("Created empty NoiseModel.")

# 4. Add the Pauli error channel to the NoiseModel.
# This applies the 'cx_pauli_channel' to all instances of the 'cx' gate.
if cx_pauli_channel:
    noise_model.add_all_qubit_quantum_error(cx_pauli_channel, ['cx'])
    print("Added Pauli error channel to 'cx' gates in the noise model.")

else:
    print("Skipping adding Pauli error due to creation failure.")

# 5. Instantiate the AerSimulator instances
# One with the noise model, one ideal (no noise)
sim_noise = AerSimulator(noise_model=noise_model)
sim_ideal = AerSimulator()      # No noise model argument means ideal simulation

print("Created AerSimulator instances (noisy and ideal).")
print("-" * 20 + "\n")

# --- End of Phase 2 ---

# Phase 3: Define Circuit Generation Function
print("--- Defining Circuit Generation Functions ---")

# 1. Define the CNOT pattern function (Alternating CX gates)
def add_alternating_cnots(qc: QuantumCircuit, M: int):
    """
    Applies M CNOT gates, alternating between CX(0, 1) and CX(1, 0).
    """
    if (M < 0):
        raise ValueError("Number of CNOT gates (M) cannot be negative.")
    
    for i in range(M):

        if (i % 2 == 0):
            # Even index (0, 2, 4...): Apply CNOT(0, 1)
            qc.cx(0, 1)
        
        else:
            # Odd index (1, 3, 5...): Apply CNOT(1, 0)
            qc.cx(1, 0)
        
        # Add a barrier for visual separation after the CNOT block
        qc.barrier()

# 2. Define the map for basis change gates before measurement
basis_gate_map = {
    'X': [('h', 0), ('h', 1)],                            # H on both qubits
    'Y': [('sdg', 0), ('h', 0), ('sdg', 1), ('h', 1)],    # Apply Sdg then H to each qubit (to measure in Y basis: |0> -> |+i>, |1> -> |-i>)
    'Z': []                                               # No gates needed for Z basis
}
print("Defined CNOT pattern function (alternating) and basis gate map.")

# 3. Define the main circuit generation function
def create_measurement_circuit(M: int, basis: str) -> QuantumCircuit:
    """
    Creates a 2-qubit circuit for noise analysis.

    Structure:
    1. Initial H gates on both qubits.
    2. M alternating CNOT gates (CX(0, 1), CX(1, 0), ...).
    3. Final basis change gates (determine by 'basis').
    4. Measurement in the Z-basis.

    Args:
        M: The total number of alternating CNOT gates to apply.
        basis: The desired measurement basis ('X', 'Y', or 'Z').
    
    Returns:
        The constructed QuantumCircuit object.

    Raises:
        ValueError: If the specified basis is not 'X', 'Y', or 'Z'.
    """
    if basis not in basis_gate_map:
        raise ValueError(f"Invalid basis '{basis}'. Must be one of {list(basis_gate_map.keys())}")

    # Create the quantum circuit with 2 qubits and 2 classical bits
    qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS, name=f"M={M}_basis={basis}")

    # Apply initial Hadamard gates to create superposition state |++>
    qc.h(range(NUM_QUBITS))
    qc.barrier() # Visual separator

    # Add the sequence of M alternating CNOT gates
    add_alternating_cnots(qc, M)
    # Barrier added within add_alternating_cnots
    # Apply final basis change gates
    final_gates = basis_gate_map[basis]
    
    if final_gates:   # Only add if there are gates for this basis
        
        for gate_name, qubit_index in final_gates:
            # getattr gets the method (like qc.h, qc.sdg) by name
            getattr(qc, gate_name)(qubit_index)
        
        qc.barrier() # Visual separator
    
    # Add measurement operations
    qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))
    return qc

print("Defined main circuit generation function 'create_measurement_circuit'.")

# --- Example Usage ---
test_M = 4
test_basis = 'X'
example_circuit= create_measurement_circuit(test_M, test_basis)
print(f"\nExample circuit for M={test_M}, basis={test_basis}:")
print(example_circuit.draw('text'))

print("-" * 20 + "\n")

# --- End of Phase 3 ---

# Phase 4: Run Simulations (or Load Results)
print("--- Running Simulations or Loading Results ---")

# Initialise dictionaries to store the results (counts)
# Try loading first
ideal_counts = {}
noisy_counts = {}
simulation_duration = 0
needs_rerun = FORCE_RERUN_SIMULATIONS

# Check if results file exists and if we should load it
if ((os.path.exists(RESULTS_FILENAME)) and (not FORCE_RERUN_SIMULATIONS)):
    print(f"Loading existing results from {RESULTS_FILENAME}...")

    try:

        with open(RESULTS_FILENAME, 'r') as f:
            results_data = json.load(f)
        # Extract data (Need to convert keys back to int for M_values if loading from JSON)
        loaded_M_values = results_data.get('M_values', M_values) # Use saved M_values if available
        ideal_counts = {int(k): v for k, v in results_data.get('ideal_counts', {}).items()}
        noisy_counts = {int(k): v for k, v in results_data.get('noisy_counts', {}).items()}
        simulation_duration = results_data.get('simulation_duration_sec', 0)
        
        # Check if all M_values in the script are present in the loaded data
        if not all(m in loaded_M_values for m in M_values):
            print("Not all required M values found in the results file. Will run simulations for missing values.")
            needs_rerun = True
        
        else:
            print(f"Results loaded successfully. Simulation duration from file: {simulation_duration:.2f} seconds.") 
    
    except Exception as e:
        print(f"Error loading results file: {e}. Re-running simulations.")
        needs_rerun = True # Force re-run if loading fails.
        ideal_counts = {}  # Reset dictionaries if loading failed
        noisy_counts = {}
        simulation_duration = 0

else:
    # File doesn't exist or force re-run is True
    needs_rerun = True
    print("Results file not found or re-run forced.")

# If we need to run simulations (either all or just missing ones)
if needs_rerun:
    print("Running simulations (for all or missing M values)...")

    # Record start time only for the simulations we are running now
    simulation_start_time = time.time()
    new_sims_run = False # Flag to check if we actually ran new simulations

    # Outer loop: Iterate through the M values defined in the script
    for M in M_values:
        # Initialise inner dictionaries if M is new
        if M not in noisy_counts: noisy_counts[M] = {}
        if M not in ideal_counts: ideal_counts[M] = {}

        # Check if data for this M already exists (from loaded file)
        # We need to run if either ideal or noisy data is missing for any basis
        should_run_for_M = False

        if ((M not in ideal_counts) or (M not in noisy_counts)):
            should_run_for_M = True
        
        else:

            for basis in BASES:
                
                if ((basis not in ideal_counts.get(M, {})) or (basis not in noisy_counts.get(M, {}))):
                    should_run_for_M = True
                    break 
        
        if not should_run_for_M:
            print(f"--- M = {M}: Data already loaded. Skipping simulation. ---")
            continue # Skip to the next M value if data exists

        # If we need to run simulations for this M:
        new_sims_run = True
        print(f"--- M = {M} ---")

        # Inner loop: Iterate through the measurement bases (X, Y, Z)
        for basis in BASES:
            # Check if data for this specific basis exists before running
            if ((basis in ideal_counts.get(M, {})) and (basis in noisy_counts.get(M, {}))):
                print(f"  Skipping basis {basis} (already loaded).")
                continue

            print(f"  Simulating basis: {basis}")

            # 1. Generate the circuit for the current M and basis
            try:
                qc = create_measurement_circuit(M, basis)
        
            except ValueError as e:
                print(f"    Error generating circuit: {e}")
                continue   # Skip to next basis if circuit generation fails

            # --- Ideal Simulation ---
            try:
                # 2a. Transpile the circuit for the ideal simulator
                #   (Optimisation level 0 is often sufficient for ideal sims)
                qc_ideal_t = transpile(qc, sim_ideal, optimization_level=0)

                # 3a. Run the ideal simulation
                job_ideal = sim_ideal.run(qc_ideal_t, shots=SHOTS)
                result_ideal = job_ideal.result()

                # 4a. Get the counts and store them
                ideal_counts[M][basis] = result_ideal.get_counts(qc_ideal_t) # Pass circuit for correct key format
                print(f"    Ideal counts: {ideal_counts[M][basis]}")
        
            except Exception as e:
                print(f"    Error during IDEAL simulation for M={M}, basis={basis}: {e}")
                ideal_counts[M][basis] = {} # Store empty dict on error
        
            # --- Noisy Simulation ---
            try:
                # 2b. Transpile the circuit for the noisy simulator
                #   (Need basis_gates from noise model for better transpilation)
                #   Optimisation level might need adjustment based on noise model complexity
                qc_noisy_t = transpile(qc, sim_noise, optimization_level=1) # Use level 1 for some optimisation

                # 3b. Run the noisy simulation
                job_noisy = sim_noise.run(qc_noisy_t, shots=SHOTS)
                result_noisy = job_noisy.result()

                # 4b. Get the counts and store them
                noisy_counts[M][basis] = result_noisy.get_counts(qc_noisy_t) # Pass circuit for correct key format
                print(f"    Noisy counts: {noisy_counts[M][basis]}")
        
            except Exception as e:
                print(f"    Error during NOISY simulation for M={M}, basis={basis}: {e}")
                noisy_counts[M][basis] = {}   # Store empty dict on error

    # Record end time and calculate duration ONLY for the sims just run
    if new_sims_run:
        simulation_end_time = time.time()
        current_simulation_duration = simulation_end_time - simulation_start_time
        print(f"\n--- Simulation Complete (New M values) ---")
        print(f"Time for new simulations: {simulation_duration:.2f} seconds")
        # For simplicity, we'll just save the total data now.
        simulation_duration += current_simulation_duration # Add to total duration if needed

        # --- Save Updated Results to File (Good Practice) ---
        results_data = {
            # Save the potentially updated M_values list from the script
            'M_values': M_values,
            'bases': BASES,
            'shots': SHOTS,
            'prob_cx_pauli': prob_cx_pauli,
            # Convert M_values keys to strings for JSON compatibility
            'ideal_counts': {str(k): v for k, v in ideal_counts.items()},
            'noisy_counts': {str(k): v for k, v in noisy_counts.items()},
            'simulation_duration_sec': simulation_duration        # Save potentially updated duration
        }

        try:

            with open(RESULTS_FILENAME, 'w') as f:
                json.dump(results_data, f, indent=4) # Use indent for readability
    
            print(f"Successfully saved simulation results to {RESULTS_FILENAME}")

        except Exception as e:
            print(f"Error saving results to {RESULTS_FILENAME}: {e}")
    
    else:
        print("No new simulations were required.")

# Ensure the counts dictionaries are available if loaded/run
if ((not ideal_counts) or (not noisy_counts)):
    print("Error: Failed to load or run simulations. Counts data is missing.")
    # Exit or handle error appropriately
    exit()

print("-" * 20 + "\n")

# --- End of Phase 4 ---

# Phase 5: Calculate Survival Probabilities
print("--- Calculating Survival Probabilities (P(00) in X-basis) ---")

# Initialise list to store the P(00) values from noisy X-basis measurements
P00X_noisy_values = []

# Loop through the M values for which we have results
analysis_M_values = sorted(noisy_counts.keys()) # Use keys from loaded/run data
print(f"Analysing M values: {analysis_M_values}")

# Filter M_values from config to match available analysis_M_values for consistency
# This ensures M_values used for plotting/fitting matches the data we actually have
M_values_for_analysis = [m for m in M_values if m in analysis_M_values]

for M in M_values_for_analysis:    # Iterate through the M values we actually have data for
    # Get the noisy counts dictionary for the 'X' basis measurement for this M
    counts = noisy_counts.get(M, {}).get('X', {}) # Use .get for safety

    # Calculate the probability P(00) = counts['00'] / total_shots
    # Use .get('00', 0) to handle cases where '00' might not be present (though unlikely here)
    p00 = (counts.get('00', 0) / SHOTS)

    # Append the calculated probability to our list
    P00X_noisy_values.append(p00)
    print(f"  M = {M:2d}, P(00)_X = {p00:.4f}")

# Convert the list of probabilities to a NumPy array for easier use in fitting
P00X_noisy_values = np.array(P00X_noisy_values)

# Also convert M_values used in analysis to numpy array
analysis_M_values = np.array(M_values_for_analysis)   # Use the filtered list

# --- Sanity Check: Ideal P(00) in X-basis ---
print("\n--- Sanity Check: Ideal P(00) in X-basis ---")
ideal_p00_check_passed = True

for M in M_values_for_analysis:    # Use the filtered list
    ideal_x_counts = ideal_counts.get(M, {}).get('X', {})
    ideal_p00 = (ideal_x_counts.get('00', 0) / SHOTS)
    print(f"  M = {M:3d}, Ideal P(00)_X = {ideal_p00:.4f}")     # Adjusted formatting
    # Check if ideal result is close to 1.0
    if not np.isclose(ideal_p00, 1.0):
        print("    Warning: Ideal P(00) for M={M} is not 1.0!")
        ideal_p00_check_passed = False

if ideal_p00_check_passed:
    print("Ideal P(00)_X check passed (all are 1.0).")

else:
    print("WARING: Ideal P(00)_X check failed for some M values.")

print("-" * 20 + "\n")

# --- End of Phase 5 ---

# Phase 6: Plotting and Fitting
print("--- Plotting and Fitting Results ---")

# Check if we have data to plot
if ((len(analysis_M_values) == 0) or (len(P00X_noisy_values) == 0)):
    print("No data available for plotting and fitting. Exiting.")
    exit()

else:
    # 1. Create the plot figure
    plt.figure(figsize=(10, 6)) # Set figure size for better readability

    # 2. Plot the noisy simulation data points
    plt.scatter(analysis_M_values, P00X_noisy_values, label='Noisy Simulation $P(00)$ (X-basis)', marker='o', color='blue')

    # 3. Define the fittin function (Model: A * p^M + 0.25)
    def fit_func(m, A, p):
        """
        Exponential decay model with baseline.
        """
        return A * (p**m) + 0.25

    # 4. Define bounds for the parameters [A, p]
    #   0 <= A <= 0.75
    #   0 <= p <= 1.0
    bounds = ([0, 0], [0.75, 1.0])

    # 5. Provide initial guesses for parameters [A, p] to help the fitter
    #   Guess A starts near 0.75, guess p starts near 1 (but slightly less)
    initial_guesses = [0.7, 0.99]

    # 6. Perform the curve fitting using scipy.optimize.curve_fit
    try:
        # Increased maxfev just in case
        popt, pcov = curve_fit(fit_func, analysis_M_values, P00X_noisy_values, p0=initial_guesses, bounds=bounds, maxfev=5000)
        A_fit, p_fit = popt

        # 7. Print the fitted parameters
        print(f"Curve Fit Results:")
        print(f"  A = {A_fit:.4f}")
        print(f"  p = {p_fit:.4f}")
        
        # Print standard errors
        try:
            perr = np.sqrt(np.diag(pcov))
            print(f"  Std Error: A_err = {perr[0]:.4f}, p_err = {perr[1]:.4f}")
        
        except Exception:       # Handle cases where covariance matrix is ill-defined
            print("  Could not estimate standard errors for fit parameters.")

        # 8. Generate points for the fitted curve for smooth plotting
        m_plot = np.linspace(min(analysis_M_values), max(analysis_M_values), 200) # More points for smooth curve
        p00_fit = fit_func(m_plot, A_fit, p_fit)

        # 9. Plot the fitted curve
        plt.plot(m_plot, p00_fit, 'r-', label=f'Fit: $A p^M + 0.25$\n$A={A_fit:3f}$, $p={p_fit:.3f}$')

    except RuntimeError as e:
        print(f"Error during curve fitting: {e}")
        print("Could not determine fit parameters. Plotting data points only.")

    except Exception as e:
        print(f"An unexpected error occurred during fitting: {e}")

    # 10. Finalise the plot
    plt.xlabel("Number of Alternating CNOT Gates (M)")
    plt.ylabel("Survival Probability P(00) (X-basis)")
    plt.title(f"Survival Probability vs M (CX Error Prob: {prob_cx_pauli*100:.1f}%)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=0) # Ensure y-axis starts at 0

    # 11. Save the plot to a file
    try:
        plt.savefig(PLOT_FILENAME, dpi=300, bbox_inches='tight') # Save with good resolution
        print(f"Successfully saved plot to {PLOT_FILENAME}")
    
    except Exception as e:
        print(f"Error saving plot to {PLOT_FILENAME}: {e}")
    
    print("-" * 20 + "\n")

    # --- End of Phase 6 ---

    print("--- Script Finished ---")