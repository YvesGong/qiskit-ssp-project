# Phase 0: Setup and Essential Definitions

# --- Standard Libraries ---
import numpy as np
import pandas as pd
import pickle
import os
import glob                             # For finding files matching a pattern
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Configuration Constants ---
print("--- Configuration ---")
NUM_QUBITS = 3 # The experiment was run on 3 qubits
SHOTS = 4096   # Number of times each circuit was run

# M values used in the experiment (number of CNOT block iterations)
M_values = np.array([2, 4, 8, 16, 32, 64])

# The 27 three-qubit basis strings in the exact order the result files are saved
# This order is crucial for correctly mapping result files to basis settings.
BASES_3Q = [
    'XXX', 'XYX', 'XZX', 'YXY', 'YYY', 'YZY', 'ZXZ', 'ZYZ', 'ZZZ',
    'XXY', 'XYY', 'XZY', 'YXX', 'YYX', 'YZX', 'XXZ', 'XYZ', 'XZZ',
    'ZXX', 'ZYX', 'ZZX', 'YXZ', 'YYZ', 'YZZ', 'ZXY', 'ZYY', 'ZZY'
]
RESULTS_DIR = "results" # Directory where our pickle files (count_0.pickle, etc.) are stored
print(f"Number of Qubits: {NUM_QUBITS}")
print(f"Shots per circuit: {SHOTS}")
print(f"M values to analyse: {M_values}")
print(f"Number of 3-qubit basis settings: {len(BASES_3Q)}")
print(f"Results directory: {RESULTS_DIR}")

# --- Definitions for 2-Qubit Subsystem (q2, q0) Analysis ---

print("\n--- Definitions for (q2, q0) Subsystem Analysis ---")
# Order of Pauli terms corresponding to the output of the 4x4 WHT when applied to P(0_0), P(0_1), P(1_0), P(1_1)
# This assumes q0 is the "first" qubit in the 2-qubit WHT context (rightmost) and q2 is the "second" qubit (leftmost).
# If WHT output is [lambda_II, lambda_IZ, lambda_ZI, lambda_ZZ] for Z-basis measurements, then for X-basis it's [lambda_II, lambda_IX, lambda_XI, lambda_XX] etc.
# We will map these effective eigenvalues to the fundamental Pauli terms later.
# The WHT_MATRIX_4x4 below assumes input [P00, P01, P10, P11] -> output [II, IZ, ZI, ZZ] for Z basis.
# For X basis, it would be [P++, P+-, P-+, P--] -> [II, IX, XI, XX] (mapping 0->+, 1->-)
# For Y basis, it would be [P(y0+y2+), P(y0-y2+), P(y0+y2-), P(y0-y2-)] -> [II, IY, YI, YY]
PAULI_SUBSYSTEM_EFFECTIVE_ORDER = ['II', 'IB0', 'B2I', 'B2B0'] # Placeholder names for WHT output
# Walsh-Hadamard Transform matrix for 2 qubits
# Input: [P(q2=0,q0=0), P(q2=0,q0=1), P(q2=1,q0=0), P(q2=1,q0=1)]
# Output: Eigenvalues corresponding to [II, I(B0), (B2)I, (B2)(B0)] where B0 is the basis of q0 and B2 is the basis of q2.
WHT_MATRIX_4x4 = np.array([
    [1, 1, 1, 1],
    [1, -1, 1, -1],    # Corresponds to measuring sigma_z on q0 (Pauli B0)
    [1, 1, -1, -1],    # Corresponds to measuring sigma_z on q2 (Pauli B2)
    [1, -1, -1, 1]     # Corresponds to measuring sigma_z on q0 AND q2 (Pauli B2B0)
])
# Note: This WHT matrix definition gives expectation values.
# For Pauli channel eigenvalues (lambda_P = sum_Q chi_PQ p_Q), the matrix elements are +/-1.
# The PTM eigenvalues are directly related to these expectation values.
# A common convention for PTM eigenvalues from probabilities P(s0,s1) is:
# lambda_II = P00+P01+P10+P11 (=1)
# lambda_IZ = P00-P01+P10-P11 (corresponds to <Z_0>)
# lambda_ZI = P00+P01-P10-P11 (corresponds to <Z_2>)
# lambda_ZZ = P00-P01-P10+P11 (corresponds to <Z_2 Z_0>)
# This matches the WHT_MATRIX_4x4 rows if we map P(q2,q0) with q0 as the first index in P_ij.
# The 15 fundamental non-identity 2-qubit Pauli terms for the (q2, q0) subsystem (q2 is the first char, q0 is the second char)
FUNDAMENTAL_2Q_PAULIS = [
    'IX', 'IY', 'IZ',
    'XI', 'XX', 'XY', 'XZ',
    'YI', 'YX', 'YY', 'YZ',
    'ZI', 'ZX', 'ZY', 'ZZ'
]
print(f"Walsh-Hadamard Matrix (4x4):\n{WHT_MATRIX_4x4}")
print(f"Fundamental 2Q Pauli terms to analyse for (q2, q0): {FUNDAMENTAL_2Q_PAULIS}")
print("-" * 20 + "\n")

# --- Fitting Function Definition ---

def decay_model(M, A, p_decay):
    """Exponential decay model for fitting: A * (p_decay^M)"""
    return A * (p_decay**M)

# --- Placeholder for results storage (will be populated in later phases) ---
# This will store the raw eigenvalue series for each 3Q basis and effective Pauli term
# raw_eigenvalue_series[basis_3q_str][effective_pauli_term_str] = [list_of_6_eigenvalues_vs_M]
raw_eigenvalue_series = {}

# This will store lists of eigenvalue series, aggregated by fundamental 2Q Pauli term
# aggregated_eigenvalue_series['IX'] = [series1_from_XXX, series2_from_YXX, ...]
aggregated_eigenvalue_series = {pauli: [] for pauli in FUNDAMENTAL_2Q_PAULIS}

# This will store the final fitted parameters (A, p) for each fundamental 2Q Pauli term
final_fitted_params = {}
print("Phase 0: Setup and Definitions Complete.")

# --- Phase 1: Data Loading and Initial Eigenvalue Extraction ---

print("--- Phase 1: Data Loading and Initial Eigenvalue Extraction ---")

# Helper Function: Load data from a single file
def load_results_file(filepath):
    """
    Loads and returns data from a pickle file.
    """
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        # Assuming data is a list of 6 qiskit_ibm_runtime.utils.BitArray objects (or similar containing counts)
        if ((not isinstance(data, list)) or (len(data) != len(M_values))):
            print(f"Warning: Data in {filepath} is not a list of expected length {len(M_values)}. Found type {type(data)} with length {len(data) if isinstance(data, list) else 'N/A'}.")
            # Depending on the exact structure, we might need to adjust how counts are extracted.
            # For now, we'll assume it's a list of objects that have a .get_counts() method.
        
        return data

    except FileNotFoundError:
        print(f"Error: Results file not found at {filepath}")
        return None
    
    except Exception as e:
        print(f"Error loading pickle file {filepath}: {e}")
        return None

# Helper Function: Convert Qiskit BitArray (or similar) to a standard counts dictionary
def get_qiskit_counts_from_bitarray(bit_array_data):
    """
    Converges a Qiskit BitArray object (from SamplerV2 results) into a standard counts dictionary {'bitstring': count}.
    """
    if (hasattr(bit_array_data, 'get_counts')):
        return bit_array_data.get_counts()

    else:
        # Fallback or error if the structure is different than expected
        print(f"Warning: Loaded data object (type: {type(bit_array_data)}) does not have .get_counts(). We may need to adjust data extraction.")
        # Example: if bit_array_data is already a counts dict:
        # if isinstance(bit_array_data, dict): return bit_array_data
        return {} # Return empty dict to avoid crashing, but flag the issue

# Helper Function: Calculate 2-qubit marginal probabilities for (q2, q0)
def calculate_marginal_probabilities(counts_3q, total_shots):
    """
    Calculates 2-qubit marginal probabilities for the (q2, q0) subsystem.
    Input: 3-qubit counts dictionary, total_shots.
    Output: NumPy array [P(q2=0,q0=0), P(q2=0,q0=1), P(q2=1,q0=0), P(q2=1,q0=1)].
    Qiskit convention: q2q1q0
    """
    p_0_0 = (counts_3q.get('000', 0) + counts_3q.get('010', 0)) / total_shots
    p_0_1 = (counts_3q.get('001', 0) + counts_3q.get('011', 0)) / total_shots
    p_1_0 = (counts_3q.get('100', 0) + counts_3q.get('110', 0)) / total_shots
    p_1_1 = (counts_3q.get('101', 0) + counts_3q.get('111', 0)) / total_shots
    return np.array([p_0_0, p_0_1, p_1_0, p_1_1])

# Helper Function: Apply WHT to get effective eigenvalues
def get_effective_eigenvalues(prob_vector_2q, wht_matrix):
    """
    Applies the Walsh-Hadamard Transform to the 2-qubit probability vector.
    Output order: [lambda_II, lambda_IB0, lambda_B2I, lambda_B2B0]
    """
    return wht_matrix @ prob_vector_2q

# --- Main Data Extraction Loop ---

# raw_eigenvalue_series dictionary is initialised in Phase 0
# raw_eigenvalue_series[basis_3q_str][effective_pauli_term_str] = [list_of_6_eigenvalues_vs_M]

print(f"Processing {len(BASES_3Q)} basis settings...")

for idx, current_3q_basis_str in enumerate(BASES_3Q):
    # Adjust file indexing: BASES_3Q[0] maps to count_1.pickle, etc.
    file_index = idx + 1
    filepath = os.path.join(RESULTS_DIR, f"count_{file_index}.pickle")
    print(f"  Loading data for basis: {current_3q_basis_str} from {filepath}...")
    data_for_6_Ms = load_results_file(filepath)

    if data_for_6_Ms is None:
        print(f"    Skipping basis {current_3q_basis_str} due to missing or unreadable file.")
        raw_eigenvalue_series[current_3q_basis_str] = {} # Ensure key exists
        continue

    if (len(data_for_6_Ms) != len(M_values)):
        print(f"    Warning: Data for {current_3q_basis_str} has {len(data_for_6_Ms)} M-values, expected {len(M_values)}. Skipping.")
        raw_eigenvalue_series[current_3q_basis_str] = {} # Ensure key exists
        continue

    # Initialise lists to store eigenvalue series for this current_3q_basis_str
    # These will hold the eigenvalues corresponding to I_2 B_2, B_2 I_0, and B_2 B_0 for the (q2,q0) subsystem
    series_lambda_IB0_eff = [] # Effective eigenvalue for I (on q2) and B0 (on q0)
    series_lambda_B2I_eff = [] # Effective eigenvalue for B2 (on q2) and I (on q0)
    series_lambda_B2B0_eff = [] # Effective eigenvalue for B2 (on q2) and B0 (on q0)

    for m_idx, m_data_raw in enumerate(data_for_6_Ms):
        # m_data_raw is expected to be a Qiskit BitArray or similar object for one M-value
        counts_3q = get_qiskit_counts_from_bitarray(m_data_raw)

        if not counts_3q: # If counts are empty (e.g., due to error in get_qiskit_counts)
            print(f"    Warning: Could not get counts for M={M_values[m_idx]} in {current_3q_basis_str}. Appending NaNs.")
            series_lambda_IB0_eff.append(np.nan)
            series_lambda_B2I_eff.append(np.nan)
            series_lambda_B2B0_eff.append(np.nan)
            continue

        prob_vector_2q = calculate_marginal_probabilities(counts_3q, SHOTS)
        lambda_eff_vector = get_effective_eigenvalues(prob_vector_2q, WHT_MATRIX_4x4)
        # WHT_MATRIX_4x4 output order: [lambda_II, lambda_IB0, lambda_B2I, lambda_B2B0]
        # We ignore lambda_II (lambda_eff_vector[0]) for decay fitting.
        series_lambda_IB0_eff.append(lambda_eff_vector[1])  # Corresponds to I_2 B_0
        series_lambda_B2I_eff.append(lambda_eff_vector[2])  # Corresponds to B_2 I_0
        series_lambda_B2B0_eff.append(lambda_eff_vector[3]) # Corresponds to B_2 B_0
    
    # Store these series in the main dictionary
    # The keys for the inner dict will be the fundamental Pauli strings for (q2,q0)
    # e.g., if current_3q_basis_str is 'XYX':
    # B0_char = 'X' (for q0)
    # B2_char = 'X' (for q2, because basis_str[0] is X)
    # No, B2 is basis_str[0], B0 is basis_str[2]
    B2_char = current_3q_basis_str[0] # Basis for q2 (leftmost in 3Q string, first in 2Q Pauli string)
    # q1_basis_char = current_3q_basis_str[1] # Basis for q1 (middle, ignored for (q2, q0) subsystem)
    B0_char = current_3q_basis_str[2] # Basis for q0 (rightmost in 3Q string, second in 2Q Pauli string)
    # Construct the fundamental 2-qubit Pauli strings for the (q2,q0) subsystem based on the measurement settings B2 and B0 for the current 3-qubit experiment
    effective_pauli_IB0 = 'I' + B0_char      # e.g., if BO='X', this is 'IX'
    effective_pauli_B2I = B2_char + 'I'      # e.g., if B2='Y', this is 'YI'
    effective_pauli_B2B0 = B2_char + B0_char # e.g., if B2='Y', B0='X', this is 'YX'
    raw_eigenvalue_series[current_3q_basis_str] = {
        effective_pauli_IB0: np.array(series_lambda_IB0_eff),
        effective_pauli_B2I: np.array(series_lambda_B2I_eff),
        effective_pauli_B2B0: np.array(series_lambda_B2B0_eff)
    }
    print(f"    Stored effective eigenvalue series for {current_3q_basis_str}: {effective_pauli_IB0}, {effective_pauli_B2I}, {effective_pauli_B2B0}")

print("\nPhase 1: Initial Eigenvalue Extraction Complete.")
# At this point, raw_eigenvalue_series contains the data needed for Phase 2 (Grouping)
# Example: How to access data for a specific case after this phase
# if 'XXX' in raw_eigenvalue_series and 'IX' in raw_eigenvalue_series['XXX']:
#   print(f"\nExample: Eigenvalue series for IB0 (IX) from 'XXX' experiments (vs M):")
#   print(raw_eigenvalue_series['XXX']['IX'])

# --- Phase 2: Grouping Eigenvalue Series by Fundamental 2-Qubit Pauli Terms ---

print("--- Phase 2: Grouping Eigenvalue Series by Fundamental 2-Qubit Pauli Terms ---")
# The 'aggregated_eigenvalue_series' dictionary was initialised in Phase 0:
# aggregated_eigenvalue_series = {pauli: [] for pauli in FUNDAMENTAL_2Q_PAULIS}
# It will store lists of eigenvalue series. For example:
# aggregated_eigenvalue_series['IX'] will be a list of NumPy arrays, where each array is a series of 6 eigenvalues (vs. M) for the I2X0 term obtained from one of the 3-qubit experiments.

if not raw_eigenvalue_series:
    print("Error: 'raw_eigenvalue_series' is empty. Please ensure Phase 1 ran successfully.")

else:
    
    for basis_3q_str, effective_pauli_data in raw_eigenvalue_series.items():

        if not effective_pauli_data:
            print(f"  Warning: No data found for 3-qubit basis {basis_3q_str} in raw_eigenvalue_series. Skipping.")
            continue

        # effective_pauli_data is a dictionary like:
        # {'IX': np.array([...]), 'XI': np.array([...]), 'XX': np.array([...])} where the keys are the effective 2-qubit Pauli terms for the (q2, q0) subsystem based on the measurement settings B2 and B0 of the current 3-qubit experiment.
        for effective_pauli_term_str, eigenvalue_series_vs_M in effective_pauli_data.items():

            if effective_pauli_term_str in FUNDAMENTAL_2Q_PAULIS:
                # Append a dictionary containing source and data
                aggregated_eigenvalue_series[effective_pauli_term_str].append({
                    'source_3q_basis': basis_3q_str,
                    'series_data': eigenvalue_series_vs_M
                })
            
            else:
                # This case should ideally not happen if keys were constructed correctly in Phase 1 and FUNDAMENTAL_2Q_PAULIS is comprehensive for non-identity terms.
                print(f"  Warning: Encounted an unexpected effective Pauli term '{effective_pauli_term_str}' for 3Q-basis '{basis_3q_str}'. It will not be aggregated.")

    print("\nAggregation complete. Number of eigenvalue series collected for each fundamental 2Q Pauli term:")

    for pauli_term, series_list_of_dicts in aggregated_eigenvalue_series.items():
        print(f"  {pauli_term}: {len(series_list_of_dicts)} series")
        # We can add more detailed checks here, e.g., verify all series have the correct length.
        for i, entry in enumerate(series_list_of_dicts):

            if not ((isinstance(entry, dict)) and ('series_data' in entry) and (isinstance(entry['series_data'], np.ndarray)) and (len(entry['series_data']) == len(M_values))):
                source = entry.get('source_3q_basis', 'Unknown')
                data_type = type(entry.get('series_data'))
                data_len = len(entry.get('series_data'))
                print(f"    Warning: Entry {i} for {pauli_term} (source: {source}), 'series_data' invalid. Type: {data_type}, Len: {data_len}.")
            

print("\nPhase 2: Grouping Eigenvalue Series Complete.")
# At this point, aggregated_eigenvalue_series is populated.
# For example, aggregated_eigenvalue_series['IX'] will contain nine NumPy arrays (one from 'XXX', one from 'YXX', one from 'ZXX', etc., where q0 was X and q2 was effectively I).
# And aggregated_eigenvalue_series['XX'] will contain three NumPy arrays (one from 'XXX', one from 'XYX', one from 'XZX').
# Example: How to access data after this phaase
# if aggregated_eigenvalue_series['XX']:
#   print(f"\nExample: First eigenvalue series collected for XX (from the first relevant 3Q experiment):")
#   print(aggregated_eigenvalue_series['XX'][0])
#   print(f"Total series collected for XX: {len(aggregated_eigenvalue_series['XX'])}")

# --- Phase 3: Visualisation for Variance Assessment ---

print("--- Phase 3: Visualisation for Variance Assessment ---")
# Directory to save these variance assessment plots
VARIANCE_PLOT_DIR = "plots_variance_assessment"

if not os.path.exists(VARIANCE_PLOT_DIR):
    os.makedirs(VARIANCE_PLOT_DIR)
print(f"Variance assessment plots will be saved to: {VARIANCE_PLOT_DIR}")

if not aggregated_eigenvalue_series:
    print("Error: 'aggregated_eigenvalue_series' is empty or not defined. Please ensure Phase 2 ran successfully.")

else:
    for fundamental_pauli_term in FUNDAMENTAL_2Q_PAULIS:
        list_of_context_entries = aggregated_eigenvalue_series.get(fundamental_pauli_term, [])

        if not list_of_context_entries:
            print(f"  No eigenvalue series found for fundamental Pauli term: {fundamental_pauli_term}. Skipping plot.")
            continue
    
        plt.figure(figsize=(12, 7))
        num_series_plotted = 0

        for i, context_entry in enumerate(list_of_context_entries):
            
            series_vs_M = context_entry.get('series_data')
            source_3q_basis_label = context_entry.get('source_3q_basis', f'Unknown Context {i+1}')

            if ((isinstance(series_vs_M, np.ndarray)) and (len(series_vs_M) == len(M_values))):
                # Plot each individual series. Alpha makes overlapping lines more visible.
                plt.plot(M_values, series_vs_M, marker='o', linestyle='-', alpha=0.6, label=f'Context {source_3q_basis_label}')
                num_series_plotted += 1
            
            else:
                print(f"    Warning: Invalid or mismatched series data for {fundamental_pauli_term}, context{source_3q_basis_label}. Skipping this series.")
        
        if (num_series_plotted == 0):
            print(f"  No valid series were plotted for {fundamental_pauli_term}. Skipping plot generation.")
            plt.close() # Close the empty figure
            continue
        
        plt.xlabel("Number of CNOT Block Iterations (M)")
        plt.ylabel(rf"Effective Eigenvalue ($\lambda_{{{fundamental_pauli_term}}}$)")
        plt.title(rf"Variance assessment for (q2,q0) Term: $\lambda_{{{fundamental_pauli_term}}}$\n(Each line is a different q1 basis context)")

        # Only show legend if there are few lines, otherwise it gets cluttered
        if (num_series_plotted <= 10): # Arbitrary threshold for legend readability
            plt.legend(title="3Q Basis Context", loc="best", fontsize='small')
        
        else:
            print(f"  More than 10 series for {fundamental_pauli_term}, omitting legend for clarity.")
        
        plt.grid(True, which='both', linestyle=':', linewidth=0.7)
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--') # Add a y=0 line for reference
        all_values_list = [entry['series_data'] for entry in list_of_context_entries if ((isinstance(entry.get('series_data'), np.ndarray)) and (len(entry['series_data']) == len(M_values)))]
        # Set y-limits based on typical eigenvalue range, e.g., -1 to 1, but allow for overshoot
        min_val = -1.1
        max_val = 1.1

        if all_values_list:
            all_values = np.concatenate(all_values_list)

            if (all_values.size > 0):
                min_val_data, max_val_data = np.nanmin(all_values), np.nanmax(all_values)
                min_val = min(min_val_data - 0.1, -0.1) if not np.isnan(min_val_data) else -1.1
                max_val = max(max_val_data + 0.1, 0.1) if not np.isnan(max_val_data) else 1.1
        
        plt.ylim(min_val, max_val)
        plot_filename = os.path.join(VARIANCE_PLOT_DIR, f"variance_{fundamental_pauli_term}.png")

        try:
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            print(f"  Saved variance plot to: {plot_filename}")
        
        except Exception as e:
            print(f"  Error saving plot {plot_filename}: {e}")
        
        plt.close()

print("\nPhase 3: Visualisation for Variance Assessment Complete.")

# --- Phase 4: Fitting and Plotting Decay Curves (Detailed Analysis) ---

print("--- Phase 4: Fitting and Plotting Decay Curves (Detailed Analysis) ---")

# Directory to save these detailed fit plots
DETAILED_FIT_PLOT_DIR = "plots_detailed_fits"

if not os.path.exists(DETAILED_FIT_PLOT_DIR):
    os.makedirs(DETAILED_FIT_PLOT_DIR)

print(f"Detailed fit plots will be saved to: {DETAILED_FIT_PLOT_DIR}")

# final_fitted_params dictionary was initialised in Phase 0.
# it will be structured as:
# final_fitted_params[fundamental_pauli_term][source_3q_basis_str] = {'A': A_fit, 'p': p_fit, 'cov': pcov}

if not aggregated_eigenvalue_series:
    print("Error: 'aggregated_eigenvalue_series' is empty. Please ensure Phase 2 ran successfully.")

else:
    
    for fundamental_pauli_term, list_of_series_with_source in aggregated_eigenvalue_series.items():

        if not list_of_series_with_source:
            print(f"  No eigenvalue series found for fundamental Pauli term: {fundamental_pauli_term}. Skipping.")
            continue

        print(f"\n Fitting and plotting for fundamental term: {fundamental_pauli_term}")
        final_fitted_params[fundamental_pauli_term] = {} # initialise inner dict

        for context_entry in list_of_series_with_source:

            if not isinstance(context_entry, dict): # Added safety check
                print(f"    Error: context_entry for {fundamental_pauli_term} is not a dict! Value: {context_entry}")

            source_3q_basis = context_entry['source_3q_basis']
            current_series_to_fit = context_entry['series_data']

            if ((current_series_to_fit is None) or (len(current_series_to_fit) != len(M_values)) or (np.isnan(current_series_to_fit).any())):
                print(f"    Skipping fit for {fundamental_pauli_term} (context: {source_3q_basis}) due to invalid data (None, wrong length, or NaNs).")
                final_fitted_params[fundamental_pauli_term][source_3q_basis] = {'A': np.nan, 'p': np.nan, 'cov': np.empty((2,2))*np.nan, 'error': 'Invalid data'}
                continue

            # --- Curve Fitting ---
            try:
                # Initial guesses: A can be around the first data point, p around 0.9
                # Bounds: A can be negative for eigenvalues, p between 0 and 1.
                initial_A = current_series_to_fit[0] if current_series_to_fit[0] != 0 else (0.5 if np.mean(current_series_to_fit) > 0 else -0.5)
                p0 = [initial_A, 0.9]
                bounds = ([-1.5, 0], [1.5, 1.0]) # A between -1.5 and 1.5, p_decay between 0 and 1
                popt, pcov = curve_fit(decay_model, M_values, current_series_to_fit, p0=p0, bounds=bounds, maxfev=10000)
                A_fit, p_fit = popt
                # Store fitted parameters
                final_fitted_params[fundamental_pauli_term][source_3q_basis] = {'A': A_fit, 'p': p_fit, 'cov': pcov, 'error': None}
                print(f"    Fit for {fundamental_pauli_term} (context: {source_3q_basis}): A={A_fit:.4f}, p={p_fit:.4f}")
            
            except RuntimeError as e:
                print(f"    RuntimeError during curve fitting for {fundamental_pauli_term} (context: {source_3q_basis}): {e}")
                final_fitted_params[fundamental_pauli_term][source_3q_basis] = {'A': np.nan, 'p': np.nan, 'cov': np.empty((2,2))*np.nan, 'error': str(e)}
                A_fit, p_fit = np.nan, np.nan # Ensure these are defined for plotting
            
            except Exception as e_gen:
                print(f"    Unexpected error during fitting for {fundamental_pauli_term} (context: {source_3q_basis}): {e_gen}")
                final_fitted_params[fundamental_pauli_term][source_3q_basis] = {'A': np.nan, 'p': np.nan, 'cov': np.empty((2,2))*np.nan, 'error': str(e_gen)}
                A_fit, p_fit = np.nan, np.nan
            
            # --- Plotting ---

            plt.figure(figsize=(10,6))
            plt.plot(M_values, current_series_to_fit, 'o', label=rf'Experimental $\lambda$ (Context: {source_3q_basis})', color='blue', markersize=7)

            if not np.isnan(p_fit): # Only plot fit if successful
                M_smooth = np.linspace(min(M_values), max(M_values), 200)
                fit_curve = decay_model(M_smooth, A_fit, p_fit)
                plt.plot(M_smooth, fit_curve, 'r-', label=rf'Fit: $A \cdot p^M$\n$A={A_fit:.3f}, p={p_fit:.3f}$')
            
            plt.xlabel("Number of CNOT Block Iterations (M)")
            plt.ylabel(rf"Effective Eigenvaue ($\lambda_{{{fundamental_pauli_term}}}$)")
            plt.title(rf"Decay of $\lambda_{{{fundamental_pauli_term}}}$ for (q2,q0)\nContext (3Q Basis): {source_3q_basis}")
            plt.legend(loc="best")
            plt.grid(True, which='both', linestyle=':', linewidth=0.7)
            plt.axhline(0, color='black', linewidth=0.5, linestyle='--')

            # Determine y-limits dynamically but keep them reasonable
            min_data_val = np.nanmin(current_series_to_fit) if current_series_to_fit.size > 0 else -1.1
            max_data_val = np.nanmax(current_series_to_fit) if current_series_to_fit.size > 0 else 1.1
            y_lower = min(min_data_val - 0.1, -0.1) if not np.isnan(min_data_val) else -1.1
            y_upper = max(max_data_val + 0.1, 0.1) if not np.isnan(max_data_val) else 1.1
            plt.ylim(max(-1.2, y_lower), min(1.2, y_upper)) # Cap at +/-1.2 for typical eigenvalue plots
            plot_filename = os.path.join(DETAILED_FIT_PLOT_DIR, f"fit_{fundamental_pauli_term}_context_{source_3q_basis}.png")

            try:
                plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                print(f"    Saved plot: {plot_filename}")
            
            except Exception as e_save:
                print(f"    Error saving plot {plot_filename}: {e_save}")
            
            plt.close()

print("\nPhase 4: Detailed Fitting and Plotting Complete.")

# Example": How to access fitted parameters after this phase
# if (('XX' in final_fitted_params) and ('XXX' in final_fitted_params['XX'])):
#   xx_xxx_params = final_fitted_params['XX']['XXX']
#   print(f"\nExample: Fitted params for XX term from 'XXX' context: A={xx_xxx_params['A']:.4f}, p={xx_xxx_params['p']:.4f}")

# --- Phase 5: Results Summary ---

print("--- Phase 5: Results Summary ---")

if not final_fitted_params:
    print("Error: 'final_fitted_params' is empty. Please ensure Phase 4 ran successfully and produced fits.")

else:
    results_for_table = []

    # Iterate through the fundamental Pauli terms in a defined order for consistency
    for fundamental_pauli_term in FUNDAMENTAL_2Q_PAULIS:
        context_data = final_fitted_params.get(fundamental_pauli_term)

        if not context_data:
            print(f"  Not fit data found for fundamental term: {fundamental_pauli_term}")
            results_for_table.append({
                'Pauli Term (q2,q0)': fundamental_pauli_term,
                '3Q Basis Context': 'N/A',
                'Fitted A': np.nan,
                'Std Err A': np.nan,
                'Fitted p': np.nan,
                'Std Err p': np.nan,
                'Fit Error Msg': 'No contexts found'
            })
            continue

        # Sort contexts for consistent table output
        sorted_contexts = sorted(context_data.keys())

        for source_3q_basis in sorted_contexts:
            params = context_data.get(source_3q_basis)
            
            if not params: # Should not happen if keys are present
                results_for_table.append({
                    'Pauli Term (q2,q0)': fundamental_pauli_term,
                    '3Q Basis Context': source_3q_basis,
                    'Fitted A': np.nan,
                    'Std Err A': np.nan,
                    'Fitted p': np.nan,
                    'Std Err p': np.nan,
                    'Fit Error Msg': 'Params not found'
                })
                continue
        
            A_fit = params.get('A', np.nan)
            p_fit = params.get('p', np.nan)
            cov_matrix = params.get('cov')
            fit_error_msg = params.get('error', None)
            A_err, p_err = np.nan, np.nan

            if ((fit_error_msg is None) and (cov_matrix is not None) and (not np.all(np.isnan(cov_matrix)))):
                
                try:
                    # Ensure diagonal elements are non-negative for sqrt
                    diag_cov = np.diag(cov_matrix)

                    if (np.any(diag_cov < 0)): # Check for negative variances
                        print(f"    Warning: Negative variance in cov matrix for {fundamental_pauli_term}, context {source_3q_basis}. Errors set to NaN.")
                        pass # Keep A_err, p_err as NaN

                    else:
                        perr = np.sqrt(diag_cov)
                        A_err, p_err = perr[0], perr[1]
                    
                except Exception as e_stderr:
                    print(f"    Could not estimate standard errors for {fundamental_pauli_term}, context {source_3q_basis}: {e_stderr}")
                    pass # Keep A_err, p_err as NaN
            
            results_for_table.append({
                'Pauli Term (q2,q0)': fundamental_pauli_term,
                '3Q Basis Context': source_3q_basis,
                'Fitted A': A_fit,
                'Std Err A': A_err,
                'Fitted p': p_fit,
                'Std Err p': p_err,
                'Fit Error Msg': fit_error_msg if fit_error_msg else ''
            })
    
    # Create a Pandas DataFrame for nice printing and potential CSV export
    results_df = pd.DataFrame(results_for_table)

    # Set display options for Pandas
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)                    # Adjust width for our terminal
    pd.set_option('display.float_format', '{:.4f}'.format) # Format floats
    print("\n--- Summary of Fitted Decay Parameters (A * p^M) ---")
    print(results_df)

    # Save to CSV
    csv_filename = "fitted_decay_parameters_detailed.csv"

    try:
        results_df.to_csv(csv_filename, index=False, float_format='%.6f')
        print(f"\nResults table saved to: {csv_filename}")
    
    except Exception as e_csv:
        print(f"\nError saving results to CSV {csv_filename}: {e_csv}")

print("\nPhase 5: Results Summary Complete.")

# --- Phase 6: Averaging Fitted 'p' Values by q1 Basis ---

print("\n--- Phase 6: Averaging Fitted 'p' Values by q1 Basis ---")

if (('final_fitted_params' not in globals()) or (not final_fitted_params)):
    print("Error: 'final_fitted_params' is not defined or is empty. Please ensure Phase 4 ran successfully.")
    # Depending on the environment, we might exit or raise an error here.
    # For now, we'll just print an error and try to continue, though it might fail.
    final_fitted_params = {} # Initialise to prevent further errors if it's missing

# This dictionary will store the averaged p values.
# Structure: averaged_p_values[pauli_term_2q] = {'q1_X_avg_p': ..., 'q1_Y_avg_p': ..., 'q1_Z_avg_p': ...}
averaged_p_values_by_q1_basis = {}

for pauli_term_2q in FUNDAMENTAL_2Q_PAULIS:
    # Initialise lists to store p_values for the current pauli_term_2q, grouped by the basis of q1.
    p_values_for_q1_X = []
    p_values_for_q1_Y = []
    p_values_for_q1_Z = []
    
    if pauli_term_2q in final_fitted_params:
        # context_data_for_pauli is a dictionary where keys are source_3q_basis_strings
        # (e.g., 'XXX', 'YXY') and values are dictionaries of fit parameters {'A': ..., 'p': ..., ...}
        context_data_for_pauli = final_fitted_params[pauli_term_2q]

        for source_3q_basis, params_dict in context_data_for_pauli.items():
            # Ensure source_3q_basis is a valid 3-character string
            if not (isinstance(source_3q_basis, str) and (len(source_3q_basis) == 3)):
                print(f"  Warning: Invalid source_3q_basis format '{source_3q_basis}' for Pauli term {pauli_term_2q}. Skipping.")
                continue

            q1_basis = source_3q_basis[1]  # The middle character determines q1's basis
            p_value = params_dict.get('p') # Get the fitted 'p' value

            # Check if p_value is valid (not None and not NaN)
            if ((p_value is not None) and (not np.isnan(p_value))):

                if (q1_basis == 'X'):
                    p_values_for_q1_X.append(p_value)
                
                elif (q1_basis == 'Y'):
                    p_values_for_q1_Y.append(p_value)
                
                elif (q1_basis == 'Z'):
                    p_values_for_q1_Z.append(p_value)
                
                else:
                    print(f"  Warning: Unknown q1_basis '{q1_basis}' from source '{source_3q_basis}' for Pauli {pauli_term_2q}. Skipping p-value.")
            
            # If p_value is None or NaN (e.g., fit failed), it's implicitly skipped for averaging.
            # We could add an 'else' here to log skipped p_values if desired.
        
        # Calculate the average p_value for each q1 basis group
        # np.mean of an empty list results in NaN, which is desired behaviour.
        avg_p_q1_X = np.mean(p_values_for_q1_X) if p_values_for_q1_X else np.nan
        avg_p_q1_Y = np.mean(p_values_for_q1_Y) if p_values_for_q1_Y else np.nan
        avg_p_q1_Z = np.mean(p_values_for_q1_Z) if p_values_for_q1_Z else np.nan
        averaged_p_values_by_q1_basis[pauli_term_2q] = {
            'q1_X_avg_p': avg_p_q1_X,
            'q1_X_count': len(p_values_for_q1_X),
            'q1_Y_avg_p': avg_p_q1_Y,
            'q1_Y_count': len(p_values_for_q1_Y),
            'q1_Z_avg_p': avg_p_q1_Z,
            'q1_Z_count': len(p_values_for_q1_Z)
        }

    else:
        # If the pauli_term_2q was not found in final_fitted_params (e.g., no data or all fits failed)
        print(f"  Info: No fitted parameters found for fundamental Pauli term '{pauli_term_2q}'. Averages will be NaN.")
        averaged_p_values_by_q1_basis[pauli_term_2q] = {
            'q1_X_avg_p': np.nan, 'q1_X_count': 0,
            'q1_Y_avg_p': np.nan, 'q1_Y_count': 0,
            'q1_Z_avg_p': np.nan, 'q1_Z_count': 0
        }

# --- Displaying the Averaged Results ---
# Convert the dictionary of averages into a list of dictionaries for DataFrame creation
results_for_df_avg_p = []

for pauli_term, avg_data in averaged_p_values_by_q1_basis.items():
    results_for_df_avg_p.append({
        'Pauli Term (q2,q0)': pauli_term,
        'Avg p (q1=X)': avg_data['q1_X_avg_p'],
        'Count (q1=X)': avg_data['q1_X_count'],
        'Avg p (q1=Y)': avg_data['q1_Y_avg_p'],
        'Count (q1=Y)': avg_data['q1_Y_count'],
        'Avg p (q1=Z)': avg_data['q1_Z_avg_p'],
        'Count (q1=Z)': avg_data['q1_Z_count']
    })

# Create a Pandas DataFrame for formatted output
summary_avg_p_df = pd.DataFrame(results_for_df_avg_p)
# Set display options for Pandas (makes for nicer printing)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.4f}'.format) # Format floats to 4 decimal places
print("\n--- Summary of Averaged Decay Parameters (p) Grouped by q1 Basis ---")
print(summary_avg_p_df)
# Save this summary to a new CSV file
csv_filename_avg_p = "averaged_decay_parameters_by_q1_basis.csv"

try:
    summary_avg_p_df.to_csv(csv_filename_avg_p, index=False, float_format='%.6f')
    print(f"\nAveraged 'p' values table saved to: {csv_filename_avg_p}")

except Exception as e_csv_avg:
    print(f"\nError saving averaged 'p' values to CSV {csv_filename_avg_p}: {e_csv_avg}")

print("\nPhase 6: Averaging Fitted 'p' Values Complete.")

# --- Phase 7: Pauli Error Probability Calculation and Visualisation ---

print("\n--- Phase 7: Pauli Error Probability Calculation and Visualisation ---")

if (('averaged_p_values_by_q1_basis' not in globals()) or (not averaged_p_values_by_q1_basis)):
    print("Error: 'averaged_p_values_by_q1_basis' is not defined or is empty. Please ensure Phase 6 ran successfully.")
    # Initialise to prevent errors, though results will be meaningless
    averaged_p_values_by_q1_basis = {
        p: {'q1_X_avg_p': np.nan, 'q1_Y_avg_p': np.nan, 'q1_Z_avg_p': np.nan}
        for p in FUNDAMENTAL_2Q_PAULIS
    }

if 'FUNDAMENTAL_2Q_PAULIS' not in globals():
    print("Error: FUNDAMENTAL_2Q_PAULIS not defined.")
    FUNDAMENTAL_2Q_PAULIS = [ # Define a fallback for robustness
        'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ',
        'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ'
    ]

# --- Helper Functions and Definitions ---

# Define the order of Pauli strings for calculations (16 terms)
CALC_PAULI_ORDER = ['II'] + FUNDAMENTAL_2Q_PAULIS
# Expected order: ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']
# Define the desired order for plotting on the x-axis
PLOT_PAULI_ORDER = [
    'II',               # Identity
    'IX', 'IY', 'IZ',   # Single on q0
    'XI', 'YI', 'ZI',   # Single on q2
    'XX', 'XY', 'XZ',   # Two-qubit, q2=X
    'YX', 'YY', 'YZ',   # Two-qubit, q2=Y
    'ZX', 'ZY', 'ZZ'    # Two-qubit, q2=Z  
]
# Create the list for plotting without 'II'
PLOT_PAULI_ORDER_NO_II = [p for p in PLOT_PAULI_ORDER if p != 'II']

def pauli_char_to_symplectic_pair(char):
    """
    Converts a single Pauli character ('I', 'X', 'Y', 'Z') to its (x, z) binary symplectic pair.
    """
    if (char == 'I'):
        return (0, 0)
    
    if (char == 'X'):
        return (1, 0)
    
    if (char == 'Y'):
        return (1, 1) # Y = XZ
    
    if (char == 'Z'):
        return (0, 1)
    
    raise ValueError(f"Invalid Pauli character: {char}")

def pauli_str_to_symplectic_vec(pauli_str_2q):
    """
    Converts a 2-qubit Pauli string (e.g., 'IX', 'ZY') to a 4-element symplectic binary vector (x2, z2, x0, z0).
    """
    if (len(pauli_str_2q) != 2):
        raise ValueError("Pauli string must have 2 characters for 2 qubits.")
    
    p2_char, p0_char = pauli_str_2q[0], pauli_str_2q[1]
    x2, z2 = pauli_char_to_symplectic_pair(p2_char)
    x0, z0 = pauli_char_to_symplectic_pair(p0_char)
    return np.array([x2, z2, x0, z0])

def symplectic_inner_products(s_vec1, s_vec2):
    """
    Calculates the symlectic inner product for 2-qubit Pauli binary vectors.
    """
    # s_vec = (x2, z2, x0, z0)
    # product = x2*z'2 + z2*x'2 + x0*z'0 + z0*x'0 (mod 2)
    # Note: indices for s_vec: 0=x2, 1=z2, 2=x0, 3=z0
    prod = ((s_vec1[0] * s_vec2[1]) + (s_vec1[1] * s_vec2[0]) + (s_vec1[2] * s_vec2[3]) + (s_vec1[3] * s_vec2[2])) % 2
    return prod

# Generate the 16x16 S-matrix for the transformation
N_PAULIS = 16
S_matrix = np.zeros((N_PAULIS, N_PAULIS))
symplectic_vectors = [pauli_str_to_symplectic_vec(p_str) for p_str in CALC_PAULI_ORDER]

for i in range(N_PAULIS):
    for j in range(N_PAULIS):
        s_i = symplectic_vectors[i]
        s_j = symplectic_vectors[j]
        S_matrix[i, j] = (-1)**(symplectic_inner_products(s_i, s_j))

# --- Main Calculation Loop ---

pauli_error_probabilites = {} # To store results: {'q1_X': prob_vector, ...}
q1_bases_map = {
    'X': 'q1_X_avg_p',
    'Y': 'q1_Y_avg_p',
    'Z': 'q1_Z_avg_p'
} 

for q1_basis_char, avg_p_key in q1_bases_map.items():
    # 1. Assemble the 16-element lambda_vector (p_II=1, then 15 avg_p values)
    lambda_vector = np.zeros(N_PAULIS)
    lambda_vector[0] = 1.0 # p_II is 1.0

    for idx, pauli_str in enumerate(CALC_PAULI_ORDER):
        
        if (idx == 0):
            continue   # Skip 'II' as it's already set

        # averaged_p_values_by_q1_basis stores data per fundamental Pauli
        # FUNDAMENTAL_2Q_PAULIS is CALC_PAULI_ORDER[1:]
        # So pauli_str is one of FUNDAMENTAL_2Q_PAULIS
        if pauli_str in averaged_p_values_by_q1_basis:
            avg_p_val = averaged_p_values_by_q1_basis[pauli_str].get(avg_p_key, np.nan)
            lambda_vector[idx] = avg_p_val if not np.isnan(avg_p_val) else 0.0 # Replace NaN with 0 for transform
            # Or handle NaNs more explicitly if preferred
        
        else: # Should not happen if FUNDAMENTAL_2Q_PAULIS is consistent
            lambda_vector[idx] = 0.0 # Default if Pauli term somehow missing
            print(f"Warning: Pauli term {pauli_str} not found in averaged_p_values_by_q1_basis for q1={q1_basis_char}.")

    # Handle cases where the entire lambda_vector might be based on NaNs (except lambda_vector[0])
    if np.all(np.isnan(lambda_vector[1:])): # if all avg_p values were NaN
        print(f"Info: All averaged 'p' values for q1={q1_basis_char} are NaN. Resulting error probabilities might not be meaningful.")
        # Keep NaNs in lambda_vector to propagate them, or fill with 0s if transform requires finite numbers
        lambda_vector = np.nan_to_num(lambda_vector, nan=0.0) # Replace NaNs with 0 for the transform

    # 2. Transform to Pauli error probabilities: p_errors = (1/16) * S_matrix @ lambda_vector
    prob_vector_calc_order = (1/16) * (S_matrix @ lambda_vector)
    pauli_error_probabilites[f'q1_{q1_basis_char}'] = prob_vector_calc_order
    # Sanity check: sum of probabilities should be close to 1 if input lambdas were physical
    # This might not hold perfectly if input p values were unphysical or NaNs were handled by 0.
    print(f"  Sum of calculated Pauli error probabilities for q1={q1_basis_char}: {np.sum(prob_vector_calc_order):.4f}")
    # The p_II term from prob_vector_calc_order is the probability of the Identity error.
    idx_II_in_calc_order = CALC_PAULI_ORDER.index('II')
    print(f"    Calculated p_II (Identity error prob) for q1={q1_basis_char}: {prob_vector_calc_order[idx_II_in_calc_order]:.4f}")

# --- Plotting Results ---

PLOT_DIR_PAULI_PROBS = "plots_pauli_error_probs"

if not os.path.exists(PLOT_DIR_PAULI_PROBS):
    os.makedirs(PLOT_DIR_PAULI_PROBS)

print(f"Pauli error probability plots will be saved to: {PLOT_DIR_PAULI_PROBS}")

# Create a mapping from CALC_PAULI_ORDER to indicees for easy reordering
calc_order_to_idx = {pauli: i for i, pauli in enumerate(CALC_PAULI_ORDER)}

for q1_basis_char in ['X', 'Y', 'Z']:
    prob_key = f'q1_{q1_basis_char}'

    if prob_key not in pauli_error_probabilites:
        print(f"  No probability data to plot for q1={q1_basis_char}.")
        continue
    
    # Get the probability vector in the calculation order
    prob_vector_calc = pauli_error_probabilites[prob_key]
    # Prepare data for plotting (excluding 'II')
    # Reorder the probabilities according to PLOT_PAULI_ORDER
    prob_vector_plot_order_no_II = np.array([prob_vector_calc[calc_order_to_idx[p_str]] for p_str in PLOT_PAULI_ORDER_NO_II])
    plt.figure(figsize=(15, 7))
    bars = plt.bar(PLOT_PAULI_ORDER_NO_II, prob_vector_plot_order_no_II, color='skyblue')
    
    # Color negative bars differently
    for bar in bars:

        if (bar.get_height() < 0):
            bar.set_color('salmon') # Or 'red', 'lightcoral', etc.

    plt.xlabel("2-Qubit Pauli Term (on q2, q0)", fontsize=12)
    plt.ylabel("Probability ($p_P$)", fontsize=12)
    plt.title(f"Calculated Pauli Error Probabilities (excluding $p_{{II}}$) for (q2,q0) Subsystem\n(q1 measured in {q1_basis_char}-basis)", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.8) # Add a zero line for reference

    # Add text labels on bars
    for bar_idx, bar in enumerate(bars):
        yval = bar.get_height()
        # Adjust text position slightly for better readability
        vertical_offset = 0.002 if (yval >= 0) else -0.005
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + vertical_offset, f'{yval:.4f}', ha='center', va='bottom' if (yval >= 0) else 'top', fontsize=8)
    
    # Determine y-limits dynamically, ensuring 0 is visible
    min_val_plot = np.min(prob_vector_plot_order_no_II)
    max_val_plot = np.max(prob_vector_plot_order_no_II)
    y_lower_bound = min(0, min_val_plot) - 0.01 # Add some padding
    y_upper_bound = max(0, max_val_plot) + 0.01 # Add some padding

    # Ensure a minimum range if all values are very close to zero
    if ((y_upper_bound - y_lower_bound) < 0.05):
        y_upper_bound = max(y_upper_bound, y_lower_bound + 0.05)

        if (y_lower_bound == 0): # if all values were positive and small
            y_lower_bound = min(0, y_upper_bound - 0.05)

    plt.ylim(y_lower_bound, y_upper_bound) # Adjust y-limits
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plot_filename = os.path.join(PLOT_DIR_PAULI_PROBS, f"pauli_error_probs_q1_{q1_basis_char}.png")

    try:
        plt.savefig(plot_filename, dpi=150)
        print(f"  Saved plot: {plot_filename}")
    
    except Exception as e_save:
        print(f"  Error saving plot {plot_filename}: {e_save}")
    
    plt.close()

# --- Display probabilities in a table ---

prob_summary_list = []
# Use PLOT_PAULI_ORDER for table rows for consistency with plots
for pauli_op_plot_order in PLOT_PAULI_ORDER:
    row_data = {'Pauli Term (q2,q0)': pauli_op_plot_order}
    # Get index in calculation order
    idx_calc_order = calc_order_to_idx[pauli_op_plot_order]

    for q1_basis_char in ['X', 'Y', 'Z']:
        prob_key = f'q1_{q1_basis_char}'
        prob_val = pauli_error_probabilites.get(prob_key, np.full(N_PAULIS, np.nan))[idx_calc_order]
        row_data[f'Prob (q1={q1_basis_char})'] = prob_val

    prob_summary_list.append(row_data)

prob_summary_df = pd.DataFrame(prob_summary_list)
prob_summary_df = prob_summary_df.set_index('Pauli Term (q2,q0)') # Set Pauli term as index
print("\n--- Summary of Calculated Pauli Error Probabilites ---")
# Ensure float format is set for pandas display
pd.set_option('display.float_format', '{:.4f}'.format)
print(prob_summary_df)
csv_filename_probs = "pauli_error_probabilites_summary.csv"

try:
    prob_summary_df.to_csv(csv_filename_probs, float_format='%.6f')
    print(f"\nPauli error probabilities table saved to: {csv_filename_probs}")

except Exception as e_csv_probs:
    print(f"\nError saving Pauli error probabilities to CSV {csv_filename_probs}: {e_csv_probs}")

print("\nPhase 7: Pauli Error Probability Calculation and Visualisation Complete.")

# --- Phase 8: Qubit q1 Error Analysis ---

print("\n--- Phase 8: Qubit q1 Error Analysis ---")

# --- Helper Function: Calculate 1-qubit marginal probabilities for q1 ---

def calculate_q1_marginal_probabilities(counts_3q, total_shots):
    """
    Calculates 1-qubit marginal probabilities for q1 from 3-qubit counts.
    Qiskit convention: q2q1q0.
    Output: NumPy array [P(q1=0), P(q1=1)]
    """
    # P(q1=0) = sum of counts where q1 is 0
    p_q1_0 = (counts_3q.get('000', 0) + counts_3q.get('001', 0) + counts_3q.get('100', 0) + counts_3q.get('101', 0)) / total_shots
    # P(q1=1) = sum of counts where q1 is 1
    p_q1_1 = (counts_3q.get('010', 0) + counts_3q.get('011', 0) + counts_3q.get('110', 0) + counts_3q.get('111', 0)) / total_shots
    return np.array([p_q1_0, p_q1_1])

# --- 1. Extract Raw Lambda Series for q1 ---
# q1_raw_lambda_series[basis_3q_str] = {'q1_basis_char': char, 'lambda_series': np.array([...])}

q1_raw_lambda_series = {}
print("  Extracting raw lambda series for q1...")

for idx, current_3q_basis_str in enumerate(BASES_3Q):
    file_index = idx + 1 # Assuming count_1.pickle corresponds to BASES_3Q[0]
    filepath = os.path.join(RESULTS_DIR, f"count_{file_index}.pickle")
    # Use existing load_results_file and get_qiskit_counts_from_bitarray from Phase 1
    data_for_6_Ms = load_results_file(filepath)

    if data_for_6_Ms is None:
        print(f"    Skipping basis {current_3q_basis_str} for q1 analysis (file missing).")
        continue

    if (len(data_for_6_Ms) != len(M_values)):
        print(f"    Warning: Data for {current_3q_basis_str} has {len(data_for_6_Ms)} M-values, expected {len(M_values)}. Skipping for q1 analysis.")
        continue

    current_q1_basis_char = current_3q_basis_str[1] # Middle character is q1's basis
    lambda_series_for_current_experiment = []
    
    for m_data_raw in data_for_6_Ms:
        counts_3q = get_qiskit_counts_from_bitarray(m_data_raw)

        if not counts_3q:
            lambda_series_for_current_experiment.append(np.nan)
            continue

        probs_q1_vec = calculate_q1_marginal_probabilities(counts_3q, SHOTS) # [P(q1=0), P(q1=1)]
        # Apply 1Q WHT: lambda_I = P(0)+P(1), lambda_B_q1 = P(0)-P(1)
        # We are interested in lambda_B_q1 for fitting decay
        lambda_B_q1_val = probs_q1_vec[0] - probs_q1_vec[1]
        lambda_series_for_current_experiment.append(lambda_B_q1_val)
        # Sanity check for lambda_I_q1 (should be ~1.0)
        lambda_I_q1_val = probs_q1_vec[0] + probs_q1_vec[1]

        if not np.isclose(lambda_I_q1_val, 1.0, atol=1e-3):
          print(f"    Sanity Check Warning: lambda_I for q1 in {current_3q_basis_str} is {lambda_I_q1_val:.4f}")

    q1_raw_lambda_series[current_3q_basis_str] = {
        'q1_basis_char': current_q1_basis_char,
        'lambda_series': np.array(lambda_series_for_current_experiment)
    }

# --- 2. Group Lambda Series and Fit for 'p' ---
# q1_fitted_p_values[q1_basis_char] = [p1_from_exp1, p2_from_exp2, ...]

q1_fitted_p_values = {'X': [], 'Y': [], 'Z': []}
print("  Fitting q1 lambda series...")

for basis_3q_str, data in q1_raw_lambda_series.items():
    q1_basis = data['q1_basis_char']
    series_to_fit = data['lambda_series']

    if ((np.isnan(series_to_fit).any()) or (len(series_to_fit) != len(M_values))):
        print(f"   Skipping fit for q1 (basis {q1_basis} from {basis_3q_str}) due to invalid data.")
        continue
    
    try:
        initial_A = series_to_fit[0] if (series_to_fit[0] != 0) else (0.5 if (np.mean(series_to_fit) > 0) else -0.5)
        p0 = [initial_A, 0.9] # Initial guess for A, p
        bounds = ([-1.5, 0.0], [1.5, 1.0]) # Bounds for A, p
        popt, _ = curve_fit(decay_model, M_values, series_to_fit, p0=p0, bounds=bounds, maxfev=5000)
        p_fit = popt[1] # We only need the decay parameter 'p'
        q1_fitted_p_values[q1_basis].append(p_fit)
    
    except RuntimeError:
        print(f"    RuntimeError fitting q1 data for basis {q1_basis} from {basis_3q_str}.")
    
    except Exception as e:
        print(f"   Error fitting q1 data for basis {q1_basis} from {basis_3q_str}.")

# --- 3. Average Fitted 'p' Values ---

avg_p_q1 = {}
print("  Averaging fitted 'p' values for q1...")

for basis, p_list in q1_fitted_p_values.items():
    
    if p_list:
        avg_p_q1[basis] = np.mean([p for p in p_list if not np.isnan(p)])
        print(f"    Avg 'p' for q1 in {basis}-basis (from {len(p_list)} fits): {avg_p_q1[basis]:.4f}")
    
    else:
        avg_p_q1[basis] = np.nan # Should not happen if data loading and fitting were okay for some
        print(f"    No successful fits for q1 in {basis}-basis to average.")

# --- 4. Convert Averaged 'p' to Pauli Error Probabilities for q1 ---
# Eigenvalues for the 1Q channel on q1: [lambda_I, lambda_X, lambda_Y, lambda_Z]
# We use 1.0 for lambda_I, and the averaged 'p' values for lambda_X, lambda_Y, lambda_Z.

q1_channel_lambdas = np.array([
    1.0,
    avg_p_q1.get('X', np.nan), # Default to NaN if a basis had no successful fits
    avg_p_q1.get('Y', np.nan),
    avg_p_q1.get('Z', np.nan)
])
print(f"  q1 Channel Lambdas (I,X,Y,Z): {q1_channel_lambdas}")
# 1-Qubit WHT matrix for transforming eigenvalues to Pauli error probabilities
# Rows/Cols order: I, X, Y, Z for probabilities and lambdas
WHT_1Q_ERROR_PROB_MATRIX = np.array([
    [1, 1, 1, 1],    # p_I = (1/4)(lambda_I + lambda_X + lambda_Y + lambda_Z)
    [1, 1, -1, -1],  # p_X = (1/4)(lambda_I + lambda_X - lambda_Y - lambda_Z)
    [1, -1, 1, -1],  # p_Y = (1/4)(lambda_I - lambda_X + lambda_Y - lambda_Z)
    [1, -1, -1, 1]   # p_Z = (1/4)(lambda_I - lambda_X - lambda_Y + lambda_Z)
])

# Handle potential NaNs in q1_channel_lambdas before matrix multiplication
if np.isnan(q1_channel_lambdas).any():
    print("    Warning: NaN found in averaged 'p' values for q1. Resulting error probabilities might be NaN.")
    q1_channel_lambdas_finite = np.nan_to_num(q1_channel_lambdas, nan=0.0) # Or propagate NaNs if preferred

else:
    q1_channel_lambdas_finite = q1_channel_lambdas

q1_error_probs_vector = (1/4) * (WHT_1Q_ERROR_PROB_MATRIX @ q1_channel_lambdas_finite)
# Order: [p_I_q1, p_X_q1, p_Y_q1, p_Z_q1]
q1_error_labels = ['I', 'X', 'Y', 'Z']
q1_error_probabilities_map = {label: prob for label, prob in zip(q1_error_labels, q1_error_probs_vector)}
print("  Calculated Pauli Error Probabilities for q1:")

for label, prob in q1_error_probabilities_map.items():
    print(f"    p_{label}(q1): {prob:.4f}")

print(f"    Sum of q1 error probabilities: {np.sum(q1_error_probs_vector):.4f}")

# --- 5. Plot Bar Graph for q1 (excluding Identity error) ---

PLOT_DIR_Q1_ERRORS = "plots_q1_error_analysis"

if not os.path.exists(PLOT_DIR_Q1_ERRORS):
    os.makedirs(PLOT_DIR_Q1_ERRORS)

error_types_to_plot = ['X', 'Y', 'Z']
probabilities_to_plot = [q1_error_probabilities_map[label] for label in error_types_to_plot]
plt.figure(figsize=(8, 6))
bars = plt.bar(error_types_to_plot, probabilities_to_plot, color=['#FF6347', '#4682B4', '#32CD32']) # Tomato, SteelBlue, LimeGreen

# Color negative bars differently (though less likely for 1Q with p_I ~1 and others <1)
for bar_idx, bar in enumerate(bars):

    if (bar.get_height() < 0):
        bar.set_color('darkgrey')
    
    # Add text labels
    yval = bar.get_height()
    vertical_offset = 0.001 if (yval >= 0) else -0.003
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + vertical_offset, f'{yval:.4f}', ha='center', va='bottom' if (yval >= 0) else 'top', fontsize=9)

plt.xlabel("Pauli Error Type on q1", fontsize=12)
plt.ylabel("Probability ($p_K$)", fontsize=12)
plt.title("Calculated Pauli Error Probabilities for Qubit q1", fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axhline(0, color='black', linewidth=0.8)
# Dynamic y-limits
min_val_plot = np.min(probabilities_to_plot) if probabilities_to_plot else 0
max_val_plot = np.max(probabilities_to_plot) if probabilities_to_plot else 0.1
y_lower_bound = min(0, min_val_plot) - 0.005
y_upper_bound = max(0, max_val_plot) + 0.005

if ((y_upper_bound - y_lower_bound) < 0.02): # Ensure some visible range
    y_upper_bound = max(y_upper_bound, y_lower_bound + 0.02)

plt.ylim(y_lower_bound, y_upper_bound)
plt.tight_layout()
q1_plot_filename = os.path.join(PLOT_DIR_Q1_ERRORS, "q1_pauli_error_probabilities.png")

try:
    plt.savefig(q1_plot_filename, dpi=150)
    print(f"  Saved q1 eror plot to: {q1_plot_filename}")

except Exception as e_save:
    print(f"  Error saving q1 error plot {q1_plot_filename}: {e_save}")

plt.close()

# --- Save q1 error probabilities to CSV ---

q1_summary_df = pd.DataFrame(list(q1_error_probabilities_map.items()), columns=['Pauli Error on q1', 'Probability'])
q1_csv_filename = "q1_pauli_error_probabilities_summary.csv"

try:
    q1_summary_df.to_csv(q1_csv_filename, index=False, float_format='%.6f')
    print("  q1 error probabilities table saved to: {q1_csv_filename}")

except Exception as e_csv_q1:
    print(f"  Error saving q1 error probabilities to CSV {q1_csv_filename}: {e_csv_q1}")

print("\nPhase 8: Qubit q1 Error Analysis Complete.")