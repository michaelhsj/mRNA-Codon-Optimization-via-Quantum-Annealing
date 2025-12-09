import json
import numpy as np
from matplotlib import pyplot as plt
import sys

import hamiltonian
import python_codon_tables as pct
from pennylane import numpy as np
import pennylane as qml
from pennylane import qaoa
import networkx as nx

# ============================================================================
# LOAD DATA
# ============================================================================

def load_results(json_filename):
    """Load results from JSON file."""
    with open(json_filename, 'r') as f:
        results = json.load(f)
    return results


# ============================================================================
# SETUP (needed for energy calculations and state reconstruction)
# ============================================================================

def setup_problem(polypeptide_sequence="MALW"):
    """Setup the problem to reconstruct states."""
    e_coli_table = pct.get_codons_table('e_coli_316407')
    amino_acids = e_coli_table
    
    Q, offset = hamiltonian.Q(polypeptide_sequence, 1.5)
    h, J, offset = hamiltonian.Q_to_Ising(Q, offset)
    wires = range(len(h))
    
    def get_position_structure(polypeptide_sequence, amino_acids):
        position_to_qubits = []
        qubit_idx = 0
        for aa in polypeptide_sequence:
            num_codons = len(amino_acids[aa])
            position_to_qubits.append(list(range(qubit_idx, qubit_idx + num_codons)))
            qubit_idx += num_codons
        return position_to_qubits
    
    position_to_qubits = get_position_structure(polypeptide_sequence, amino_acids)
    
    cost_h = sum([h[i] * qml.Z(wires=i) for i in range(len(h))]) + \
             sum([J[i, j] * qml.Z(wires=i) @ qml.Z(wires=j) for i in range(len(h)) for j in range(i)])
    
    def build_xy_mixer_same_position(position_to_qubits):
        """XY mixer connecting only qubits at the same position."""
        graph = nx.Graph()
        
        position_to_codons = {}  # Track which codons belong to each position
        codon_count = 0
        
        for pos_idx, aa in enumerate(polypeptide_sequence):
            codons = []
            for i in range(len(amino_acids[aa])):
                codon = codon_count
                codons.append(codon)
                codon_count += 1
            
            position_to_codons[pos_idx] = codons
            graph.add_nodes_from(codons)
            
            # Fully connect codons for the same amino acid position
            for i in range(len(codons)):
                for j in range(i + 1, len(codons)):
                    graph.add_edge(codons[i], codons[j])
        
        return qaoa.mixers.xy_mixer(graph)
    
    mixer_h = build_xy_mixer_same_position(position_to_qubits)
    
    return h, J, offset, wires, position_to_qubits, cost_h, mixer_h, amino_acids


def SCS(qubits, m_idx, k):
    """Split & Cycle Shift unitary for Dicke state preparation."""
    m = qubits[m_idx - 1]
    m_prev = qubits[m_idx - 2]
    
    qml.CNOT(wires=[m_prev, m])
    qml.CRY(2 * np.arccos(np.sqrt(1 / m_idx)), wires=[m, m_prev])
    qml.CNOT(wires=[m_prev, m])
    
    for l in range(2, k + 1):
        control_qubit = qubits[m_idx - l]
        middle_qubit = qubits[m_idx - l + 1]
        
        qml.CNOT(wires=[control_qubit, m])
        qml.ctrl(qml.RY, control=[m, middle_qubit])(
            2 * np.arccos(np.sqrt(l / m_idx)), 
            wires=control_qubit
        )
        qml.CNOT(wires=[control_qubit, m])


def prepare_dicke_state(qubits, k):
    """Prepares a Dicke state with m qubits and k excitations."""
    m = len(qubits)
    
    if k > m:
        raise ValueError(f"Cannot have {k} excitations with only {m} qubits")
    
    for i in range(m - k, m):
        qml.PauliX(wires=qubits[i])
    
    for i in reversed(range(k + 1, m + 1)):
        SCS(qubits, i, k)
    
    for i in reversed(range(2, k + 1)):
        SCS(qubits, i, i - 1)


def init_dicke_per_position(position_to_qubits):
    """Initialize each position to a W-state (Dicke with k=1)."""
    for position_qubits in position_to_qubits:
        n = len(position_qubits)
        if n == 1:
            qml.PauliX(wires=position_qubits[0])
        else:
            prepare_dicke_state(position_qubits, k=1)


# ============================================================================
# PLOT 1: Cost Convergence (Hadamard vs Dicke strategies)
# ============================================================================

def plot_cost_convergence(results, offset, output_filename='plot1_cost_convergence.png'):
    """Plot convergence for Hadamard + X, Dicke + X, Dicke + XY."""
    
    strategies_to_plot = [
        'Hadamard + X Mixer',
        'Dicke + X Mixer', 
        'Dicke + XY Mixer'
    ]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    plt.figure(figsize=(10, 6))
    
    for strategy, color in zip(strategies_to_plot, colors):
        if strategy in results:
            trials = results[strategy]
            # Find best trial
            best_trial = min(trials, key=lambda x: x['final_cost_with_offset'])
            cost_history_with_offset = [c + offset for c in best_trial['cost_history']]
            plt.plot(cost_history_with_offset, label=strategy, linewidth=2.5, color=color)
    
    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('Cost', fontsize=13)
    plt.title('Cost Convergence (Best Trial per Strategy)', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_filename}")
    plt.close()


# ============================================================================
# PLOT 2: Mean Final Cost Comparison (excluding Hadamard + X)
# ============================================================================

def plot_mean_cost_comparison(results, offset, output_filename='plot2_mean_cost_comparison.png'):
    """Bar chart of mean final costs (excluding Hadamard + X Mixer)."""
    
    strategies = [
        'Dicke + X Mixer',
        'Dicke + XY Mixer',
        'Random Feasible + XY Mixer',
        'Greedy + XY Mixer'
    ]
    colors = ['#4ECDC4', '#45B7D1', '#95E1D3', '#FFA07A']
    
    mean_costs = []
    std_costs = []
    
    for strategy in strategies:
        if strategy in results:
            trials = results[strategy]
            costs_with_offset = [t['final_cost_with_offset'] for t in trials]
            mean_costs.append(np.mean(costs_with_offset))
            std_costs.append(np.std(costs_with_offset))
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(strategies))
    bars = plt.bar(x, mean_costs, yerr=std_costs, capsize=5, alpha=0.8, color=colors)
    plt.xticks(x, strategies, rotation=30, ha='right', fontsize=11)
    plt.ylabel('Mean Cost', fontsize=13)
    plt.title('Mean Final Cost Comparison', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Highlight best strategy
    best_idx = np.argmin(mean_costs)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    # Add value labels
    for bar, mean, std in zip(bars, mean_costs, std_costs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.3f}\n±{std:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_filename}")
    plt.close()


# ============================================================================
# PLOT 3: Distribution of Final Costs (excluding Hadamard + X)
# ============================================================================

def plot_cost_distribution(results, offset, output_filename='plot3_cost_distribution.png'):
    """Bar plot of mean cost with error bars (excluding Hadamard + X Mixer)."""
    
    strategies = [
        'Dicke + X Mixer',
        'Dicke + XY Mixer',
        'Random Feasible + XY Mixer',
        'Greedy + XY Mixer'
    ]
    colors = ['#4ECDC4', '#45B7D1', '#95E1D3', '#FFA07A']
    
    means = []
    std_devs = []
    std_errors = []
    
    for strategy in strategies:
        if strategy in results:
            trials = results[strategy]
            costs = [t['final_cost_with_offset'] for t in trials]
            means.append(np.mean(costs))
            std_devs.append(np.std(costs))
            std_errors.append(np.std(costs) / np.sqrt(len(costs)))  # Standard error
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(strategies))
    
    # Use standard deviation for error bars (can change to std_errors for SEM)
    bars = plt.bar(x, means, yerr=std_devs, capsize=5, alpha=0.8, 
                   color=colors, error_kw={'linewidth': 2, 'elinewidth': 2})
    
    plt.xticks(x, strategies, rotation=30, ha='right', fontsize=11)
    plt.ylabel('Cost', fontsize=13)
    plt.title('Distribution of Final Costs (All Trials)', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Highlight best strategy
    best_idx = np.argmin(means)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, std_devs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.3f}\n±{std:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_filename}")
    plt.close()


# ============================================================================
# PLOT 4: Final State Probability Distribution (Dicke + XY best result)
# ============================================================================

def plot_final_state_distribution(results, polypeptide_sequence="MALW", 
                                  output_filename='plot4_final_state_distribution.png'):
    """
    Plot probability distribution of final state from best Dicke + XY trial,
    ordered by energy, using stored probabilities and energies.
    """
    
    # Setup problem (needed for position_to_qubits to check feasibility)
    h, J, offset, wires, position_to_qubits, cost_h, mixer_h, amino_acids = setup_problem(polypeptide_sequence)
    
    # Get best trial from Dicke + XY Mixer
    strategy_name = 'Dicke + XY Mixer'
    if strategy_name not in results:
        print(f"Warning: {strategy_name} not found in results")
        return
    
    trials = results[strategy_name]
    best_trial = min(trials, key=lambda x: x['final_cost_with_offset'])
    
    print(f"\nUsing stored data from best {strategy_name} trial...")
    print(f"Best trial final cost: {best_trial['final_cost_with_offset']:.4f}")
    
    # Extract stored data from JSON
    probabilities = np.array(best_trial['final_probabilities'])
    bitstrings = best_trial['bitstrings']
    state_energies = best_trial['state_energies']
    
    num_qubits = len(wires)
    
    # Check which states are feasible (satisfy constraints)
    feasible_states = []
    for bitstring in bitstrings:
        valid = True
        for position_qubits in position_to_qubits:
            ones_count = sum(int(bitstring[q]) for q in position_qubits)
            if ones_count != 1:
                valid = False
                break
        feasible_states.append(valid)
    
    # Sort states by energy
    sorted_indices = np.argsort(state_energies)
    sorted_energies = [state_energies[i] for i in sorted_indices]
    sorted_probabilities = [probabilities[i] for i in sorted_indices]
    sorted_bitstrings = [bitstrings[i] for i in sorted_indices]
    sorted_feasible = [feasible_states[i] for i in sorted_indices]
    
    # Plot: Top 20 lowest energy states with bitstrings and probabilities
    fig, ax = plt.subplots(figsize=(12, 10))
    
    top_k = min(20, len(sorted_energies))
    top_energies = sorted_energies[:top_k]
    top_probabilities = sorted_probabilities[:top_k]
    top_bitstrings = sorted_bitstrings[:top_k]
    top_feasible = sorted_feasible[:top_k]
    
    colors_top = ['#2ECC71' if feasible else '#E74C3C' for feasible in top_feasible]
    
    # Create horizontal bar chart
    y_positions = range(top_k)
    bars = ax.barh(y_positions, top_probabilities, color=colors_top, alpha=0.7)
    
    # Set y-axis labels with rank, bitstring, and energy
    labels = [f"{i+1}: {bs} (E={e:.3f})" 
              for i, (bs, e) in enumerate(zip(top_bitstrings, top_energies))]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9, family='monospace')
    
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_ylabel('State (Rank: Bitstring, Energy)', fontsize=12)
    ax.set_title('20 Lowest Energy States', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ECC71', label='Feasible'),
                      Patch(facecolor='#E74C3C', label='Infeasible')]
    ax.legend(handles=legend_elements, fontsize=11, loc='lower right')
    
    # Add probability values on bars
    for i, (bar, prob) in enumerate(zip(bars, top_probabilities)):
        if prob > 0.001:  # Only show label if probability is significant
            ax.text(prob, bar.get_y() + bar.get_height()/2., 
                   f'{prob:.4f}',
                   ha='left', va='center', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_filename}")
    plt.close()
    
    # Print summary
    print(f"\nProbability Distribution Analysis:")
    print(f"  Total states: {len(probabilities)}")
    print(f"  States with P > 0.001: {sum(probabilities > 0.001)}")
    print(f"  States with P > 0.01: {sum(probabilities > 0.01)}")
    print(f"  Most probable state: {best_trial['most_probable_bitstring']} "
          f"(P={best_trial['most_probable_probability']:.4f})")
    print(f"  Lowest energy: {min(sorted_energies):.4f}")
    print(f"  Lowest feasible energy: {min([e for e, f in zip(sorted_energies, sorted_feasible) if f]):.4f}")
    print(f"  Total probability on feasible states: {sum([p for p, f in zip(probabilities, feasible_states) if f]):.4f}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(json_filename):
    """Generate all plots from JSON results file."""
    
    print("="*70)
    print("GENERATING PLOTS FROM STRATEGY COMPARISON RESULTS")
    print("="*70)
    
    # Load results
    print(f"\nLoading results from: {json_filename}")
    results = json.load(open(json_filename, 'r'))
    
    # Get offset (need to recalculate)
    __, offset = hamiltonian.Q("MALW", 1.5)
    print(f"Offset: {offset:.4f}")
    
    # Generate plots
    print("\nGenerating plots...")
    print("-"*70)
    
    plot_cost_convergence(results, offset)
    plot_mean_cost_comparison(results, offset)
    plot_cost_distribution(results, offset)
    plot_final_state_distribution(results)
    
    print("\n" + "="*70)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  - plot1_cost_convergence.png")
    print("  - plot2_mean_cost_comparison.png")
    print("  - plot3_cost_distribution.png")
    print("  - plot4_final_state_distribution.png")
    print("="*70)


if __name__ == "__main__":
    # Check if filename provided as argument
    if len(sys.argv) > 1:
        json_filename = sys.argv[1]
    else:
        # Default filename - user should update this
        json_filename = "strategy_comparison_20251209_180942.json"  # Update with your filename
        print(f"No filename provided, using default: {json_filename}")
        print("Usage: python plot_results.py <json_filename>")
        print()
    
    main(json_filename)