from pennylane import qaoa
from pennylane import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import pennylane as qml
import json
from datetime import datetime

import hamiltonian
import python_codon_tables as pct

# ============================================================================
# SETUP
# ============================================================================

# Load codon table
e_coli_table = pct.get_codons_table('e_coli_316407')
amino_acids = e_coli_table

# Define the optimization problem
polypeptide_sequence = "MALW"
print(f"Protein sequence: {polypeptide_sequence}")

Q, offset = hamiltonian.Q(polypeptide_sequence, 1.5)
h, J, offset = hamiltonian.Q_to_Ising(Q, offset)
wires = range(len(h))

print(f"Number of qubits: {len(wires)}")
print(f"Offset: {offset:.4f}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_position_structure(polypeptide_sequence, amino_acids):
    """Map amino acid positions to their corresponding qubit indices."""
    position_to_qubits = []
    qubit_idx = 0
    
    for aa in polypeptide_sequence:
        num_codons = len(amino_acids[aa])
        position_to_qubits.append(list(range(qubit_idx, qubit_idx + num_codons)))
        qubit_idx += num_codons
    
    return position_to_qubits


def check_constraint_satisfaction(probs, position_to_qubits, threshold=0.0):
    """
    Check what fraction of probability mass is on constraint-satisfying states.
    A state satisfies constraints if exactly one qubit is 1 per position.
    """
    total_valid_prob = 0.0
    num_qubits = sum(len(pos) for pos in position_to_qubits)
    
    for state_idx, prob in enumerate(probs):
        if prob < threshold:
            continue
        
        bitstring = format(state_idx, f'0{num_qubits}b')
        
        # Check if this state satisfies constraints
        valid = True
        for position_qubits in position_to_qubits:
            ones_count = sum(int(bitstring[q]) for q in position_qubits)
            if ones_count != 1:
                valid = False
                break
        
        if valid:
            total_valid_prob += prob
    
    return total_valid_prob


position_to_qubits = get_position_structure(polypeptide_sequence, amino_acids)
print(f"Position structure: {position_to_qubits}")


# ============================================================================
# INITIALIZATION FUNCTIONS
# ============================================================================

def init_hadamard(wires):
    """Hadamard initialization on all qubits."""
    for w in wires:
        qml.Hadamard(wires=w)


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
    
    # Prepare input state: set last k qubits to |1⟩
    for i in range(m - k, m):
        qml.PauliX(wires=qubits[i])
    
    # Apply the SCS unitaries
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


def init_random_feasible(position_to_qubits):
    """Initialize to a random valid solution (one qubit per position)."""
    for position_qubits in position_to_qubits:
        selected = np.random.choice(position_qubits)
        qml.PauliX(wires=selected)


def init_greedy_solution(position_to_qubits, h):
    """Initialize to a greedy solution based on linear terms."""
    for position_qubits in position_to_qubits:
        local_h = [h[q] for q in position_qubits]
        best_qubit = position_qubits[np.argmin(local_h)]
        qml.PauliX(wires=best_qubit)


# ============================================================================
# MIXER FUNCTIONS
# ============================================================================

def build_x_mixer(wires):
    """X mixer: sum of X operators on all wires."""
    return sum([qml.X(wires=i) for i in wires])


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


# ============================================================================
# BUILD COST HAMILTONIAN
# ============================================================================

cost_h = sum([h[i] * qml.Z(wires=i) for i in range(len(h))]) + \
         sum([J[i, j] * qml.Z(wires=i) @ qml.Z(wires=j) for i in range(len(h)) for j in range(i)])


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_strategy_experiment(
    strategy_name,
    init_function,
    mixer_hamiltonian,
    depth=2,
    num_iterations=100,
    num_trials=5
):
    """Run multiple trials for a given strategy."""
    
    print(f"\n{'='*70}")
    print(f"Strategy: {strategy_name}")
    print(f"{'='*70}")
    
    trial_results = []
    
    for trial in range(num_trials):
        print(f"  Trial {trial+1}/{num_trials}...")
        
        def qaoa_layer(gamma, alpha):
            qaoa.cost_layer(gamma, cost_h)
            qaoa.mixer_layer(alpha, mixer_hamiltonian)
        
        def circuit(params):
            init_function()
            qml.layer(qaoa_layer, depth, params[0], params[1])
        
        dev = qml.device("default.qubit", wires=wires)
        
        @qml.qnode(dev)
        def cost_function(params):
            circuit(params)
            return qml.expval(cost_h)
        
        @qml.qnode(dev)
        def get_probs(params):
            circuit(params)
            return qml.probs(wires=wires)
        
        # Initialize parameters
        params = np.random.uniform(0, 2*np.pi, size=(2, depth))
        optimizer = qml.AdamOptimizer(stepsize=0.02)
        
        cost_history = []
        
        # Optimization loop
        for iteration in range(num_iterations):
            params = optimizer.step(cost_function, params)
            current_cost = cost_function(params)
            cost_history.append(float(current_cost))
            
            if iteration % 25 == 0:
                print(f"    Iteration {iteration}: Cost = {current_cost:.4f}")
        
        # Final evaluation
        final_cost = cost_history[-1]
        probs = get_probs(params)
        constraint_satisfaction = check_constraint_satisfaction(probs, position_to_qubits)
        
        # Get most probable state
        most_probable_idx = np.argmax(probs)
        bitstring = format(most_probable_idx, f'0{len(wires)}b')
        z = np.array([1 if b == '0' else -1 for b in bitstring])
        final_energy = hamiltonian.Ising_energy(z, h, J, offset)
        
        # Store all probabilities and corresponding bitstrings
        all_probs = [float(p) for p in probs]  # Convert to regular Python floats for JSON serialization
        all_bitstrings = [format(i, f'0{len(wires)}b') for i in range(len(probs))]
        
        # Calculate energies for all states
        all_energies = []
        for bs in all_bitstrings:
            z_state = np.array([1 if b == '0' else -1 for b in bs])
            energy = hamiltonian.Ising_energy(z_state, h, J, offset)
            all_energies.append(float(energy))
        
        trial_results.append({
            'cost_history': cost_history,
            'final_cost': final_cost,
            'final_cost_with_offset': final_cost + offset,
            'final_energy': float(final_energy),
            'constraint_satisfaction': float(constraint_satisfaction),
            'final_probabilities': all_probs,
            'bitstrings': all_bitstrings,
            'state_energies': all_energies,
            'most_probable_bitstring': bitstring,
            'most_probable_probability': float(probs[most_probable_idx])
        })
        
        print(f"    Final Cost (+ offset): {final_cost + offset:.4f}")
        print(f"    Constraint Satisfaction: {constraint_satisfaction:.2%}")
        print(f"    Most Probable State: {bitstring} (P={probs[most_probable_idx]:.4f})")
    
    return trial_results


def run_all_experiments():
    """Run experiments for all strategies."""
    
    # Define strategies to test
    strategies = {
        'Hadamard + X Mixer': {
            'init': lambda: init_hadamard(wires),
            'mixer': build_x_mixer(wires)
        },
        'Dicke + X Mixer': {
            'init': lambda: init_dicke_per_position(position_to_qubits),
            'mixer': build_x_mixer(wires)
        },
        'Dicke + XY Mixer': {
            'init': lambda: init_dicke_per_position(position_to_qubits),
            'mixer': build_xy_mixer_same_position(position_to_qubits)
        },
        'Random Feasible + XY Mixer': {
            'init': lambda: init_random_feasible(position_to_qubits),
            'mixer': build_xy_mixer_same_position(position_to_qubits)
        },
        'Greedy + XY Mixer': {
            'init': lambda: init_greedy_solution(position_to_qubits, h),
            'mixer': build_xy_mixer_same_position(position_to_qubits)
        }
    }
    
    results = {}
    
    for strategy_name, strategy_config in strategies.items():
        trial_results = run_strategy_experiment(
            strategy_name,
            strategy_config['init'],
            strategy_config['mixer'],
            depth=2,
            num_iterations=100,
            num_trials=5
        )
        
        results[strategy_name] = trial_results
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(results):
    """Create comprehensive visualizations."""
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    strategies = list(results.keys())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3', '#FFA07A']
    
    # ========================================================================
    # Plot 1: Convergence of cost (+offset) for best trial of each strategy
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :])
    
    for strategy, color in zip(strategies, colors):
        trials = results[strategy]
        # Find best trial (lowest final cost)
        best_trial = min(trials, key=lambda x: x['final_cost_with_offset'])
        cost_history_with_offset = [c + offset for c in best_trial['cost_history']]
        ax1.plot(cost_history_with_offset, label=strategy, linewidth=2.5, color=color)
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Cost + Offset', fontsize=12)
    ax1.set_title('Cost Convergence (Best Trial per Strategy)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 2: Mean cost (+offset) bar chart with error bars
    # ========================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    
    mean_costs = []
    std_costs = []
    
    for strategy in strategies:
        trials = results[strategy]
        costs_with_offset = [t['final_cost_with_offset'] for t in trials]
        mean_costs.append(np.mean(costs_with_offset))
        std_costs.append(np.std(costs_with_offset))
    
    x = np.arange(len(strategies))
    bars = ax2.bar(x, mean_costs, yerr=std_costs, capsize=5, alpha=0.8, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Mean Cost + Offset', fontsize=12)
    ax2.set_title('Mean Final Cost Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Highlight best strategy
    best_idx = np.argmin(mean_costs)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, mean_costs, std_costs)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.3f}\n±{std:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    # ========================================================================
    # Plot 3: Probability mass on feasible solutions
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    
    mean_feasibility = []
    std_feasibility = []
    
    for strategy in strategies:
        trials = results[strategy]
        feasibility = [t['constraint_satisfaction'] * 100 for t in trials]
        mean_feasibility.append(np.mean(feasibility))
        std_feasibility.append(np.std(feasibility))
    
    bars2 = ax3.bar(x, mean_feasibility, yerr=std_feasibility, capsize=5, alpha=0.8, color=colors)
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies, rotation=45, ha='right', fontsize=10)
    ax3.set_ylabel('Probability Mass on Feasible Solutions (%)', fontsize=12)
    ax3.set_title('Constraint Satisfaction', fontsize=14, fontweight='bold')
    ax3.axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=2, label='100% feasible')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 105])
    
    # Highlight best strategy
    best_feasible_idx = np.argmax(mean_feasibility)
    bars2[best_feasible_idx].set_edgecolor('gold')
    bars2[best_feasible_idx].set_linewidth(3)
    
    # Add value labels
    for bar, mean, std in zip(bars2, mean_feasibility, std_feasibility):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}%\n±{std:.1f}%',
                ha='center', va='bottom', fontsize=9)
    
    # ========================================================================
    # Plot 4: All trials cost distribution (violin plot)
    # ========================================================================
    ax4 = fig.add_subplot(gs[2, 0])
    
    all_costs = []
    positions = []
    for i, strategy in enumerate(strategies):
        trials = results[strategy]
        costs = [t['final_cost_with_offset'] for t in trials]
        all_costs.append(costs)
        positions.append(i)
    
    parts = ax4.violinplot(all_costs, positions=positions, widths=0.7, 
                           showmeans=True, showmedians=True)
    
    # Color the violin plots
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax4.set_xticks(positions)
    ax4.set_xticklabels(strategies, rotation=45, ha='right', fontsize=10)
    ax4.set_ylabel('Cost + Offset', fontsize=12)
    ax4.set_title('Distribution of Final Costs (All Trials)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # Plot 5: Cost vs Constraint Satisfaction scatter
    # ========================================================================
    ax5 = fig.add_subplot(gs[2, 1])
    
    for strategy, color in zip(strategies, colors):
        trials = results[strategy]
        costs = [t['final_cost_with_offset'] for t in trials]
        feasibility = [t['constraint_satisfaction'] * 100 for t in trials]
        ax5.scatter(feasibility, costs, label=strategy, s=100, alpha=0.7, color=color, edgecolors='black')
    
    ax5.set_xlabel('Probability Mass on Feasible Solutions (%)', fontsize=12)
    ax5.set_ylabel('Cost + Offset', fontsize=12)
    ax5.set_title('Cost vs Constraint Satisfaction', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=9, loc='best')
    ax5.grid(True, alpha=0.3)
    
    plt.savefig('strategy_comparison_comprehensive.png', dpi=300, bbox_inches='tight')
    print("\n" + "="*70)
    print("Visualization saved as 'strategy_comparison_comprehensive.png'")
    print("="*70)
    plt.show()
    
    return fig


def print_summary(results):
    """Print detailed summary of results."""
    
    print("\n" + "="*70)
    print("DETAILED RESULTS SUMMARY")
    print("="*70)
    
    strategies = list(results.keys())
    
    # Calculate metrics
    summary_data = []
    for strategy in strategies:
        trials = results[strategy]
        costs = [t['final_cost_with_offset'] for t in trials]
        feasibility = [t['constraint_satisfaction'] for t in trials]
        
        summary_data.append({
            'strategy': strategy,
            'mean_cost': np.mean(costs),
            'std_cost': np.std(costs),
            'best_cost': np.min(costs),
            'mean_feasibility': np.mean(feasibility) * 100,
            'std_feasibility': np.std(feasibility) * 100
        })
    
    # Sort by mean cost
    summary_data.sort(key=lambda x: x['mean_cost'])
    
    print(f"\n{'Rank':<6} {'Strategy':<35} {'Mean Cost':<15} {'Best Cost':<15} {'Feasibility'}")
    print("-" * 95)
    
    for i, data in enumerate(summary_data, 1):
        print(f"{i:<6} {data['strategy']:<35} "
              f"{data['mean_cost']:.4f} ± {data['std_cost']:.4f}    "
              f"{data['best_cost']:.4f}        "
              f"{data['mean_feasibility']:.1f}% ± {data['std_feasibility']:.1f}%")
    
    print("\n" + "="*70)
    print(f"WINNER: {summary_data[0]['strategy']}")
    print("="*70)


# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("QAOA STRATEGY COMPARISON EXPERIMENT")
    print(f"Protein: {polypeptide_sequence}")
    print(f"Number of qubits: {len(wires)}")
    print("="*70)
    
    # Run all experiments
    results = run_all_experiments()
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'strategy_comparison_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: strategy_comparison_{timestamp}.json")
    
    # Print summary
    print_summary(results)
    
    # Create visualizations
    plot_results(results)