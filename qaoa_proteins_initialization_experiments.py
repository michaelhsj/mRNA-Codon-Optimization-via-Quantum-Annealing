from pennylane import qaoa
from pennylane import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import pennylane as qml

import hamiltonian
import python_codon_tables as pct

# Load codon table
e_coli_table = pct.get_codons_table('e_coli_316407')
amino_acids = e_coli_table

polypeptide_sequence = "MALW"  # Example protein sequence (insulin fragment)
print(f"Protein sequence: {polypeptide_sequence}")

# Build problem
Q, offset = hamiltonian.Q(polypeptide_sequence, 1.5)
h, J, offset = hamiltonian.Q_to_Ising(Q, offset)
wires = range(len(h))

# Build cost Hamiltonian
cost_h = sum([h[i] * qml.Z(wires=i) for i in range(len(h))]) + \
         sum([J[i, j] * qml.Z(wires=i) @ qml.Z(wires=j) for i in range(len(h)) for j in range(i)])

# Track codon positions
def get_position_structure(polypeptide_sequence, amino_acids):
    """Map amino acid positions to their corresponding qubit indices."""
    position_to_qubits = []
    qubit_idx = 0
    
    for aa in polypeptide_sequence:
        num_codons = len(amino_acids[aa])
        position_to_qubits.append(list(range(qubit_idx, qubit_idx + num_codons)))
        qubit_idx += num_codons
    
    return position_to_qubits

position_to_qubits = get_position_structure(polypeptide_sequence, amino_acids)
print(f"Position structure: {position_to_qubits}")
print(f"Total qubits: {len(wires)}")


# ============================================================================
# INITIALIZATION STRATEGIES
# ============================================================================

def init_uniform_superposition(wires):
    """Standard initialization: Hadamard on all qubits."""
    for w in wires:
        qml.Hadamard(wires=w)


def init_w_state_per_position(position_to_qubits):
    """
    Initialize each position to a W-state (equal superposition of one-hot states).
    This is the BEST initialization for constrained one-hot problems!
    """
    for position_qubits in position_to_qubits:
        n = len(position_qubits)
        if n == 1:
            qml.PauliX(wires=position_qubits[0])
        elif n == 2:
            # W-state for 2 qubits: (|10⟩ + |01⟩)/√2
            qml.Hadamard(wires=position_qubits[0])
            qml.CNOT(wires=[position_qubits[0], position_qubits[1]])
            qml.PauliX(wires=position_qubits[0])
        else:
            # General W-state using Dicke state preparation
            # Approximate W-state for n qubits
            _prepare_dicke_state(position_qubits, 1)


def SCS(qubits, m_idx, k):
    """
    Implements the Split & Cycle Shift unitary for Dicke state preparation.
    
    Args:
        qubits: List of qubit indices to operate on
        m_idx: Current position index in the recursive preparation (1-indexed)
        k: Number of excitations
    """
    # Map to actual qubit indices (m_idx is 1-indexed in the algorithm)
    m = qubits[m_idx - 1]
    m_prev = qubits[m_idx - 2]
    
    # Two-qubit gate
    qml.CNOT(wires=[m_prev, m])
    qml.CRY(2 * np.arccos(np.sqrt(1 / m_idx)), wires=[m, m_prev])
    qml.CNOT(wires=[m_prev, m])
    
    # k-1 three-qubit gates
    for l in range(2, k + 1):
        control_qubit = qubits[m_idx - l]
        middle_qubit = qubits[m_idx - l + 1]
        
        qml.CNOT(wires=[control_qubit, m])
        qml.ctrl(qml.RY, control=[m, middle_qubit])(
            2 * np.arccos(np.sqrt(l / m_idx)), 
            wires=control_qubit
        )
        qml.CNOT(wires=[control_qubit, m])


def _prepare_dicke_state(qubits, k):
    """
    Prepares a Dicke state |D^k_m⟩ with m qubits and k excitations.
    Uses the efficient inductive method from your example.
    
    Args:
        qubits: List of qubit indices to prepare as Dicke state
        k: Number of excitations (typically 1 for W-state)
    """
    m = len(qubits)
    
    if k > m:
        raise ValueError(f"Cannot have {k} excitations with only {m} qubits")
    
    # Prepare input state: set last k qubits to |1⟩
    for i in range(m - k, m):
        qml.PauliX(wires=qubits[i])
    
    # Apply the SCS unitaries in reverse order
    for i in reversed(range(k + 1, m + 1)):
        SCS(qubits, i, k)
    
    for i in reversed(range(2, k + 1)):
        SCS(qubits, i, i - 1)


def init_random_feasible(position_to_qubits):
    """Initialize to a random valid solution (one qubit per position)."""
    for position_qubits in position_to_qubits:
        # Randomly select one qubit to excite
        selected = np.random.choice(position_qubits)
        qml.PauliX(wires=selected)


def init_greedy_solution(position_to_qubits, h):
    """Initialize to a greedy solution based on linear terms."""
    qubit_idx = 0
    for position_qubits in position_to_qubits:
        # Select the qubit with the most negative h value (best single-qubit energy)
        local_h = [h[q] for q in position_qubits]
        best_qubit = position_qubits[np.argmin(local_h)]
        qml.PauliX(wires=best_qubit)


# ============================================================================
# BUILD MIXER GRAPH
# ============================================================================

def build_constrained_mixer_graph(position_to_qubits):
    """Build graph that connects only codons at the same position."""
    graph = nx.Graph()
    
    for position_qubits in position_to_qubits:
        graph.add_nodes_from(position_qubits)
        # Fully connect qubits at the same position
        for i in range(len(position_qubits)):
            for j in range(i + 1, len(position_qubits)):
                graph.add_edge(position_qubits[i], position_qubits[j])
    
    return graph

mixer_graph = build_constrained_mixer_graph(position_to_qubits)
mixer_h = qaoa.mixers.xy_mixer(mixer_graph)


# ============================================================================
# EXPERIMENT: Compare Initialization Strategies
# ============================================================================

def run_initialization_experiment(init_strategies, depth=2, num_iterations=100, num_trials=5):
    """Compare different initialization strategies."""
    
    results = {}
    
    for strategy_name, init_function in init_strategies.items():
        print(f"\n{'='*60}")
        print(f"Testing: {strategy_name}")
        print(f"{'='*60}")
        
        trial_results = []
        
        for trial in range(num_trials):
            def qaoa_layer(gamma, alpha):
                qaoa.cost_layer(gamma, cost_h)
                qaoa.mixer_layer(alpha, mixer_h)
            
            def circuit(params):
                # Apply initialization strategy
                init_function()
                # Apply QAOA layers
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
            
            @qml.qnode(dev)
            def get_state(params):
                circuit(params)
                return qml.state()
            
            # Optimize
            params = np.random.uniform(0, np.pi, size=(2, depth))
            optimizer = qml.AdamOptimizer(stepsize=0.02)
            
            cost_history = []
            energy_history = []
            
            for iteration in range(num_iterations):
                #TODO: clean up
                #print(get_state(params))

                params = optimizer.step(cost_function, params)
                current_cost = cost_function(params)
                cost_history.append(current_cost)
                
                # Calculate energy of most probable state
                probs = get_probs(params)
                most_probable_idx = np.argmax(probs)
                bitstring = format(most_probable_idx, f'0{len(wires)}b')
                z = np.array([1 if b == '0' else -1 for b in bitstring])
                energy = hamiltonian.Ising_energy(z, h, J, offset)
                energy_history.append(energy)
                
                if iteration % 25 == 0:
                    print(f"  Trial {trial+1}, Iter {iteration}: "
                          f"Cost={current_cost:.4f}, Energy={energy:.4f}")
            
            # Check constraint satisfaction
            probs = get_probs(params)
            constraint_satisfaction = check_constraint_satisfaction(
                probs, position_to_qubits, threshold=0.
            )
            
            trial_results.append({
                'cost_history': cost_history,
                'energy_history': energy_history,
                'final_cost': float(cost_history[-1]),
                'final_energy': float(energy_history[-1]),
                'constraint_satisfaction': constraint_satisfaction
            })
        
        # Aggregate results
        results[strategy_name] = {
            'trials': trial_results,
            'avg_final_cost': float(np.mean([t['final_cost'] for t in trial_results])),
            'std_final_cost': float(np.std([t['final_cost'] for t in trial_results])),
            'avg_final_energy': float(np.mean([t['final_energy'] for t in trial_results])),
            'std_final_energy': float(np.std([t['final_energy'] for t in trial_results])),
            'avg_constraint_satisfaction': float(np.mean([t['constraint_satisfaction'] for t in trial_results]))
        }
        
        print(f"\n  Summary:")
        print(f"    Avg Final Cost: {results[strategy_name]['avg_final_cost']:.4f} ± "
              f"{results[strategy_name]['std_final_cost']:.4f}")
        print(f"    Avg Final Energy: {results[strategy_name]['avg_final_energy']:.4f} ± "
              f"{results[strategy_name]['std_final_energy']:.4f}")
        print(f"    Constraint Satisfaction: {results[strategy_name]['avg_constraint_satisfaction']:.2%}")
    
    return results


def check_constraint_satisfaction(probs, position_to_qubits, threshold=0.01):
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


def test_dicke_state_preparation():
    """Test that the Dicke state preparation creates the correct state."""
    print("\n" + "="*80)
    print("TESTING DICKE STATE PREPARATION")
    print("="*80)
    
    for n in [2, 3, 4, 5]:
        print(f"\nTesting W-state (Dicke with k=1) for n={n} qubits:")
        
        test_wires = range(n)
        dev = qml.device("default.qubit", wires=test_wires)
        
        @qml.qnode(dev)
        def prepare_test_state():
            _prepare_dicke_state(list(test_wires), k=1)
            return qml.state()
        
        state = prepare_test_state()
        
        # Check which computational basis states have non-zero amplitude
        valid_states = []
        for i, amplitude in enumerate(state):
            if abs(amplitude) > 1e-6:
                bitstring = format(i, f'0{n}b')
                ones_count = bitstring.count('1')
                prob = abs(amplitude)**2
                valid_states.append((bitstring, amplitude, prob))
                
        print(f"  Non-zero states (should all have exactly 1 bit set):")
        for bitstring, amplitude, prob in valid_states:
            print(f"    |{bitstring}⟩: amplitude={amplitude:.4f}, prob={prob:.4f}")
        
        # Verify it's a proper W-state
        expected_amplitude = 1.0 / np.sqrt(n)
        all_correct = all(
            bitstring.count('1') == 1 and abs(abs(amplitude) - expected_amplitude) < 1e-6
            for bitstring, amplitude, prob in valid_states
        )
        
        if all_correct and len(valid_states) == n:
            print(f"  ✓ Valid W-state: {n} states with amplitude ≈ 1/√{n}")
        else:
            print(f"  ✗ Invalid state!")

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

if __name__ == "__main__":
    test_dicke_state_preparation()
    
    # Define initialization strategies to test
    init_strategies = {
        #TODO: change back
        'W-State per Position (BEST)': lambda: init_w_state_per_position(position_to_qubits),
        #'Uniform Superposition (Hadamard)': lambda: init_uniform_superposition(wires),
        'Random Feasible Solution': lambda: init_random_feasible(position_to_qubits),
        'Greedy Solution': lambda: init_greedy_solution(position_to_qubits, h)
    }
    
    results = run_initialization_experiment(
        init_strategies, 
        depth=2, 
        num_iterations=100, 
        num_trials=5
    )
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Average final energy comparison
    ax1 = axes[0, 0]
    strategies = list(results.keys())
    avg_energies = [results[s]['avg_final_energy'] for s in strategies]
    std_energies = [results[s]['std_final_energy'] for s in strategies]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    bars = ax1.bar(range(len(strategies)), avg_energies, yerr=std_energies, 
                   capsize=5, alpha=0.8, color=colors)
    ax1.set_xticks(range(len(strategies)))
    ax1.set_xticklabels(strategies, rotation=45, ha='right')
    ax1.set_ylabel('Final Energy')
    ax1.set_title('Final Energy by Initialization Strategy')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Highlight best
    best_idx = np.argmin(avg_energies)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    # Plot 2: Constraint satisfaction
    ax2 = axes[0, 1]
    constraint_sats = [results[s]['avg_constraint_satisfaction'] * 100 for s in strategies]
    bars2 = ax2.bar(range(len(strategies)), constraint_sats, alpha=0.8, color=colors)
    ax2.set_xticks(range(len(strategies)))
    ax2.set_xticklabels(strategies, rotation=45, ha='right')
    ax2.set_ylabel('Constraint Satisfaction (%)')
    ax2.set_title('Probability Mass on Valid Solutions')
    ax2.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100% valid')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Highlight best
    best_constraint_idx = np.argmax(constraint_sats)
    bars2[best_constraint_idx].set_edgecolor('gold')
    bars2[best_constraint_idx].set_linewidth(3)
    
    # Plot 3: Energy convergence curves
    ax3 = axes[1, 0]
    for strategy, color in zip(strategies, colors):
        # Plot best trial for each strategy
        best_trial = min(results[strategy]['trials'], key=lambda x: x['final_energy'])
        ax3.plot(best_trial['energy_history'], label=strategy, linewidth=2, color=color)
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Energy (Best State)')
    ax3.set_title('Energy Convergence (Best Trial per Strategy)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cost convergence curves
    ax4 = axes[1, 1]
    for strategy, color in zip(strategies, colors):
        best_trial = min(results[strategy]['trials'], key=lambda x: x['final_cost'])
        ax4.plot(best_trial['cost_history'], label=strategy, linewidth=2, color=color)
    
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Cost (QAOA Objective)')
    ax4.set_title('Cost Convergence (Best Trial per Strategy)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('initialization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: Best Initialization Strategy")
    print("="*80)
    best_strategy = min(strategies, key=lambda s: results[s]['avg_final_energy'])
    print(f"Winner: {best_strategy}")
    print(f"  Average Final Energy: {results[best_strategy]['avg_final_energy']:.4f}")
    print(f"  Constraint Satisfaction: {results[best_strategy]['avg_constraint_satisfaction']:.2%}")
    print("="*80)