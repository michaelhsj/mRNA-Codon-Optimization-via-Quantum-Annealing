from pennylane import qaoa
from pennylane import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import pennylane as qml
import json
from datetime import datetime

import hamiltonian
import python_codon_tables as pct

from qaoa_proteins_initialization_experiments import init_w_state_per_position

# Load codon table
e_coli_table = pct.get_codons_table('e_coli_316407')
amino_acids = e_coli_table

# Select a real but small protein sequence
# Example: Insulin signal peptide (first 10 amino acids) - adjust as needed
# You can replace this with any real protein sequence
polypeptide_sequence = "MALW"  # 10 amino acids
print(f"Protein sequence: {polypeptide_sequence}")
print(f"Length: {len(polypeptide_sequence)} amino acids")



# Build Q matrix
Q, offset = hamiltonian.Q(polypeptide_sequence, 1.5)
h, J, offset = hamiltonian.Q_to_Ising(Q, offset)

wires = range(len(h))
print(f"Number of qubits: {len(wires)}")

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

# Build cost Hamiltonian
cost_h = sum([h[i] * qml.Z(wires=i) for i in range(len(h))]) + \
         sum([J[i, j] * qml.Z(wires=i) @ qml.Z(wires=j) for i in range(len(h)) for j in range(i)])


def build_mixer_graph(polypeptide_sequence, amino_acids, mixer_type='same_position'):
    """Build graph for mixer Hamiltonian based on type."""
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
    
    if mixer_type == 'all_codons':
        # Connect ALL codons to each other
        all_nodes = list(graph.nodes())
        for i in range(len(all_nodes)):
            for j in range(i + 1, len(all_nodes)):
                graph.add_edge(all_nodes[i], all_nodes[j])
    
    return graph, position_to_codons


def get_mixer_hamiltonian(mixer_type, polypeptide_sequence, amino_acids):
    """Generate mixer Hamiltonian based on type."""
    if mixer_type == 'xy_same_position':
        graph, _ = build_mixer_graph(polypeptide_sequence, amino_acids, 'same_position')
        return qaoa.mixers.xy_mixer(graph)
    
    elif mixer_type == 'xy_all_codons':
        graph, _ = build_mixer_graph(polypeptide_sequence, amino_acids, 'all_codons')
        return qaoa.mixers.xy_mixer(graph)
    
    elif mixer_type == 'x_same_position':
        graph, position_to_codons = build_mixer_graph(polypeptide_sequence, amino_acids, 'same_position')
        # X mixer: sum of X operators on nodes
        return qaoa.mixers.x_mixer(graph)
    
    elif mixer_type == 'bit_flip_same_position':
        # Bit flip mixer: X on all wires
        return sum([qml.X(wires=i) for i in wires])
    
    else:
        raise ValueError(f"Unknown mixer type: {mixer_type}")


# ============================================================================
# EXPERIMENT 1: High-Iteration Optimization with Energy Tracking
# ============================================================================

def experiment_1_energy_tracking(depth=2, num_iterations=200):
    """Track Ising energy over many optimization iterations."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Energy Tracking Over Iterations")
    print("=" * 80)
    
    mixer_h = get_mixer_hamiltonian('xy_same_position', polypeptide_sequence, amino_acids)
    
    def qaoa_layer(gamma, alpha):
        qaoa.cost_layer(gamma, cost_h)
        qaoa.mixer_layer(alpha, mixer_h)
    
    def circuit(params, **kwargs):
        for w in wires:
            qml.Hadamard(wires=w)
        qml.layer(qaoa_layer, depth, params[0], params[1])
    
    dev = qml.device("default.qubit", wires=wires)
    
    @qml.qnode(dev)
    def cost_function(params):
        circuit(params)
        return qml.expval(cost_h)
    
    @qml.qnode(dev)
    def get_state_vector(params):
        circuit(params)
        return qml.probs(wires=wires)
    
    # Track metrics
    iteration_history = []
    energy_history = []
    cost_history = []
    
    # Initialize parameters
    params = np.random.uniform(0, 2*np.pi, size=(2, depth))
    optimizer = qml.AdamOptimizer(stepsize=0.01)
    
    print(f"Running {num_iterations} iterations with depth={depth}")
    print(f"Initial cost: {cost_function(params):.6f}")
    
    for iteration in range(num_iterations):
        # Optimize
        params = optimizer.step(cost_function, params)
        
        # Calculate metrics
        current_cost = cost_function(params)
        probs = get_state_vector(params)
        
        # Get most probable state and calculate its energy
        most_probable_idx = np.argmax(probs)
        bitstring = format(most_probable_idx, f'0{len(wires)}b')
        z = np.array([1 if b == '0' else -1 for b in bitstring])
        ising_energy = hamiltonian.Ising_energy(z, h, J, offset)
        
        iteration_history.append(iteration)
        cost_history.append(current_cost)
        energy_history.append(ising_energy)
        
        if iteration % 20 == 0:
            print(f"Iteration {iteration:3d}: Cost = {current_cost:.6f}, "
                  f"Best State Energy = {ising_energy:.6f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1 = axes[0]
    ax1.plot(iteration_history, cost_history, 'b-', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Expected Cost (QAOA Objective)')
    ax1.set_title('QAOA Cost Function Over Iterations')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(iteration_history, energy_history, 'r-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Ising Energy (Best State)')
    ax2.set_title('Ising Energy of Most Probable State Over Iterations')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment1_energy_tracking.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    results = {
        'iterations': iteration_history,
        'costs': cost_history,
        'energies': energy_history,
        'final_params': params.tolist(),
        'final_cost': float(cost_history[-1]),
        'final_energy': float(energy_history[-1])
    }
    
    return results


# ============================================================================
# EXPERIMENT 2: Mixer Hamiltonian Comparison
# ============================================================================

def experiment_2_mixer_comparison(depth=2, num_iterations=100):
    """Compare different mixer Hamiltonians."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Mixer Hamiltonian Comparison")
    print("=" * 80)
    
    mixer_types = [
        'xy_same_position',
        'xy_all_codons',
        'x_same_position',
        'bit_flip_same_position'
    ]
    
    results = {}
    
    for mixer_type in mixer_types:
        print(f"\n--- Testing Mixer: {mixer_type} ---")
        
        try:
            mixer_h = get_mixer_hamiltonian(mixer_type, polypeptide_sequence, amino_acids)
            
            def qaoa_layer(gamma, alpha):
                qaoa.cost_layer(gamma, cost_h)
                qaoa.mixer_layer(alpha, mixer_h)
            
            def circuit(params, **kwargs):
                init_w_state_per_position(position_to_qubits)
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
            
            # Optimize
            params = np.random.uniform(0, 2*np.pi, size=(2, depth))
            optimizer = qml.AdamOptimizer(stepsize=0.01)
            
            cost_history = []
            for iteration in range(num_iterations):
                params = optimizer.step(cost_function, params)
                cost_history.append(cost_function(params))
                
                if iteration % 25 == 0:
                    print(f"  Iteration {iteration}: Cost = {cost_history[-1]:.6f}")
            
            # Get final energy
            probs = get_probs(params)
            most_probable_idx = np.argmax(probs)
            bitstring = format(most_probable_idx, f'0{len(wires)}b')
            z = np.array([1 if b == '0' else -1 for b in bitstring])
            final_energy = hamiltonian.Ising_energy(z, h, J, offset)
            
            results[mixer_type] = {
                'cost_history': cost_history,
                'final_cost': float(cost_history[-1]),
                'final_energy': float(final_energy),
                'final_params': params.tolist()
            }
            
            print(f"  Final Cost: {cost_history[-1]:.6f}")
            print(f"  Final Energy: {final_energy:.6f}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results[mixer_type] = {'error': str(e)}
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Cost convergence
    ax1 = axes[0]
    for mixer_type, data in results.items():
        if 'cost_history' in data:
            ax1.plot(data['cost_history'], label=mixer_type, linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost')
    ax1.set_title('Cost Convergence by Mixer Type')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final performance
    ax2 = axes[1]
    mixer_names = [m for m in results.keys() if 'final_cost' in results[m]]
    final_costs = [results[m]['final_cost'] for m in mixer_names]
    final_energies = [results[m]['final_energy'] for m in mixer_names]
    
    x = np.arange(len(mixer_names))
    width = 0.35
    
    ax2.bar(x - width/2, final_costs, width, label='Final Cost', alpha=0.8)
    ax2.bar(x + width/2, final_energies, width, label='Final Energy', alpha=0.8)
    ax2.set_xlabel('Mixer Type')
    ax2.set_ylabel('Value')
    ax2.set_title('Final Performance by Mixer Type')
    ax2.set_xticks(x)
    ax2.set_xticklabels(mixer_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('experiment2_mixer_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


# ============================================================================
# EXPERIMENT 3: Circuit Depth Analysis
# ============================================================================

def experiment_3_depth_analysis(depths=[1, 2, 3, 4, 5], num_iterations=100):
    """Analyze performance across different circuit depths."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Circuit Depth Analysis")
    print("=" * 80)
    
    mixer_h = get_mixer_hamiltonian('xy_same_position', polypeptide_sequence, amino_acids)
    
    results = {}
    
    for depth in depths:
        print(f"\n--- Testing Depth: {depth} ---")
        
        def qaoa_layer(gamma, alpha):
            qaoa.cost_layer(gamma, cost_h)
            qaoa.mixer_layer(alpha, mixer_h)
        
        def circuit(params, **kwargs):
            for w in wires:
                qml.Hadamard(wires=w)
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
        
        # Run multiple trials for each depth
        num_trials = 5
        trial_results = []
        
        for trial in range(num_trials):
            params = np.random.uniform(0, 2*np.pi, size=(2, depth))
            optimizer = qml.AdamOptimizer(stepsize=0.01)
            
            cost_history = []
            for iteration in range(num_iterations):
                params = optimizer.step(cost_function, params)
                cost_history.append(cost_function(params))
            
            # Get final energy
            probs = get_probs(params)
            most_probable_idx = np.argmax(probs)
            bitstring = format(most_probable_idx, f'0{len(wires)}b')
            z = np.array([1 if b == '0' else -1 for b in bitstring])
            final_energy = hamiltonian.Ising_energy(z, h, J, offset)
            
            trial_results.append({
                'cost_history': cost_history,
                'final_cost': float(cost_history[-1]),
                'final_energy': float(final_energy)
            })
            
            print(f"  Trial {trial+1}/{num_trials}: Final Cost = {cost_history[-1]:.6f}, "
                  f"Energy = {final_energy:.6f}")
        
        # Aggregate results
        best_trial = min(trial_results, key=lambda x: x['final_cost'])
        avg_final_cost = np.mean([t['final_cost'] for t in trial_results])
        std_final_cost = np.std([t['final_cost'] for t in trial_results])
        avg_final_energy = np.mean([t['final_energy'] for t in trial_results])
        std_final_energy = np.std([t['final_energy'] for t in trial_results])
        
        results[depth] = {
            'trials': trial_results,
            'best_cost': best_trial['final_cost'],
            'best_energy': best_trial['final_energy'],
            'avg_cost': float(avg_final_cost),
            'std_cost': float(std_final_cost),
            'avg_energy': float(avg_final_energy),
            'std_energy': float(std_final_energy)
        }
        
        print(f"  Average Final Cost: {avg_final_cost:.6f} ± {std_final_cost:.6f}")
        print(f"  Average Final Energy: {avg_final_energy:.6f} ± {std_final_energy:.6f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Best cost vs depth
    ax1 = axes[0, 0]
    depths_list = list(results.keys())
    best_costs = [results[d]['best_cost'] for d in depths_list]
    ax1.plot(depths_list, best_costs, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Circuit Depth')
    ax1.set_ylabel('Best Cost')
    ax1.set_title('Best Cost vs Circuit Depth')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average cost with error bars
    ax2 = axes[0, 1]
    avg_costs = [results[d]['avg_cost'] for d in depths_list]
    std_costs = [results[d]['std_cost'] for d in depths_list]
    ax2.errorbar(depths_list, avg_costs, yerr=std_costs, fmt='o-', 
                 linewidth=2, markersize=8, capsize=5)
    ax2.set_xlabel('Circuit Depth')
    ax2.set_ylabel('Average Cost')
    ax2.set_title('Average Cost vs Circuit Depth (with std dev)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Best energy vs depth
    ax3 = axes[1, 0]
    best_energies = [results[d]['best_energy'] for d in depths_list]
    ax3.plot(depths_list, best_energies, 'o-', linewidth=2, markersize=8, color='red')
    ax3.set_xlabel('Circuit Depth')
    ax3.set_ylabel('Best Energy')
    ax3.set_title('Best Ising Energy vs Circuit Depth')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Convergence curves for all depths
    ax4 = axes[1, 1]
    for depth in depths_list:
        best_trial = min(results[depth]['trials'], key=lambda x: x['final_cost'])
        ax4.plot(best_trial['cost_history'], label=f'Depth {depth}', linewidth=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Cost')
    ax4.set_title('Cost Convergence for Different Depths (Best Trial)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment3_depth_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


# ============================================================================
# Run All Experiments
# ============================================================================

if __name__ == "__main__":
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 80)
    print("QAOA PROTEIN OPTIMIZATION EXPERIMENTS")
    print(f"Protein: {polypeptide_sequence}")
    print(f"Qubits: {len(wires)}")
    print("=" * 80)
    
    # Experiment 1: Energy tracking
    exp1_results = experiment_1_energy_tracking(depth=2, num_iterations=200)
    
    # Experiment 2: Mixer comparison
    exp2_results = experiment_2_mixer_comparison(depth=2, num_iterations=100)
    
    # Experiment 3: Depth analysis
    exp3_results = experiment_3_depth_analysis(depths=[1, 2, 3, 4, 5], num_iterations=100)
    
    # Save all results to JSON
    all_results = {
        'protein_sequence': polypeptide_sequence,
        'num_qubits': len(wires),
        'timestamp': timestamp,
        'experiment_1': exp1_results,
        'experiment_2': exp2_results,
        'experiment_3': exp3_results
    }
    
    with open(f'experiment_results_{timestamp}.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Results saved to: experiment_results_{timestamp}.json")
    print("Figures saved as: experiment1_energy_tracking.png")
    print("                  experiment2_mixer_comparison.png")
    print("                  experiment3_depth_analysis.png")
    print("=" * 80)