from pennylane import qaoa
from pennylane import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import pennylane as qml

import hamiltonian
import python_codon_tables as pct

from qaoa_proteins_initialization_experiments import init_w_state_per_position

# Load codon table
e_coli_table = pct.get_codons_table('e_coli_316407')
amino_acids = e_coli_table

# Define the optimization problem
polypeptide_sequence = "MALW"
Q, offset = hamiltonian.Q(polypeptide_sequence, 1.5)
h, J, offset = hamiltonian.Q_to_Ising(Q, offset)

wires = range(len(h))

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

# Build mixer graph
graph = nx.Graph()
for aa in polypeptide_sequence:
    codons = []
    for i in range(len(amino_acids[aa])):
        codon = len(codons) + len(graph.nodes)
        codons.append(codon)
    graph.add_nodes_from(codons)
    # Fully connect codons for the same amino acid
    for i in range(len(codons)):
        for j in range(i + 1, len(codons)):
            graph.add_edge(codons[i], codons[j])

mixer_h = qaoa.mixers.xy_mixer(graph)

# QAOA circuit setup
depth = 2  # Increased depth for better optimization

def qaoa_layer(gamma, alpha):
    qaoa.cost_layer(gamma, cost_h)
    qaoa.mixer_layer(alpha, mixer_h)

def circuit(params, **kwargs):
    for w in wires:
        init_w_state_per_position(position_to_qubits)
    qml.layer(qaoa_layer, depth, params[0], params[1])

dev = qml.device("lightning.qubit", wires=wires)

@qml.qnode(dev)
def cost_function(params):
    circuit(params)
    return qml.expval(cost_h)

@qml.qnode(dev)
def probability_circuit(params):
    circuit(params)
    return qml.probs(wires=wires)

# Improved optimization with multiple strategies
def optimize_qaoa(num_attempts=20, steps_per_attempt=50):
    """Run multiple optimization attempts with different strategies."""
    
    results = []
    
    # Strategy 1: Random initialization
    print("=" * 60)
    print("Strategy 1: Random Initialization")
    print("=" * 60)
    optimizer = qml.GradientDescentOptimizer(stepsize=0.1)
    
    for attempt in range(num_attempts // 2):
        params = np.random.uniform(0, 2*np.pi, size=(2, depth))
        
        for step in range(steps_per_attempt):
            params = optimizer.step(cost_function, params)
            
            if step % 10 == 0:
                cost = cost_function(params)
                print(f"Attempt {attempt+1}/{num_attempts//2}, Step {step}: Cost = {cost:.6f}")
        
        final_cost = cost_function(params)
        results.append({
            'params': params.copy(),
            'cost': final_cost,
            'strategy': 'random'
        })
    
    # Strategy 2: Warm start from best result
    print("\n" + "=" * 60)
    print("Strategy 2: Warm Start from Best Result")
    print("=" * 60)
    
    best_so_far = min(results, key=lambda x: x['cost'])
    optimizer = qml.AdamOptimizer(stepsize=0.02)
    
    for attempt in range(num_attempts // 2):
        # Add small perturbations to best parameters
        params = best_so_far['params'] + np.random.normal(0, 0.3, size=(2, depth))
        
        for step in range(steps_per_attempt):
            params = optimizer.step(cost_function, params)
            
            if step % 10 == 0:
                cost = cost_function(params)
                print(f"Attempt {attempt+1}/{num_attempts//2}, Step {step}: Cost = {cost:.6f}")
        
        final_cost = cost_function(params)
        results.append({
            'params': params.copy(),
            'cost': final_cost,
            'strategy': 'warm_start'
        })
    
    # Sort results by cost
    results.sort(key=lambda x: x['cost'])
    
    return results

# Run optimization
print("Starting QAOA optimization...")
results = optimize_qaoa(num_attempts=20, steps_per_attempt=50)

# Get best result
best_result = results[0]
print("\n" + "=" * 60)
print("OPTIMIZATION COMPLETE")
print("=" * 60)
print(f"Best Cost: {best_result['cost']:.6f}")
print(f"Best Parameters:\n{best_result['params']}")

# Analyze and display solutions
def decode_bitstring(bitstring, polypeptide_sequence, amino_acids):
    """Convert a bitstring to a codon sequence."""
    codon_sequence = []
    idx = 0
    
    for aa in polypeptide_sequence:
        codons_for_aa = list(amino_acids[aa].keys())
        
        # Find which codon is selected (should be one-hot encoded)
        selected = None
        for i, codon in enumerate(codons_for_aa):
            if idx < len(bitstring) and bitstring[idx] == '1':
                selected = codon
                break
            idx += 1
        
        codon_sequence.append(selected if selected else "???")
    
    return codon_sequence

def analyze_top_solutions(results, top_k=10):
    """Analyze and display the top k solutions."""
    print("\n" + "=" * 60)
    print(f"TOP {top_k} SOLUTIONS")
    print("=" * 60)
    
    best_params = results[0]['params']
    probs = probability_circuit(best_params)
    
    # Get indices of top probability states
    top_indices = np.argsort(probs)[-top_k:][::-1]
    
    print(f"\n{'Rank':<6} {'Probability':<12} {'Bitstring':<20} {'Energy':<12} {'Codon Sequence'}")
    print("-" * 80)
    
    valid_solutions = []
    
    for rank, idx in enumerate(top_indices, 1):
        bitstring = format(idx, f'0{len(wires)}b')
        prob = probs[idx]
        
        # Convert bitstring to z values (-1, +1) for Ising
        z = np.array([1 if b == '0' else -1 for b in bitstring])
        energy = hamiltonian.Ising_energy(z, h, J, offset)
        
        # Decode to codon sequence
        codon_seq = decode_bitstring(bitstring, polypeptide_sequence, amino_acids)
        codon_str = '-'.join(codon_seq)
        
        print(f"{rank:<6} {prob:<12.6f} {bitstring:<20} {energy:<12.6f} {codon_str}")
        
        valid_solutions.append({
            'rank': rank,
            'bitstring': bitstring,
            'probability': prob,
            'energy': energy,
            'codons': codon_seq
        })
    
    return valid_solutions, probs

# Analyze solutions
valid_solutions, probs = analyze_top_solutions(results, top_k=15)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Probability distribution
ax1 = axes[0, 0]
ax1.bar(range(len(probs)), probs)
ax1.set_xlabel('Bitstring Index')
ax1.set_ylabel('Probability')
ax1.set_title('Full Probability Distribution')
ax1.set_yscale('log')

# Plot 2: Top solutions
ax2 = axes[0, 1]
top_k = 20
top_indices = np.argsort(probs)[-top_k:][::-1]
top_probs = probs[top_indices]
ax2.bar(range(top_k), top_probs)
ax2.set_xlabel('Solution Rank')
ax2.set_ylabel('Probability')
ax2.set_title(f'Top {top_k} Solutions by Probability')

# Plot 3: Energy vs Probability for top solutions
ax3 = axes[1, 0]
energies = []
for idx in top_indices:
    bitstring = format(idx, f'0{len(wires)}b')
    z = np.array([1 if b == '0' else -1 for b in bitstring])
    energy = hamiltonian.Ising_energy(z, h, J, offset)
    energies.append(energy)

ax3.scatter(energies, top_probs)
ax3.set_xlabel('Energy')
ax3.set_ylabel('Probability')
ax3.set_title('Energy vs Probability (Top Solutions)')

# Plot 4: Optimization convergence
ax4 = axes[1, 1]
costs_random = [r['cost'] for r in results if r['strategy'] == 'random']
costs_warm = [r['cost'] for r in results if r['strategy'] == 'warm_start']
ax4.scatter(range(len(costs_random)), costs_random, label='Random Init', alpha=0.6)
ax4.scatter(range(len(costs_random), len(results)), costs_warm, label='Warm Start', alpha=0.6)
ax4.axhline(y=best_result['cost'], color='r', linestyle='--', label=f'Best: {best_result["cost"]:.3f}')
ax4.set_xlabel('Attempt Number')
ax4.set_ylabel('Final Cost')
ax4.set_title('Optimization Attempts')
ax4.legend()

plt.tight_layout()
plt.savefig('qaoa_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Analysis complete! Results saved to 'qaoa_results.png'")
print("=" * 60)