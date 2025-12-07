from pennylane import qaoa
from pennylane import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

import pennylane as qml

import hamiltonian
import python_codon_tables as pct

# def get_first_layer_weight(codon):
#     #example: codon 0 has weight 1.0, codon 1 has weight 0.8
#     if codon == 0:
#         return 1.0
#     else:
#         return 0.8

# def get_interlayer_weight(layer, prev_codon, codon):
#     #example: weight depends on layer and codon transition
#     base_weight = 0.5
#     if prev_codon == codon:
#         return base_weight + 0.2  #higher weight for same codon transition
#     else:
#         return base_weight - 0.1  #lower weight for different codon transition

# def get_return_weight(layer):
#     #example: fixed weight for return edges
#     return 0.3 + 0.1 * layer  #increasing weight with layer

# def generate_codon_graph(max_length = 4, num_codons = 2):
#     graph = nx.DiGraph()

#     #node 0 is our "start" node
#     #each "layer" of codons takes on indices 1-num_codons, num_codons+1 to 2*num_codons, etc.
#     #the first layer connects to the start node
#     #each subsequent layer connects to the previous layer
#     #each node loops back to the start node as well

#     weighted_edges = []
#     for layer in range(max_length):
#         for codon in range(num_codons):
#             current_node = layer * num_codons + codon + 1
#             if layer == 0:
#                 #connect to start node
#                 weight = get_first_layer_weight(codon)
#                 weighted_edges.append((0, current_node, weight))
#             else:
#                 #connect to previous layer
#                 for prev_codon in range(num_codons):
#                     prev_node = (layer - 1) * num_codons + prev_codon + 1
#                     weight = np.random.rand()  #random weight for the edge
#                     weight = get_interlayer_weight(layer, prev_codon, codon)
#                     weighted_edges.append((prev_node, current_node, weight))
#             #loop back to start node
#             weight = get_return_weight(layer)
#             weighted_edges.append((current_node, 0, weight))  #fixed weight for loopback
    
#     graph.add_weighted_edges_from(weighted_edges)
#     return graph

# num_codons = 2

# graph = generate_codon_graph(max_length=2, num_codons=num_codons)
# # positions = nx.spring_layout(graph, seed=1)
# # nx.draw(graph, with_labels=True, pos=positions)
# # plt.show()
# print("Edge weights in the graph:")
# for u, v, data in graph.edges(data=True):
#     if 'weight' in data:
#         print(f"Edge ({u}, {v}) has weight: {data['weight']}")
#     else:
#         print(f"Edge ({u}, {v}) has no explicit weight attribute.")

# cost_h, mixer_h, wires_to_edges = qaoa.max_weight_cycle(graph)

print(pct.available_codon_tables_names)
e_coli_table = pct.get_codons_table('e_coli_316407')

print(e_coli_table)

amino_acids = e_coli_table

polypeptide_sequence = "**"
Q, offset = hamiltonian.Q(polypeptide_sequence, 1.5)
h, J, offset = hamiltonian.Q_to_Ising(Q, offset)

#1 wire per possibility codon per position

wires = range(len(h))

    # ---------- COST HAMILTONIAN ----------
    # for ki, v in h.items():  # single-qubit terms
    #     qml.RZ(2 * gammas[layer] * v / wmax, wires=ki[0])
    # for kij, vij in J.items():  # two-qubit terms
    #     qml.CNOT(wires=[kij[0], kij[1]])
    #     qml.RZ(2 * gammas[layer] * vij / wmax, wires=kij[1])
    #     qml.CNOT(wires=[kij[0], kij[1]])

#Hamiltonian of the Ising model with our h and J values
cost_h = sum([h[i] * qml.Z(wires=i) for i in range(len(h))]) + \
sum([J[i, j] * qml.Z(wires=i) @ qml.Z(wires=j) for i in range(len(h)) for j in range(i)])

graph = nx.Graph()

for aa in polypeptide_sequence:
    codons = []
    for i in range(len(amino_acids[aa])):
        codon = len(codons) + len(graph.nodes)
        codons.append(codon)
    graph.add_nodes_from(codons)
    #fully connect codons for the same amino acid
    for i in range(len(codons)):
        for j in range(i + 1, len(codons)):
            graph.add_edge(codons[i], codons[j])


mixer_h = qaoa.mixers.xy_mixer(graph)

print("Cost Hamiltonian", cost_h)
print("Mixer Hamiltonian", mixer_h)

depth = 1

def qaoa_layer(gamma, alpha):
    qaoa.cost_layer(gamma, cost_h)
    qaoa.mixer_layer(alpha, mixer_h)

def circuit(params, **kwargs):
    for w in wires:
        qml.Hadamard(wires=w)
    qml.layer(qaoa_layer, depth, params[0], params[1])

dev = qml.device("qulacs.simulator", wires=wires)

@qml.qnode(dev)
def cost_function(params):
    circuit(params)
    return qml.expval(cost_h)

optimizer = qml.AdamOptimizer()
#params = np.ones((2, depth), requires_grad=True) * 0.5

# for i in range(steps):
#     print(i)
#     params = optimizer.step(cost_function, params)

# print("Optimal Parameters")
# print(params)

num_attempts = 10
steps = 10

best_params = np.zeros((2, depth))
best_cost = float('-inf')
for _ in range(num_attempts):
    print("Attempt" + str(_))
    params = np.random.rand(2, depth)  # Random initialization
    for i in range(steps):
        params = optimizer.step(cost_function, params)
    final_cost = cost_function(params)
    print("Final cost:", final_cost)
    if final_cost > best_cost:
        best_cost = final_cost
        best_params = params
print("Best Parameters")
print(best_params)


@qml.qnode(dev)
def probability_circuit(gamma, alpha):
    circuit([gamma, alpha])
    return qml.probs(wires=wires)

params = best_params
probs = probability_circuit(params[0], params[1])

plt.style.use("seaborn-v0_8")
plt.bar(range(2 ** len(wires)), probs)
plt.show()