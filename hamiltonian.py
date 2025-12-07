import numpy as np
import python_codon_tables as pct

print(pct.available_codon_tables_names)
e_coli_table = pct.get_codons_table('e_coli_316407')

print(e_coli_table)

amino_acids = e_coli_table

all_codons = []
for aa in amino_acids:
    codons = amino_acids[aa].keys()
    all_codons += list(codons)

def get_s():
    """Generate the s vector for codon optimization.

    Returns:
        np.ndarray: The s vector for the QUBO codon optimization problem.
    """
    s = {}
    for codon in all_codons:
        s[codon] = codon.count('G') + codon.count('C')

    return s


def get_xi(amino_acid, codon, epsilon_f=1e-6):
    """Generate the xi vector for codon optimization.

    Returns:
        np.ndarray: The xi vector for the QUBO codon optimization problem.
    """
    C = amino_acids[amino_acid][codon]
    xi = np.log(epsilon_f + C)

    return xi


def most_consecutive_characters(sequence):
    max_count = 1
    current_count = 1

    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i - 1]:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 1

    return max_count

def r(codon_j, codon_i):
    """Generate the r value for codon optimization. (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259101)

    Returns:
        int: The r value for the QUBO codon optimization problem.
    """
    #r(Ci, Cj) represent a quadratic function that returns the maximum number of repeated sequential nucleotides 
    # between codons Ci and Cj, shifted to the origin for null cases by subtracting one

    sequence = codon_j + codon_i
    repeats = most_consecutive_characters(sequence)
    r = (repeats ** 2) - 1
    return r



def Q(polypeptide_sequence, rho_t, c_f=1.0, c_gc=1.0, c_r = 1.0, tau = 1e3):
    """Generate the Q matrix for a given sequence length and number of codons.

    Args:
        sequence_length (int): The length of the polypeptide sequence.
    Returns:
        np.ndarray: The Q matrix for the QUBO codon optimization problem.
    """
    N = len(polypeptide_sequence)

    s = get_s()

    #single body terms
    h = []
    for i_char in range(len(polypeptide_sequence)):
        amino_acid = polypeptide_sequence[i_char]
        for codon in amino_acids[amino_acid]:
            weight = 0
            weight += c_f * get_xi(amino_acid, codon)
            weight += -2 * rho_t * c_gc * s[codon] / N
            weight += c_gc * s[codon] ** 2 / (N ** 2)
            h.append(weight)

    h = np.array(h)
    epsilon = np.max(h) + 1e-2
    h += -epsilon

    Q = np.diag(h)

    #TODO: consider an adaptive tau, tau = max(|h_i|) × 100?
    tau = np.max(np.abs(h)) * 100

    i = 0
    for i_char in range(len(polypeptide_sequence)):
        amino_acid = polypeptide_sequence[i_char]
        for i_codon in amino_acids[amino_acid]:
            j = 0
            j_char = 0

            while j < i:
                amino_acid_j = polypeptide_sequence[j_char]
                for j_codon in amino_acids[amino_acid_j]:
                    if j >= i:
                        break
                    weight = 0
                    weight += 2 * c_gc * s[i_codon] * s[j_codon] / (N ** 2)
                    if j_char == i_char - 1:
                        weight += c_r * r(j_codon, i_codon)
                    if i_char == j_char:
                        weight += tau
                    Q[i, j] = weight
                    Q[j, i] = weight  # Symmetric matrix
                    j += 1
                j_char += 1
    
            i += 1

    offset = c_gc * rho_t ** 2

    return Q, offset


def Q_to_Ising(Q, offset):
    #As seen in https://pennylane.ai/qml/demos/tutorial_QUBO
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))

    for i in range(n):
        h[i] -= Q[i, i] / 2
        offset += Q[i, i] / 2
        for j in range(i - 1, -1, -1):
            J[i, j] += Q[i, j] / 4
            h[i] -= Q[i, j] / 4
            h[j] -= Q[i, j] / 4
            offset += Q[i, j] / 4

    return h, J, offset


def Ising_energy(z, h, J, offset):
    #As seen in https://pennylane.ai/qml/demos/tutorial_QUBO
    energy = offset
    n = len(z)
    for i in range(n):
        energy += h[i] * z[i]
        for j in range(i - 1, -1, -1):
            energy += J[i, j] * z[i] * z[j]
    return energy