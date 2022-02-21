# utils.py
from qutip import partial_transpose, Qobj, tensor
from qutip.measurement import measurement_statistics
from itertools import combinations
from numpy import argsort

# General
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def evolve(rho, oper, direction):
    if direction == 'effect':
        rho_out = oper * rho * oper.dag()
    elif direction == 'cause':
        rho_out = oper.dag() * rho * oper
    else:
        raise ValueError("Direction must be 'cause' or 'effect'.")
    
    return rho_out

def evolve_measurement(rho, measurement, direction):
    ops, targets = measurement
    if direction == 'effect':
        collapsed_states, probabilities = measurement_statistics(rho, ops, targets=targets)
        rho_out = sum([p*collapsed_states[c] for c,p in enumerate(probabilities) if collapsed_states[c] is not None])
                
    elif direction == 'cause':
        rho_out = oper.dag() * rho * oper
    else:
        raise ValueError("Direction must be 'cause' or 'effect'.")
    
    return rho_out


def sort_tensor(rho, partial_indices):
    indices = []
    for i in partial_indices:
        indices.extend(list(i))

    sort_index = argsort(indices)

    return rho.permute(sort_index)

# Diagonalize matrix
def diagonalize_matrix(rho):
    _, evecs = rho.eigenstates()
    rho_D = rho.transform(evecs)
    return rho_D, evecs

def entanglement_check_2qubit(rho, tol=1e-12):
    rho_pt = partial_transpose(rho, [0,1])
    evals, _ = rho_pt.eigenstates()
    evals[abs(evals) < tol] = 0
    return any(evals < 0)


def entanglement_check_3qubit(rho, tol=1e-12):
    masks = ([1,0,0], [0,1,0], [0,0,1])
    entangled = True
    for mask in masks:
        rho_pt = partial_transpose(rho, mask)
        evals, _ = rho_pt.eigenstates()
        evals[abs(evals) < tol] = 0
        entangled = entangled * any(evals < 0)

    return entangled

def entanglement_partition(rho):
    # Note: for 3 qubits the Peres-Horodecki criterion is only a necessary condition
    # a negative partial trace is sufficient for entanglement, but a positive partial trace
    # could still be entangled. This means we might miss some forms of entanglement.
    # (see Bennett et al., 1999, Unextendible product bases and bound entanglement)
    n_qubits = len(rho.dims[0])
    
    # default completely separable
    ent_partition = list(combinations(range(n_qubits), 1))
    
    if n_qubits == 2:
        if entanglement_check_2qubit(rho):
            ent_partition = [tuple(range(n_qubits))]    
    else:
    # 3 qubits
        if entanglement_check_3qubit(rho):
            ent_partition = [tuple(range(n_qubits))]
        else:
            for sub2 in combinations(range(n_qubits), 2):
                if entanglement_check_2qubit(rho.ptrace(sub2)):
                    ent_partition = [tuple(set(range(n_qubits))-set(sub2)), sub2]
                    # this should only be true for one subset
                    return ent_partition

    return ent_partition


def decorrelate_rho(rho, ent_partition): 
    p_rho_parts = []
    for part in ent_partition:
        p_rho = rho.ptrace(list(part))
        p_rho_parts.append(p_rho)

    rho_p_product = tensor(p_rho_parts)

    return sort_tensor(rho_p_product, ent_partition)