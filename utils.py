# utils.py
from qutip import partial_transpose, Qobj

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

#def permute_qbit_order(qtensor, current_order, new_order):
    # ---> there is already a permute function for Qobj in QuTip!

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
