# utils.py


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
    