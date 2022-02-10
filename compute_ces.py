# compute_ces.py

import pyphi
from qutip import Qobj, tensor
from itertools import combinations
from utils import evolve
from intrinsic_difference import intrinsic_difference

rho_mm = Qobj([[0.5, 0.],[0., 0.5]])

                  
def decorrelate_rho_p(rho_p, ent_partition):
    # only works for 2 qubits, needs reordering for more
    # ent_partition is expected to be sorted, for 2 qubits it is either [(0,1)] 
    # in which case nothing is done, or [(0,), (1,)], in which case rho_p becomes 
    # the product distribution
    p_rho_parts = []
    for part in ent_partition:
        p_rho = rho_p.ptrace(list(part))
        p_rho_parts.append(p_rho)

    rho_p_product = tensor(p_rho_parts)
    return rho_p_product

def evolve_mpart_2qubit(m_rho, ind_m, oper, direction, ent_partition = None):
    #this only works for 2 qubit systems
    #rho_m gets extended by rho_mm and then evolved
    if ind_m == (0,):
        m_rho = tensor(m_rho, rho_mm)
        ent_partition = list(combinations(range(len(m_rho.dims[0])), 1)) #rho_mm breaks entanglement
    elif ind_m == (1,):
        m_rho = tensor(rho_mm, m_rho)
        ent_partition = list(combinations(range(len(m_rho.dims[0])), 1)) #rho_mm breaks entanglement
    # evolve
    p_rho = evolve(m_rho, oper, direction)

    if ent_partition: 
        p_rho = decorrelate_rho_p(p_rho, ent_partition)

    return m_rho, p_rho


# find mip (for a mechanism purview pair)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def find_mip(rho_m, ind_m, rho_p, ind_p, oper, direction = 'effect', ent_partition = None):
    # mechanism and purview state are given (rho_m, rho_p)
    # purview state is already causally marginalized
    # compute ID over all possible partitions
       
    partitions = pyphi.partition.mip_partitions(ind_m, ind_p)
    phi_mip = float("inf")
    state = []
    
    for partition in partitions:
        p_rho_parts = []
        #phi = evaluate_partition()
        for part in partition:
            if len(part.mechanism) < 1:
                if len(part.purview) > 0:
                # purview is max ent
                    p_rho = tensor([rho_mm for i in part.purview])
                    p_rho_parts.append((p_rho, part.purview))
            else:
                if len(part.purview) > 0: 
                    m_rho = rho_m.ptrace(list(part.mechanism))
                    # send all mechanisms into evolve_mpart_2qubit for causal marginalization
                    _, p_rho = evolve_mpart_2qubit(m_rho, part.mechanism, oper, direction, ent_partition)
                    # do partial trace to get purview elements
                    p_rho = p_rho.ptrace(list(part.purview))
                    p_rho_parts.append((p_rho, part.purview))

        if len(p_rho_parts) > 1:
            p_rho_parts = sorted(p_rho_parts, key=lambda x: x[1])
            rho_p_partitioned = tensor([p[0] for p in p_rho_parts])
        else:
            rho_p_partitioned = p_rho_parts[0][0]
        
        # print(ind_m, ind_p)
        # print(rho_p)
        # print(rho_p_partitioned)
        phi, state = intrinsic_difference(rho_p, rho_p_partitioned)

        if phi == 0:
            return 0, partition, state

        if phi < phi_mip:
            phi_mip = phi
            mip = partition
            p_state = state
    
    return phi_mip, mip, p_state

    



    # powerset = pyphi.utils.powerset(range(num_el), nonempty=True)
    # rho_parts = []
    # # The powerset does not include the empty set, but does include the full rho
    # for p in powerset:
    #     rho_part = rho.ptrace(p)
    #     rho_parts.append(rho_part)

       
    #     print('ID |AB> AX: ', find_purview_AB(rho_eAB, rho_eAX))
    #     print('--------------')
    #     print('ID |AB> XB: ', find_purview_AB(rho_eAB, rho_eXB))
    #     print('--------------')
    #     # Note that there are more possible partitions that I haven't implemented yet
    #     #vvv A/A X B/B, or A/B X B/A


# find purview (for a mechanism)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def find_mice(rho_m, ind_m, rho_maxp, oper, direction = 'effect'):
    # rho_m is density matrix of system size properly extended
    purviews = pyphi.utils.powerset(range(len(rho_maxp.dims[0])), nonempty=True)

    phi_max = 0.
    max_purview = []
    max_mip = []
    max_state = []

    for purview in purviews:
        rho_p = rho_maxp.ptrace(list(purview))
        phi_mip, mip, p_state = find_mip(rho_m, ind_m, rho_p, purview, oper, direction = direction)
        print('p: ', purview, ' phi: ', phi_mip)

        if phi_max < phi_mip:
            phi_max = phi_mip
            max_mip = mip
            max_state = p_state
            max_purview = purview
        elif phi_max == phi_mip and len(purview) > len(max_purview):
            phi_max = phi_mip
            max_mip = mip
            max_state = p_state
            max_purview = purview
    
    return phi_max, max_mip, max_state, max_purview
    
# compute all mechanisms (for a system state)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compute_ces(rho_maxm, oper, direction = 'effect', ent_partition = None):

    system_size = len(rho_maxm.dims[0])
    mechanisms = pyphi.utils.powerset(range(system_size), nonempty=True)
    ces = []

    if ent_partition is None:
        ent_partition = list(combinations(range(system_size), 1))

    for mechanism in mechanisms:
        print('m: ', mechanism)
        m_rho = rho_maxm.ptrace(list(mechanism))
        rho_m, rho_maxp = evolve_mpart_2qubit(m_rho, mechanism, oper, direction, ent_partition)
        phi_max, max_mip, max_state, max_purview = find_mice(rho_m, mechanism, rho_maxp, oper, direction)
        if phi_max > 0:
            distinction = {'mech': mechanism, 'purview': max_purview, 'phi': phi_max, 'mip': max_mip, 'state': max_state}
            ces.append(distinction)
    
    return ces
