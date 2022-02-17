# compute_ces.py

from os import system
import pyphi
from qutip import Qobj, tensor
from itertools import combinations
from utils import entanglement_check_2qubit, evolve, decorrelate_rho, sort_tensor, entanglement_partition
from intrinsic_difference import intrinsic_difference
from operator import mul
from functools import reduce

rho_mm = Qobj([[0.5, 0.],[0., 0.5]])


# cause and effect repertoires 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def extend_mechanism_to_system_size(m_rho, ind_m, system_size):
    mech_size = len(ind_m)

    if mech_size < system_size:
        parts = [m_rho]
        for mm in range(system_size-mech_size):
            parts.append(rho_mm) 
        
        parts_ind = [ind_m, tuple(set(range(system_size))-set(ind_m))]
        #print(parts_ind)
        m_rho = sort_tensor(tensor(parts), parts_ind)
    
    return m_rho

def evolve_mpart_effect(m_rho, oper):    
    # first evolve, then partition purview
    p_rho = evolve(m_rho, oper, direction = 'effect')

    ent_partition = entanglement_partition(p_rho)
    
    if len(ent_partition) > 1:
        p_rho = decorrelate_rho(p_rho, ent_partition)
    #vvv
    return m_rho, p_rho


def evolve_mpart_cause(m_rho, ind_m, oper):
    # partition mechanism, then evolve parts, and multiple for purview
    system_size = len(oper.dims[0])
    ent_partition = entanglement_partition(m_rho)

    if len(ent_partition) > 1:
        p_rho_parts = []
        for part in ent_partition:
            m_rho_part = m_rho.ptrace(list(part))
            m_rho_part = extend_mechanism_to_system_size(m_rho_part, part, system_size)
            p_rho_part = evolve(m_rho_part, oper, direction='cause')
            p_rho_parts.append(p_rho_part)

        p_rho = reduce(mul, p_rho_parts, 1).unit()
    else:
        p_rho = evolve(m_rho, oper, direction='cause')

    return m_rho, p_rho

def evolve_mpart(m_rho, ind_m, oper, direction):
    
    system_size = len(oper.dims[0])
    
    m_rho = extend_mechanism_to_system_size(m_rho, ind_m, system_size)

    if direction == 'effect':
        return evolve_mpart_effect(m_rho, oper)
    else: 
        return evolve_mpart_cause(m_rho, ind_m, oper)


# find mip (for a mechanism purview pair)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def find_mip(rho_m, ind_m, rho_p, ind_p, oper, direction = 'effect'):
    # mechanism and purview state are given (rho_m, rho_p)
    # purview state is already causally marginalized
    # compute ID over all possible partitions
       
    partitions = pyphi.partition.mip_partitions(ind_m, ind_p)
    phi_mip = float("inf")
    state = []
    
    for partition in partitions:
        p_rho_parts = []
        parts_ind = []
        for part in partition:
            if len(part.mechanism) < 1:
                if len(part.purview) > 0:
                # purview is max ent
                    p_rho = tensor([rho_mm for i in part.purview])
                    p_rho_parts.append(p_rho)
                    parts_ind.append(part.purview)
            else:
                if len(part.purview) > 0: 
                    m_rho = rho_m.ptrace(list(part.mechanism))
                    # send all mechanisms into evolve_mpart_2qubit for causal marginalization
                    _, p_rho = evolve_mpart(m_rho, part.mechanism, oper, direction)
                    # do partial trace to get purview elements
                    p_rho = p_rho.ptrace(part.purview)
                    p_rho_parts.append(p_rho)
                    parts_ind.append(part.purview)

        #print(parts_ind)
        if len(p_rho_parts) > 1:
            rho_p_partitioned = sort_tensor(tensor(p_rho_parts), parts_ind)
        else:
            rho_p_partitioned = p_rho_parts[0]
        
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
def compute_ces(rho_maxm, oper, direction = 'effect'):

    system_size = len(rho_maxm.dims[0])
    mechanisms = pyphi.utils.powerset(range(system_size), nonempty=True)
    ces = []

    for mechanism in mechanisms:
        print('m: ', mechanism)
        m_rho = rho_maxm.ptrace(list(mechanism))
        rho_m, rho_maxp = evolve_mpart(m_rho, mechanism, oper, direction)
        phi_max, max_mip, max_state, max_purview = find_mice(rho_m, mechanism, rho_maxp, oper, direction)
        if phi_max > 0:
            distinction = {'mech': mechanism, 'purview': max_purview, 'phi': phi_max, 'mip': max_mip, 'state': max_state}
            ces.append(distinction)
    
    return ces
