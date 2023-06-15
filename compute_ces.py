# compute_ces.py

from os import system
import pyphi
from qutip import Qobj, tensor
from itertools import combinations
from utils import evolve_unitary, evolve_cptp, decorrelate_rho, sort_tensor, entanglement_partition
from intrinsic_difference_Barbosa2020 import intrinsic_difference
from operator import mul
from functools import reduce

rho_mm = Qobj([[0.5, 0.],[0., 0.5]])

# rho_p: \pi product distribution (cause/effect repertoire)
# p_rho: p^Z\m no product

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

def evolve_mpart_effect(m_rho, oper, purview):    
    # first evolve, then partition purview
    if type(oper) is list:
        evolve = evolve_cptp
    else: 
        evolve = evolve_unitary
        
    p_rho = evolve(m_rho, oper, direction = 'effect')

    ent_partition = entanglement_partition(p_rho)
    
    if len(ent_partition) > 1:
        rho_p = decorrelate_rho(p_rho, ent_partition)
    else:
        rho_p = p_rho
    
    # do partial trace to get purview elements                   
    return rho_p.ptrace(list(purview))


def evolve_mpart_cause(m_rho, oper, purview):
    # partition mechanism, then evolve parts, and multiple for purview
    system_size = len(m_rho.dims[0])
    ent_partition = entanglement_partition(m_rho)

    if type(oper) is list:
        evolve = evolve_cptp
    else: 
        evolve = evolve_unitary

    if len(ent_partition) > 1:
        p_rho_parts = []
        for part in ent_partition:
            m_rho_part = m_rho.ptrace(list(part))
            m_rho_part = extend_mechanism_to_system_size(m_rho_part, part, system_size)
            # do partial trace here to get only the actual purview elements
            p_rho_part = evolve(m_rho_part, oper, direction='cause').ptrace(list(purview))
            p_rho_parts.append(p_rho_part)

        rho_p = reduce(mul, p_rho_parts, 1).unit()
    else:
        rho_p = evolve(m_rho, oper, direction='cause').ptrace(list(purview))
    
    return rho_p

def evolve_mpart(m_rho, oper, direction, purview): 
    if direction == 'effect':
        return evolve_mpart_effect(m_rho, oper, purview)
    else: 
        return evolve_mpart_cause(m_rho, oper, purview)


# find mip (for a mechanism purview pair)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def find_mip(rho_m, ind_m, rho_p, ind_p, oper, direction = 'effect'):
    # mechanism and purview state are given (rho_m, rho_p)
    # mechanism state is already extended
    # purview state is already causally marginalized
    # compute ID over all possible partitions
    system_size = len(rho_m.dims[0])
    partitions = pyphi.partition.mip_partitions(ind_m, ind_p)
    phi_mip = float("inf")
    phi_mip_norm = float("inf")
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
                    m_rho = extend_mechanism_to_system_size(m_rho, part.mechanism, system_size)
                    p_rho = evolve_mpart(m_rho, oper, direction, part.purview)
                    
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

        phi_norm = phi * pyphi.models.mechanism.normalization_factor(partition)

        if phi_norm < phi_mip_norm:
            phi_mip_norm = phi_norm
            phi_mip = phi
            mip = partition
            p_state = state
    
    return phi_mip, mip, p_state


# find purview (for a mechanism)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def find_mice(m_rho, ind_m, oper, direction = 'effect'):
    # m_rho is density matrix of system size properly extended
    system_size = len(m_rho.dims[0])
    purviews = pyphi.utils.powerset(range(system_size), nonempty=True)

    phi_max = 0.
    max_purview = []
    max_mip = []
    max_state = []

    for purview in purviews:
        rho_p = evolve_mpart(m_rho, oper, direction, purview)
        phi_mip, mip, p_state = find_mip(m_rho, ind_m, rho_p, purview, oper, direction = direction)
        print('p: ', purview, ' phi: ', phi_mip)

        if phi_max < phi_mip:
            phi_max = phi_mip
            max_mip = mip
            max_state = p_state
            max_purview = purview
        # choose larger purview in case of ties
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
        m_rho = extend_mechanism_to_system_size(m_rho, mechanism, system_size)
        
        phi_max, max_mip, max_state, max_purview = find_mice(m_rho, mechanism, oper, direction)
        if phi_max > 0:
            distinction = {'mech': mechanism, 'purview': max_purview, 'phi': phi_max, 'mip': max_mip, 'state': max_state}
            ces.append(distinction)
    
    return ces
