# intrinsic_differency_infofirst.py

# Import packages
from qutip import *

from qutip.sparse import sp_eigs
from numpy import e, real, around, inf, prod
from numpy.lib.scimath import log, log2

from numpy import conj, e, inf, imag, inner, real
from numpy.lib.scimath import log, log2
from qutip.states import ket2dm, maximally_mixed_dm
from qutip.sparse import sp_eigs

def intrinsic_difference_infofirst(rho, sigma, base=2, sparse=False, tol=1e-12, debug = False):
    """
    Calculates the intrinsic difference ID(rho||sigma) between two density
    matrices.

    Parameters
    ----------
    rho : :class:`qutip.Qobj`
        First density matrix (or ket which will be converted to a density
        matrix).
    sigma : :class:`qutip.Qobj`
        Second density matrix (or ket which will be converted to a density
        matrix).
    base : {e,2}
        Base of logarithm. Defaults to e.
    sparse : bool
        Flag to use sparse solver when determining the eigenvectors
        of the density matrices. Defaults to False.
    tol : float
        Tolerance to use to detect 0 eigenvalues or dot producted between
        eigenvectors. Defaults to 1e-12.

    Returns
    -------
    qid : float
        Value of quantum intrinsic difference. Guaranteed to be greater than zero
        and should equal zero only when rho and sigma are identical.

    r_max : 

    Examples
    --------

    First we define two density matrices:

    >>> rho = qutip.ket2dm(qutip.ket("00"))
    >>> sigma = rho + qutip.ket2dm(qutip.ket("01"))
    >>> sigma = sigma.unit()

    Then we calculate their relative entropy using base 2 (i.e. ``log2``)
    and base e (i.e. ``log``).

    >>> qutip.intrinsic_difference(rho, sigma, base=2)
    1.0
    >>> qutip.intrinsic_difference(rho, sigma)
    0.6931471805599453

    References
    ----------

    See Nielsen & Chuang, "Quantum Computation and Quantum Information",
    Section 11.3.1, pg. 511 for a detailed explanation of quantum relative
    entropy. The intrinsic difference is related to the relative entropy, but
    instead of summing over the basis states of rho it evaluates the basis state
    that maximizes the difference between rho and sigma.
    """
    if rho.isket:
        rho = ket2dm(rho)
    if sigma.isket:
        sigma = ket2dm(sigma)
    if not rho.isoper or not sigma.isoper:
        raise TypeError("Inputs must be density matrices.")
    if rho.dims != sigma.dims:
        raise ValueError("Inputs must have the same shape and dims.")
    if base == 2:
        log_base = log2
    elif base == e:
        log_base = log
    else:
        raise ValueError("Base must be 2 or e.")
    # S(rho || sigma) = sum_i(p_i log p_i) - sum_ij(p_i P_ij log q_i)
    #
    # S is +inf if the kernel of sigma (i.e. svecs[svals == 0]) has non-trivial
    # intersection with the support of rho (i.e. rvecs[rvals != 0]).
    rvals, rvecs = sp_eigs(rho.data, rho.isherm, vecs=True, sparse=sparse)
    if any(abs(imag(rvals)) >= tol):
        raise ValueError("Input rho has non-real eigenvalues.")
    rvals = around(real(rvals), 12)


    svals, svecs = sp_eigs(sigma.data, sigma.isherm, vecs=True, sparse=sparse)
    if any(abs(imag(svals)) >= tol):
        raise ValueError("Input sigma has non-real eigenvalues.")
    svals = real(svals)
    
    #Only evaluate those with max information before partition
    rvals_maxent = abs(rvals - 1/len(rvals))
    max_rval = max(rvals_maxent)
    max_rval_ind = [i for i,j in enumerate(rvals_maxent) if abs(j - max_rval) <= tol]
    print(max_rval_ind)

    # Calculate inner products of eigenvectors and return +inf if kernel
    # of sigma overlaps with support of rho.
    P = abs(inner(rvecs, conj(svecs))) ** 2
    if (rvals >= tol) @ (P >= tol) @ (svals < tol):
        return inf, None
    # Avoid -inf from log(0) -- these terms will be multiplied by zero later
    # anyway
    svals[abs(svals) < tol] = 1
    nzrvals = rvals[abs(rvals) >= tol]
    
    # Calculate QID  
    QID_all = [abs(r * (log_base(r) - P[i] @ log_base(svals))) for i,r in enumerate(rvals) if abs(r) >=tol and i in max_rval_ind]
    #without abs
    #QID_all = [r * (log_base(r) - P[i] @ log_base(svals)) for i,r in enumerate(rvals) if abs(r) >=tol and i in max_rval_ind]
    r_vecs = [rvecs[i] for i,r in enumerate(rvals) if abs(r) >=tol and i in max_rval_ind]

    if debug:
        print("rvals: ", rvals)
        print("rvecs: ", rvecs)
    
        print("svals: ", svals)
        print("svecs: ", svecs)

        print("svals: ", svals)
        print("nzrvals: ", nzrvals)
    
        print("P: ", P)

        print("QID: ", QID_all, "r_vecs: ", r_vecs)

    QID = max(QID_all)
    max_vecs = [r_vecs[i] for i, j in enumerate(QID_all) if abs(j - max(QID_all)) < tol]

    # the quantum relative entropy is guaranteed to be >= 0, so we clamp the
    # calculated value to 0 to avoid small violations of the lower bound.
    return around(max(0, QID), 12), max_vecs