import numpy as np
import sys
from cvxopt import matrix, solvers
from .pymatlab import size, length

solvers.options['show_progress'] = True
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


def cvxlpbnd(c, A, lb, ub, lhs, rhs, Aeq=np.array([]), beq=np.array([]), solver=None):
    def remove_inf(bounds, constraints):
        """
        remove bounds with -inf or inf,
        cvxopt does not work with them
        :param bounds: all bound of constraints
        :param constraints: constraints matrix
        :return: bound and constraints with bound inf or -inf
        """
        if len(bounds) > 0:
            idx_inf = np.where(bounds == np.inf)
            idx_minus_inf = np.where(bounds == -np.inf)
            mask = np.ones(length(bounds), np.bool)
            mask[idx_inf[0]] = 0
            mask[idx_minus_inf[0]] = 0
            return bounds[mask], constraints[mask]
        return bounds, constraints

    def recovery_equal_constraints(bleft, bright, A):
        idx_equals = np.where(bleft == bright)[0]
        if idx_equals.size > 0:
            mask = np.ones(length(bleft), np.bool)
            mask[idx_equals] = 0
            Aeq = np.array(A[~mask])
            beq = np.array(bleft[~mask])
            return bleft[mask], bright[mask], A[mask], Aeq, beq
        return bleft, bright, A, np.array([]), np.array([])

    d = size(c)
    empty_row_constraint = np.array([], dtype=np.int64).reshape(0, d)
    lhs, rhs, A, Aeqi, beqi = recovery_equal_constraints(lhs, rhs, A)
    wrhs, wrA = remove_inf(rhs, A)
    wlhs, wlA = remove_inf(-lhs, -A)
    G_constraint = np.vstack([wrA, wlA])
    h_constraint = np.hstack([wrhs, wlhs])

    wub, wrI = remove_inf(ub, +np.eye(d))
    wlb, wlI = remove_inf(-lb, -np.eye(d))
    G_bound = np.vstack([wrI, wlI])
    h_bound = np.hstack([wub, wlb])

    # create in order to correctly numpy vstack function
    G_constraint = G_constraint if G_constraint.size else empty_row_constraint
    G_bound = G_bound if G_bound.size else empty_row_constraint
    Aeqi = Aeqi if Aeqi.size else empty_row_constraint
    Aeq = Aeq if Aeq.size else empty_row_constraint

    G = matrix(np.vstack([G_constraint, G_bound]))
    h = matrix(np.hstack([h_constraint, h_bound]))
    Aeq = matrix(np.vstack([Aeqi, Aeq]))
    beq = matrix(np.hstack([beqi, beq]))

    return solvers.lp(matrix(c), G, h, A=Aeq, b=beq, solver=solver)  # solver="glpk") #  kktsolver='ldl',


def quadprog(H, q, Aeq, beq, lower, upper):
    d = size(lower)
    ell_lower = matrix(lower, (d, 1))
    ell_upper = matrix(upper, (d, 1))
    P = matrix(H)
    q = matrix(q)
    I = matrix(0.0, (d, d))
    I[::d + 1] = 1.0
    G = matrix([I, -I])
    h = matrix([ell_upper, -ell_lower])
    Aeq = matrix(Aeq)
    beq = matrix(beq)
    return solvers.qp(P=P, q=q, G=G, h=h, A=Aeq, b=beq, kktsolver='ldl', options={'kktreg': 1e-9})
