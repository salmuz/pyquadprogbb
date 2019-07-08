import numpy as np
from .pymatlab import size, isempty, length, isfinite
from .qphelpers import scale, shift
from cvxopt import matrix, solvers
from scipy.sparse import csr_matrix
from sympy import Matrix


def cvxlpbnd(c, A, lb, ub, lhs, rhs, solver=None):
    def remove_inf(bounds, constraints):
        """
        remove bounds with -inf or inf,
        cvxopt does not work with them
        :param bounds: all bound of constraints
        :param constraints: constraints matrix
        :return: bound and constraints with bound inf or -inf
        """
        idx_inf = np.where(bounds == np.inf)
        idx_minus_inf = np.where(bounds == -np.inf)
        mask = np.ones(length(bounds), np.bool)
        mask[idx_inf[0]] = 0
        mask[idx_minus_inf[0]] = 0
        return bounds[mask], constraints[mask]

    def recovery_equal_constraints(bleft, bright, A):
        idx_equals = np.where(bleft == bright)[0]
        if idx_equals.size > 0:
            mask = np.ones(length(bleft), np.bool)
            mask[idx_equals] = 0
            Aeq = matrix(A[~mask])
            beq = matrix(bleft[~mask])
            return bleft[mask], bright[mask], A[mask], Aeq, beq
        return bleft, bright, A, matrix(), matrix()

    d = size(c)
    lhs, rhs, A, Aeq, beq = recovery_equal_constraints(lhs, rhs, A)
    wrhs, wrA = remove_inf(rhs, A)
    wlhs, wlA = remove_inf(-lhs, -A)
    G_constraint = np.vstack([wrA, wlA])
    h_constraint = np.hstack([wrhs, wlhs])

    wub, wrI = remove_inf(ub, +np.eye(d))
    wlb, wlI = remove_inf(-lb, -np.eye(d))
    G_bound = np.vstack([wrI, wlI])
    h_bound = np.hstack([wub, wlb])

    G = matrix(np.vstack([G_constraint, G_bound]))
    h = matrix(np.hstack([h_constraint, h_bound]))
    return solvers.lp(matrix(c), G, h, A=Aeq, b=beq, solver=solver)  # solver="glpk") #  kktsolver='ldl',


def cplex_bnd_solve(AA, bb, INDEQ, LB, UB, index, flag='b'):
    """
    % CPLEX_BND_SOLVE finds the lower and upper bounds for variables involved in the following feasible region:
    %  { x | AA * x <= bb , LB <= x <= UB }
    %
    %  where the equality holds for those indices identified by INDEQ, ie, AA(INDEQ,:) * x == bb(INDEQ).
    %
    % Parameters:
    %       - index: we only calculate bounds for x(index), not on the other compnents of x.
    %       - flag: can take three values:
    %               1) 'l': only lower bounds
    %               2) 'u': only upper bounds
    %               3) 'b': both lower and upper bounds, default
    """
    if index.size == 0:  # if numel(index) == 0 @matlab
        xLB = np.array([])
        xUB = np.array([])
        # time = toc(tStart)
        return xLB, xUB

    # initialize data
    n = length(LB)
    meq = length(INDEQ)
    mm = length(bb)
    m = mm - meq
    nn = length(index)

    # set bound default values
    if flag == 'l' or flag == 'b':
        xLB = LB[index]
    else:
        xLB = np.array([])

    if flag == 'u' or flag == 'b':
        xUB = UB[index]
    else:
        xUB = np.array([])

    # setup the first LP
    lhs = np.repeat(-np.inf, mm)  # -inf(mm, 1)
    lhs[INDEQ] = bb[INDEQ]
    ff = np.zeros(n)
    ff[index[0]] = 1

    # solve first lower bound
    if flag == 'l' or flag == 'b':
        solution = cvxlpbnd(ff, AA, LB, UB, lhs, bb)
        if solution['status'] != 'optimal':
            print('1st LP lower bound cannot be obtained: either unbounded or infeasible!\n\n')
            raise Exception("Not exist solution optimal!!")
        xx1 = np.array(solution['x']).reshape((n,))
        xLB[0] = ff.T @ xx1

    # solve first upper bound
    if flag == 'u' or flag == 'b':
        solution = cvxlpbnd(-1 * ff, AA, LB, UB, lhs, bb)
        if solution['status'] != 'optimal':
            print('1st LP upper bound cannot be obtained: either unbounded or infeasible!\n\n')
            raise Exception("Not exist solution optimal!!")
        xx2 = np.array(solution['x']).reshape((n,))
        xUB[0] = ff.T @ xx2

    # find the indices of xx1 and xx2 are at its lower or upper bounds, no need to solve those LPs
    if flag == 'b':
        tmp_L = np.minimum(xx1, xx2)
        tmp_U = np.maximum(xx1, xx2)
    elif flag == 'l':
        tmp_L = xx1
    else:
        tmp_U = xx2

    if flag == 'l' or flag == 'b':
        tmp_L = tmp_L[index]
        idx_L = np.where(np.absolute(tmp_L - LB[index]) < 1e-8)
        index_L = np.copy(index)
        index_L[idx_L] = -1

    if flag == 'u' or flag == 'b':
        tmp_U = tmp_U[index]
        idx_U = np.where(np.absolute(tmp_U - UB[index]) < 1e-8)
        index_U = np.copy(index)
        index_U[idx_U] = -1

    # solve upper bounds
    if flag == 'u' or flag == 'b':
        for i in range(1, nn):
            if index_U[i] >= 0:
                ff = np.zeros((n, 1))
                ff[index_U[i]] = 1
                solution = cvxlpbnd(-1 * ff, AA, LB, UB, lhs, bb)
                if solution['status'] != 'optimal':
                    print('%d-th LP upper bound cannot be obtained: either unbounded or infeasible!\n\n', i)
                    raise Exception("Not exist solution optimal!!")
                _xx = np.array(solution['x']).reshape((n,))
                xUB[i] = ff.T @ _xx

    if flag == 'l' or flag == 'b':
        for i in range(1, nn):
            if index_L[i] >= 0:
                ff = np.zeros((n, 1))
                ff[index_L[i]] = 1
                solution = cvxlpbnd(ff, AA, LB, UB, lhs, bb)
                if solution['status'] != 'optimal':
                    print('%d-th LP lower bound cannot be obtained: either unbounded or infeasible!\n\n', i);
                    raise Exception("Not exist solution optimal!!")
                _xx = np.array(solution['x']).reshape((n,))
                xLB[i] = ff.T @ _xx

    return xLB, xUB


def refm(H, f, A, b, Aeq, beq, LB, UB, cons, sstruct):
    # REF1 perform the first reformulation in appendix of the paper
    m = size(A, 1)
    n = size(H, 1)
    meq = size(Aeq, 1)
    idxU = np.where(np.isinf(LB) & np.isfinite(UB))[0]  # @salmuz testing (several cases)
    # Change of variable: new var = UB(idxU) - x(idxU) >=0
    # Original variable has no lower bounds, new variable has no upper bounds
    if idxU.size > 0:
        ub = UB[idxU]
        sstruct['ub2'] = ub
        sstruct['idxU2'] = idxU
        hh = H[idxU, :][:, idxU]
        fidxU = f[idxU]
        f[idxU] = -fidxU
        idxUc = np.setdiff1d(np.arange(n), idxU)
        cons = cons + 0.5 * np.transpose(ub) @ hh @ ub + np.transpose(fidxU) @ ub
        f[idxU] = f(idxU) - hh * ub
        f[idxUc] = f(idxUc) + H(idxUc, idxU) * ub
        H[np.ix_(idxUc, idxU)] = - H[np.ix_(idxUc, idxU)]
        H[np.ix_(idxU, idxUc)] = - H[np.ix_(idxU, idxUc)]
        if m > 0:
            b = b - A[:, idxU] @ ub
            A[:, idxU] = -A[:, idxU]

        if meq > 0:
            beq = beq - Aeq[:, idxU] @ ub
            Aeq[:, idxU] = - Aeq[:, idxU]

        LB[idxU] = 0
        UB[idxU] = np.inf

    # Shift the finite lower bounds to zero
    idxL = np.where(np.isfinite(LB))[0]  # @salmuz, possible bug
    if idxL.size > 0:
        sstruct['idxL3'] = idxL
        sstruct['lb3'] = LB[idxL]
        f, b, beq, LB, UB, cons = shift(H, f, A, b, Aeq, beq, LB, UB, cons, idxL)

    # Scale so that the upper bound is 1
    idxU = np.where(np.isfinite(UB))[0]  # @salmuz, possible bug
    if idxU.size > 0:
        sstruct['idxU4'] = idxU
        sstruct['ub4'] = UB[idxU]
        H, A, Aeq, f, UB = scale(H, f, A, Aeq, UB, idxU)

    # If not all the bounds are finite, then calculate bounds and turn it to [0,1]
    m, _ = A.shape
    meq, _ = Aeq.shape
    mLB = isfinite(LB)
    mUB = isfinite(UB)
    if sum(mLB) < n or sum(mUB) < n:
        AA = np.vstack([A, Aeq])
        bb = np.vstack([b, beq])
        if meq == 0:
            INDEQ = np.array([])
        else:
            INDEQ = np.arange(m + 1, m + meq)

        idxL = np.where(np.isinf(LB))[0]
        idxU = np.where(np.isinf(UB))[0]
        # In the case m + meq = 0, have to have finite bounds on LB and UB
        lp, = LB.shape
        up, = UB.shape
        if m + meq > 0:
            # If AA is empty, then feeding AA and bb to CPLEXINT will result
            # in error, and thus we only solve the following LPs when AA is not empty
            xL, xU = cplex_bnd_solve(AA, bb, INDEQ, LB, UB, idxL, 'l')
            LB[idxL] = xL

            xL, xU, = cplex_bnd_solve(AA, bb, INDEQ, LB, UB, idxU, 'u')
            UB[idxU] = xU
        else:
            if lp * up == 0 or (idxL.size + idxU.size) > 0:
                raise Exception('Both LB and UB must be finite.')
        # track change: 5
        sstruct['idxL5'] = idxL
        sstruct['lb5'] = LB[idxL]
        f, b, beq, LB, UB, cons = shift(H, f, A, b, Aeq, beq, LB, UB, cons, idxL)

        sstruct['idxU5'] = idxU
        sstruct['ub5'] = UB[idxU]
        H, f, A, Aeq, UB = scale(H, f, A, Aeq, UB, idxU)

        # We have calculated all the bounds and scaled them to [0,1]. But to have less complementarities,
        # we will pretend that we did not find bounds for the original unbounded values.
        LB[idxL] = -np.inf
        UB[idxU] = np.inf

    return H, f, A, b, Aeq, beq, LB, UB, cons, sstruct


def boundall(H, f, A, b, Aeq, beq, LB, UB, idxL, idxB):
    """
    BOUNDALL calculates the bounds for all the variables
    # -------------------------------------------------
    # Calculate bounds for all vars: ( x, s, lambda, y, wB, zB, zL, rB )
    # Solve the LP with variable X introduced to bound the dual vars
    #
    #      H(:)'*X(:) + f'*x + b'*lambda + beq'* y + rB= 0
    #      X_{i,j} <= x_j, X_{i,j} <= x_i
    #      X_{i,j} >= x_i + x_j - 1
    #
    # Suppose [m,n] = size(A). Dimension of variables:
    #  - x: n
    #  - lambda, s: m
    #  - y: meq
    #  - rB,zB,wB: lenB
    #  - zL: lenL
    #  - X: .5*n*(n+1)
    #
    # Order of vars: ( x, X, s, wB, lambda, y, zL, zB, rB )
    """
    Aeq0 = Aeq.copy()
    beq0 = beq.copy()
    n, _ = H.shape
    m = np.size(A, 0)
    meq = np.size(Aeq, 0)
    lenL = idxL.size
    lenB = idxB.size
    nn = n + int(.5 * n * (n + 1)) + 2 * m + meq + lenL + 3 * lenB

    # BEGIN: prepare the data required by CPLEXINT
    dH = np.diag(H)
    H1 = 2 * np.tril(H, -1)
    H1 = H1 + np.diag(dH)
    HH = np.array([])
    for j in np.arange(n):
        HH = np.concatenate([HH, H1[j:, j]])

    # Order of vars: ( x, X, s, wB, lambda, y, zL, zB, rB )
    tmp1 = np.zeros((n, lenL))
    for i in np.arange(lenL):
        k = idxL[i]
        tmp1[k, i] = -1

    tmp2 = np.zeros((n, lenB))
    for i in np.arange(lenB):
        k = idxB[i]
        tmp2[k, i] = -1

    tmp3 = -tmp2
    tmp4 = np.zeros((lenB, n))
    for i in np.arange(lenB):
        k = idxB[i]
        tmp4[k, i] = 1

    # -------------------------------------------------
    #  The KKT system now is:
    #
    #  (1)    Aeq * x = beq
    #  (2)    A * x + s = b
    #  (3)    H * x + f + A'*lambda + Aeq' * y - zL - zB + rB = 0
    #  (4)    H \dot X + f' * x + b'*lambda + beq'*y + e' * rB = 0
    #  (5)    xB + wB = 1
    # -------------------------------------------------
    if not isempty(Aeq):  # if ~isempty(Aeq)
        Aeq = np.concatenate([Aeq, np.zeros((meq, nn - n))], axis=1)

    # Aeq = [ Aeq ;
    #    A    zeros(m,.5*n*(n+1)) eye(m) zeros(m,nn-n-m-.5*n*(n+1)) ;              % (2)
    #    H    zeros(n,.5*n*(n+1)+m+lenB) A' Aeq0' tmp1 tmp2 tmp3 ;                 % (3)
    #    f'  HH'  zeros(1,m+lenB) b' beq0' zeros(1,lenL+lenB) ones(1,lenB) ;       % (4)
    #    tmp4 zeros(lenB,.5*n*(n+1)+m) eye(lenB) zeros(lenB,m+meq+lenL+2*lenB)  ]; % (5)

    equ2 = np.array([])
    equ3 = np.array([])
    equ4 = np.array([])
    equ5 = np.array([])

    # % row2 of new Aeq: [ A   zeros(m,.5*n*(n+1)) eye(m) zeros(m,nn-n-m-.5*n*(n+1)) ]
    if not np.all(A == 0):
        equ2 = np.concatenate([A, np.zeros((m, int(.5 * n * (n + 1)))),
                               np.eye(m), np.zeros((m, int(nn - n - m - .5 * n * (n + 1))))], axis=1)

    # % row 3 of new Aeq : [ H    zeros(n,.5*n*(n+1)+m+lenB) A' Aeq0' tmp1 tmp2 tmp3 ]
    equ3 = np.concatenate([H, np.zeros((n, int(.5 * n * (n + 1)) + m + lenB))], axis=1)
    if not np.all(A == 0):
        equ3 = np.concatenate([equ3, A.T], axis=1)
    if not np.all(Aeq0 == 0):
        equ3 = np.concatenate([equ3, Aeq0.T], axis=1)
    if not np.all(tmp1 == 0):
        equ3 = np.concatenate([equ3, tmp1], axis=1)
    if not np.all(tmp2 == 0):
        equ3 = np.concatenate([equ3, tmp2], axis=1)
    if not np.all(tmp3 == 0):
        equ3 = np.concatenate([equ3, tmp3], axis=1)

    # % row 4 of new Aeq: [f'  HH'  zeros(1,m+lenB) b' beq0' zeros(1,lenL+lenB) ones(1,lenB) ]
    equ4 = np.concatenate([f.T, HH.T])
    if m + lenB > 0:
        equ4 = np.concatenate([equ4, np.zeros(m + lenB)])
    if not np.all(b == 0):
        equ4 = np.concatenate([equ4, b.T])
    if not np.all(beq0 == 0):
        equ4 = np.concatenate([equ4, beq0.T])
    if lenL + lenB > 0:
        equ4 = np.concatenate([equ4, np.zeros(lenL + lenB)])
    if lenB > 0:
        equ4 = np.concatenate([equ4, np.ones(lenB)])

    # row 5 of new Aeq: [ tmp4 zeros(lenB,.5*n*(n+1)+m) eye(lenB) zeros(lenB,m+meq+lenL+2*lenB) ]
    if not np.all(tmp4):
        equ5 = tmp4.copy()

    if lenB > 0:
        equ5 = np.concatenate([equ5, np.zeros((lenB, int(.5 * n * (n + 1)) + m)),
                               np.eye(lenB), np.zeros((lenB, m + meq + lenL + 2 * lenB))], axis=1)
    if np.all(Aeq == 0):
        tmpp = np.array([size(equ2, 1), size(equ3, 1), size(equ4, 1), size(equ5, 1)])
        Aeq = np.zeros((0, max(tmpp)))
    if not np.all(equ2 == 0):
        Aeq = np.concatenate([Aeq, equ2])
    if not np.all(equ3 == 0):
        Aeq = np.concatenate([Aeq, equ3])
    if not np.all(equ4 == 0):
        Aeq = np.concatenate([Aeq, [equ4]])
    if not np.all(equ5 == 0):
        Aeq = np.concatenate([Aeq, equ5])

    beq = np.concatenate([beq, b, -f, [0], np.ones(lenB)])
    INDEQ = np.arange(np.size(beq, 0)).T

    # This process it for Aeq and A constraints (not our case) @salmuz
    # ---------------------------------------------------------------
    #  Start to prepare the part of data modeling implied bounds on X,
    #  including three parts.
    # ---------------------------------------------------------------

    # Part I & II: X_{i,j} <= x_j, X_{i,j} <= x_i
    len = int(n + .5 * n * (n + 1))
    qq = np.ones((n, n))
    qq = np.tril(qq)
    # matlab index recovery by column, numpy by row
    _A = np.transpose(np.nonzero(qq))
    _A = _A[np.lexsort((_A[:, 0], _A[:, 1]))]
    I, J = _A[:, 0], _A[:, 1]
    lenI = I.size

    block = np.zeros((1000, 3))
    range_t = 1000
    k, rowid = 0, 0
    for i in np.arange(lenI):
        ii = I[i]
        jj = J[i]
        block[k, :] = np.array([rowid, ii, -1])
        k += 1
        if k + 5 > range_t:
            block = np.concatenate([block, np.zeros((1000, 3))])
            range_t = range_t + 1000
        block[k, :] = np.array([rowid, n + i, 1])
        k += 1
        rowid += 1
        if ii != jj:
            block[k, :] = np.array([rowid, n + i, 1])
            k += 1
            block[k, :] = np.array([rowid, jj, -1])
            k += 1
            rowid += 1

    # %% Part III:  X_{i,j} >= x_i + x_j - 1
    for i in np.arange(lenI):
        ii = I[i]
        jj = J[i]
        block[k, :] = np.array([rowid, ii, 1])
        k += 1
        if k + 5 > range_t:
            block = np.concatenate((block, np.zeros((1000, 3))))
            range_t = range_t + 1000
        if ii != jj:
            block[k, :] = np.array([rowid, jj, 1])
            k += 1
        else:
            block[k - 1, 2] = 2
        block[k, :] = np.array([rowid, n + i, -1])
        k += 1
        rowid += 1

    block = block[0:k, :]
    # @salmuz verify index because matlab starts 1 and python 0
    AA = csr_matrix((block[:, 2], (block[:, 0], block[:, 1])), (n * n + lenI, nn))
    bb = np.concatenate([np.zeros(n ** 2), np.ones(lenI)])
    # Order of vars: ( x, X, s, wB, lambda, y, zL, zB, rB )
    # s >=0, lambda >=0, y free, wB>=0, xL>=0, 0<=xB
    # zL>=0, zB>=0,rB>=0
    # s .* lambda = 0, xL.*zL=0, xB.*zB=0, wB.*rB=0, zB.*rB = 0
    LB = np.zeros(nn)
    tmp = int(len + 2 * m + lenB)
    LB[tmp: tmp + meq] = -np.inf
    UB = np.inf * np.ones(nn)
    # In previous transformation, x is bounded above by 1, and so
    # does X. although we pretend that they are not there when
    # doing complementarity

    UB[0: len] = 1
    UB[len + m: len + m + lenB] = 1
    # END: prepare the data required by CPLEXINT
    # Recalculate bounds for x after adding KKT
    L = np.zeros(nn)
    U = np.ones(nn)
    AA = np.concatenate([Aeq, AA.todense()])  # @salmuz bug
    bb = np.concatenate([beq, bb])
    # total = n + nn - len
    # calculate bounds for x
    index0 = np.arange(n)
    xL, xU = cplex_bnd_solve(AA, bb, INDEQ, LB, UB, index0)
    L[index0] = xL
    U[index0] = xU

    # Recalcualte bounds for the rest vars except X
    index0 = np.arange(len, nn)  # matlab len+1:nn
    xL, xU = cplex_bnd_solve(AA, bb, INDEQ, LB, UB, index0)
    L[index0] = xL
    U[index0] = xU

    # --------------------------------------------
    #  Formulate the KKT system, turn the problem
    #  into one with A x == b, x >= 0 constraints
    #  and x's upper bounds is one is implied.
    #
    #  The order of the variables are:
    #  ( x, s, wB, lambda, y, zL, zB, rB)
    # --------------------------------------------

    # Formulating new A
    L = np.hstack([L[0:n], L[len:]])
    U = np.hstack([U[0:n], U[len:]])

    return L, U, tmp1, tmp2, tmp3, tmp4


def standardform(H, f, A, b, Aeq, beq, LB, UB, cons, tol):
    E = np.array([])
    sstruct = dict({'flag': 0, 'obj': -np.inf, 'fx1': np.array([]), 'fxval1': np.array([]), 'ub2': np.array([]),
                    'idxU2': np.array([]), 'lb3': np.array([]), 'idxL3': np.array([]), 'ub4': np.array([]),
                    'idxU4': np.array([]), 'idxU5': np.array([]), 'ub5': np.array([]), 'idxL5': np.array([]),
                    'lb5': np.array([]), 'lb6': np.array([]), 'ub6': np.array([]), 'xx': np.array([])})

    n, _ = H.shape
    FX = np.where(abs(LB - UB) < tol)[0]  # @salmuz, possible bug
    if FX.size > 0:
        sstruct['fx1'] = FX
        sstruct['fxval1'] = LB[FX]
        # FXc == complement of FX or the components of x that are not fixed
        FXc = np.setdiff1d(np.arange(n), FX)
        # Calculate the constants in objective after removing the fixed vars
        cons = cons + 0.5 * np.transpose(LB[FX]) @ H[FX, :][:, FX] @ LB[FX] + np.transpose(f[FX]) @ LB[FX]

        # Update data after removing fixed components
        f = f[FXc] + H[FXc, :][:, FX] @ LB[FX]
        H = H[FXc, :][:, FX]

        # Update the bounds for the non-fixed components of x
        LB = LB[FXc]
        UB = UB[FXc]

    tmp_m, tmp_n = size(Aeq, None)
    tmp_m1 = np.linalg.matrix_rank(Aeq) if tmp_n > 0 else 0
    if tmp_m1 < min(tmp_m, tmp_n):
        tmp = np.column_stack([Aeq, beq]).T
        _, rowidx = Matrix(tmp).rref()
        rowidx = np.array(rowidx)
        Aeq = Aeq[rowidx, :]
        beq = beq[rowidx]

    H, f, A, b, Aeq, beq, LB, UB, cons, sstruct = refm(H, f, A, b, Aeq, beq, LB, UB, cons, sstruct)

    # ----------------------------------------------
    #  Now problem becomes:
    #
    #    min  .5*x*H*x + f*x + cons
    #    s.t.   A x <=  b       ( lamabda >=0 )
    #           Aeq x = beq     ( y free )
    #           0 <= xL         ( zL >=0 )
    #           0 <= xB <= 1    ( zB>=0, rB >=0)
    #
    #  Now we are to formulate KKT system, and
    #  calculate bounds on all vars.
    # -----------------------------------------------
    n, _ = H.shape
    m, _ = A.shape
    meq, _ = Aeq.shape
    i1 = np.isfinite(LB)
    i2 = np.isfinite(UB)

    # now the meaning of idxL and idxU has changed.
    # idxL = LB finite + UB infinite
    # idxU = both LB & UB finite

    idxL = np.where(i1 & ~i2)[0]
    idxB = np.where(i1 & i2)[0]
    lenL = idxL.size
    lenB = idxB.size

    # -----------------------------
    # Calculate bounds for all vars
    # -----------------------------
    L, U, tmp1, tmp2, tmp3, tmp4 = boundall(H, f, A, b, Aeq, beq, LB, UB, idxL, idxB)

    # -------------------------------------------------
    # Prep equality constraints: A * xx = b
    # (1)    H * x + A'*lambda + Aeq' * y - zL - zB + rB = -f
    # (2)    A * x + s = b
    # (3)    Aeq * x = beq
    # (4)    xB + wB = 1
    # -------------------------------------------------

    if np.linalg.norm(L - U) < tol:
        sstruct["flag"] = 1
        xLB = L[0:n]
        sstruct["xx"] = xLB
        sstruct["obj"] = .5 * (xLB.T @ H @ xLB) + f.T @ xLB + cons
        return H, f, A, b, E, cons, L, U, sstruct

    idxx = np.where(np.absolute(U - L) <= tol)
    L[idxx] = U[idxx]

    nn = n + 2 * m + meq + lenL + 3 * lenB
    H1 = np.copy(H)
    A1 = np.copy(A)
    A2 = np.copy(A)
    Aeq1 = np.copy(Aeq)
    Aeq2 = np.copy(Aeq)

    xLB = L[0:n]
    xUB = U[0:n]
    Dx = np.diag(xUB - xLB)

    sLB = L[n:n + m]
    sUB = U[n:n + m]

    n0 = n + m
    wLB = L[n0:n0 + lenB]
    wUB = U[n0:n0 + lenB]

    n0 = n0 + lenB
    lambdaLB = L[n0:n0 + m]
    lambdaUB = U[n0:n0 + m]

    n0 = n0 + m
    yLB = L[n0:n0 + meq]
    yUB = U[n0:n0 + meq]

    n0 = n0 + meq
    zLLB = L[n0:n0 + lenL]
    zLUB = U[n0:n0 + lenL]

    n0 = n0 + lenL
    zBLB = L[n0:n0 + lenB]
    zBUB = U[n0:n0 + lenB]

    n0 = n0 + lenB
    rBLB = L[n0:]
    rBUB = U[n0:]

    # -----------------
    # Right-hand size b
    # -----------------
    #
    r1 = -1 * f - H @ xLB
    if not isempty(tmp1):
        r1 = r1 - tmp1 @ zLLB

    if not isempty(tmp2):
        r1 = r1 - tmp2 @ zBLB

    if not isempty(tmp3):
        r1 = r1 - tmp3 @ rBLB

    if not isempty(A1):
        r1 = r1 - A1.T @ lambdaLB

    if not isempty(Aeq1):
        r1 = r1 - Aeq1.T @ yLB

    if not isempty(A):
        r2 = b - A @ xLB - sLB
    else:
        r2 = np.array([])

    if not isempty(Aeq2):
        r3 = beq - Aeq2 @ xLB
    else:
        r3 = np.array([])

    r4 = np.ones(lenB) - xLB[idxB] - wLB
    b = np.hstack([r1, r2, r3, r4])

    # -----------------
    # Left-hand side A
    # -----------------
    tmp1 = tmp1 @ np.diag(zLUB - zLLB)
    tmp2 = tmp2 @ np.diag(zBUB - zBLB)
    tmp3 = tmp3 @ np.diag(rBUB - rBLB)
    row1 = np.hstack([H @ Dx, np.zeros((n, m + lenB))])

    if not isempty(A1):
        row1 = np.hstack([row1, A1.T @ np.diag(lambdaUB - lambdaLB)])

    if not isempty(Aeq1):
        row1 = np.hstack([row1, Aeq1.T @ np.diag(yUB - yLB)])

    if not isempty(A2):
        row2 = A2 @ Dx
    else:
        row2 = np.array([])

    if not isempty(Aeq2):
        row3 = Aeq2 @ Dx
    else:
        row3 = []

    if not isempty(tmp4):
        row4 = tmp4 @ Dx
    else:
        row4 = []

    # A = [ H*Dx zeros(n,m+lenB) A1'*diag(lambdaUB-lambdaLB) Aeq1'*diag(yUB-yLB) tmp1 tmp2 tmp3;
    # A = [ row1 tmp1 tmp2 tmp3;
    #       row2 diag(sUB-sLB) zeros(m,nn-n-m);
    #       row3 zeros(meq,nn-n);
    #       row4 zeros(lenB,m) diag(wUB-wLB) zeros(lenB, nn-n-m-lenB)];
    #
    # row1 of A
    if not isempty(tmp1):
        row1 = np.hstack([row1, tmp1])

    if not isempty(tmp2):
        row1 = np.hstack([row1, tmp2])

    if not isempty(tmp3):
        row1 = np.hstack([row1, tmp3])

    # row2 of A
    if not isempty(sUB):
        row2 = np.hstack([row2, np.diag(sUB - sLB)])

    if m * (nn - n - m) > 0:
        row2 = np.hstack([row2, np.zeros((m, nn - n - m))])

    # row3 of A
    if meq * (nn - n) > 0:
        row3 = np.hstack([row3, np.zeros((meq, nn - n))])

    # row4 of A
    if lenB * m > 0:
        row4 = np.hstack([row4, np.zeros((lenB, m))])

    if not isempty(wUB):
        row4 = np.hstack([row4, np.diag(wUB - wLB)])

    if lenB * (nn - n - m - lenB) > 0:
        row4 = np.hstack([row4, np.zeros((lenB, nn - n - m - lenB))])

    A = np.array([])
    if not isempty(row1):
        A = row1

    if not isempty(row2):
        A = np.vstack([A, row2])

    if not isempty(row3):
        A = np.vstack([A, row3])

    if not isempty(row4):
        A = np.vstack([A, row4])

    # ------------------------------
    # Shift and scale x affects objs
    # ------------------------------
    cons = cons + 0.5 * (xLB.T @ H @ xLB) + f.T @ xLB;
    f = f + H @ xLB

    H = np.diag(xUB - xLB) @ H @ np.diag(xUB - xLB)
    f = (xUB - xLB) * f
    n0 = nn - n
    f = np.hstack([f, np.zeros(n0)])
    H = np.vstack([np.hstack([H, np.zeros((n, n0))]),
                   np.hstack([np.zeros((n0, n)), np.zeros((n0, n0))])])

    # track change: 6
    sstruct['lb6'] = xLB
    sstruct['ub6'] = xUB

    L = np.zeros(nn)
    U = np.ones(nn)
    # ------------------------------
    # Prep complementarity
    # ------------------------------
    E = np.zeros((nn, nn))

    # s .* lambda = 0
    n0 = n + m + lenB
    for i in range(m):
        E[n + i, n0 + i] = 1

    # xL .* zL = 0
    n0 = n0 + m + meq
    for k in range(lenL):
        i = idxL[k]
        E[i, n0 + k] = 1

    # xB .* zB = 0
    n0 = nn - 2 * lenB
    for k in range(lenB):
        i = idxB[k]
        E[i, n0 + k] = 1

    # wB .* rB = 0
    n00 = n + m
    n0 = nn - lenB
    for k in range(lenB):
        E[n00 + k, n0 + k] = 1

    # zB .* rB = 0
    n00 = nn - 2 * lenB
    for k in range(lenB):
        E[n00 + k, n0 + k] = 1

    E = E + E.T
    sstruct['cmp1'] = np.concatenate([np.arange(n, m), idxL, idxB,
                                      np.arange(n + m, lenB + m + n)], axis=0)
    sstruct['cmp2'] = np.concatenate([np.arange(n + m + lenB, n + 2 * m + lenB),
                                      np.arange(nn - 2 * lenB - lenL, nn - 2 * lenB),
                                      np.arange(nn - 2 * lenB, nn - lenB),
                                      np.arange(nn - lenB, nn)], axis=0)
    sstruct['lenB'] = lenB
    sstruct['m'] = m
    sstruct['n'] = n
    sstruct['lenL'] = lenL

    return H, f, A, b, E, cons, L, U, sstruct
