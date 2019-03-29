from cvxopt import solvers, matrix
import numpy as np
from numpy import linalg as la
from numpy import inf


def scale(H, f, UB, idxU):
    # SCALE scales the part of UB indexed by 'idxU' to 1, assuming that LB(idxU)==0
    n, _ = H.shape
    if idxU.size > 0:
        idxUc = np.setdiff1d(np.arange(n), idxU)
        ub = UB[idxU]
        H[np.ix_(idxU, idxU)] = np.diag(ub) @ H[np.ix_(idxU, idxU)] @ np.diag(ub)
        H[np.ix_(idxUc, idxU)] = H[np.ix_(idxUc, idxU)] @ np.diag(ub)
        H[np.ix_(idxU, idxUc)] = np.diag(ub) @ H[np.ix_(idxU, idxUc)]
        f[idxU] = np.multiply(ub, f[idxU])
        UB[idxU] = 1

    return H, f, UB


def shift(H, f, LB, UB, cons, idxL):
    # SHIFT shifts L(idxL) to zero.
    n, _ = H.shape
    if idxL.size > 0:
        idxLc = np.setdiff1d(np.arange(n), idxL)
        lb = LB[idxL]

        cons = cons + 0.5 * np.transpose(lb) @ H[np.ix_(idxL, idxL)] @ lb + np.transpose(f[idxL]) @ lb
        f[idxL] = f[idxL] + H[np.ix_(idxL, idxL)] @ lb

        # if idxLc is empty, then dimension will not agree on the following equations
        # so add a check here
        if idxLc.size > 0:
            f[idxLc] = f[idxLc] + H[np.ix_(idxL, idxL)] @ lb
        UB[idxL] = UB[idxL] - lb
        LB[idxL] = 0

    return f, LB, UB, cons


def checkinputs(H, LB, UB):
    n, p = H.shape

    if LB.ndim > 1 or UB.ndim > 1:
        raise Exception('Both LB and UB must be column vectors.')

    lp, = LB.shape
    up, = UB.shape
    if lp != up:
        raise Exception('Both LB and UB must have dimension.')

    if n * p == 0 or la.norm(H, ord='fro') == 0:
        raise Exception('H should be nonzero.')

    if np.any(UB < LB):
        raise Exception('Need UB >= LB.')


def refm(H, f, LB, UB, cons, sstruct):
    # REF1 perform the first reformulation in appendix of the paper
    n, _ = H.shape
    idxU = np.where(np.isinf(LB) & np.isfinite(UB))[0]
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
        LB[idxU] = 0
        UB[idxU] = inf

    # Shift the finite lower bounds to zero
    idxL = np.where(np.isfinite(LB))[0]
    if idxL.size > 0:
        sstruct['idxL3'] = idxL
        sstruct['lb3'] = LB[idxL]
        f, LB, UB, cons = shift(H, f, LB, UB, cons, idxL)

    # Scale so that the upper bound is 1
    idxU = np.where(np.isfinite(UB))[0]
    if idxU.size > 0:
        sstruct['idxU4'] = idxU
        sstruct['ub4'] = UB[idxU]
        H, f, UB = scale(H, f, UB, idxU)

    # If not all the bounds are finite, then calculate bounds and turn it to [0,1]
    mLB = np.where(np.isfinite(LB))[0]
    mUB = np.where(np.isfinite(UB))[0]
    if mLB.size < n or mUB.size < n:
        idxL = np.where(np.isinf(LB))[0]
        idxU = np.where(np.isinf(UB))[0]
        # In the case m + meq = 0, have to have finite bounds on LB and UB
        lp, = LB.shape
        up, = UB.shape
        if lp * up == 0 or (idxL.size + idxU.size) > 0:
            raise Exception('Both LB and UB must be finite.')
        # track change: 5
        sstruct['idxL5'] = idxL
        sstruct['lb5'] = LB[idxL]
        f, LB, UB, cons = shift(H, f, LB, UB, cons, idxL)

        sstruct['idxU5'] = idxU
        sstruct['ub5'] = UB[idxU]
        H, f, UB = scale(H, f, UB, idxU)

        # We have calculated all the bounds and scaled them to [0,1]. But to have less complementarities,
        # we will pretend that we did not find bounds for the original unbounded values.
        LB[idxL] = -inf
        UB[idxU] = inf

    return H, f, LB, UB, cons, sstruct


def boundall(H, f, LB, UB, idxL, idxB):
    # BOUNDALL calculates the bounds for all the variables
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
    # -------------------------------------------------
    n, _ = H.shape
    lenL = idxL.size
    lenB = idxB.size
    nn = n + .5 * n * (n + 1) + lenL + 3 * lenB

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
        tmp4[k, i] = -1
        
    # -------------------------------------------------
    #  The KKT system now is:
    # 
    #  (1)    Aeq * x = beq
    #  (2)    A * x + s = b
    #  (3)    H * x + f + A'*lambda + Aeq' * y - zL - zB + rB = 0
    #  (4)    H \dot X + f' * x + b'*lambda + beq'*y + e' * rB = 0
    #  (5)    xB + wB = 1
    # -------------------------------------------------
    # if ~isempty(Aeq)
    #   Aeq = [ Aeq zeros(meq,nn-n)];  % (1) 
    # end

    print(tmp1, tmp2, tmp3, tmp4)


def standardform(H, f, LB, UB, cons, tol):
    sstruct = dict({'flag': 0, 'obj': -inf, 'fx1': [], 'fxval1': [], 'ub2': [], 'idxU2': [],
                    'lb3': [], 'idxL3': [], 'ub4': [], 'idxU4': [], 'idxU5': [], 'ub5': [],
                    'idxL5': [], 'lb5': [], 'lb6': [], 'ub6': [], 'xx': []})

    n, _ = H.shape
    FX = np.where(abs(LB - UB) < tol)[0]
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

    H, f, LB, UB, cons, sstruct = refm(H, f, LB, UB, cons, sstruct)

    # ----------------------------------------------
    #  Now problem becomes:
    #
    #    min  .5*x*H*x + f*x + cons
    #    s.t.   A x <=  b              ( lamabda >=0 )
    #           Aeq x = beq            ( y free )
    #           0 <= xL                ( zL >=0 )
    #           0 <= xB <= 1           ( zB>=0, rB >=0)
    #
    #  Now we are to formulate KKT system, and
    #  calculate bounds on all vars.
    # -----------------------------------------------
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
    boundall(H, f, LB, UB, idxL, idxB)


def quadprogbb(H, f, LB, UB, options=None):
    """
        This method use a solver implemented in matlab in
            https://github.com/sburer/QuadProgBB
            Authors: Samuel Burer
        [x,fval,time,stat] = quadprogbb(H,f,A,b,Aeq,beq,LB,UB,options)

        QUADPROGBB globally solves the following nonconvex quadratic
        programming problem:

                min      1/2*x'*H*x + f'*x
                s.t.       LB <= x <= UB

    """
    options = dict({"cons": 0, 'tol': 1e-8})

    assert type(H).__module__ == np.__name__, 'H must be a numpy array type.'
    n, m = H.shape
    assert n == m, 'H must be a square matrix.'

    H = .5 * (H + H.T)
    p, = f.shape
    assert n == p, 'Dimensions of H and f are not consistent!'

    checkinputs(H, LB, UB)

    # ======================================================
    #  Initialize the struct for statistics:
    #
    #  time_pre: time spent on preprocessing
    #  time_LP:  time spent on calculating bounds in preprocessing
    #  time_BB:  time spent on B&B
    #  nodes: total nodes_solved
    #  status: 0) solution found; 1) infeasible; 2) max_time exceeded
    # ======================================================
    stat = dict({'time_pre': 0, 'time_LP': 0, 'time_BB': 0, 'nodes': 0, 'status': []})
    I = matrix(0.0, (p, p))
    I[::p + 1] = 1
    standardform(H, f, LB, UB, options['cons'], options['tol'])


"""


stat = struct('time_pre',0,'time_LP',0,'time_BB',0,'nodes',0,'status',[]);


 check feasibility 

if (~isempty(A)) || (~isempty(Aeq))
  
  cplexopts = cplexoptimset('Display','off');
  [x,fval,exitflag,output] = cplexlp(zeros(n,1),A,b,Aeq,beq,LB,UB,[],cplexopts);
  
  if output.cplexstatus > 1

    fprintf('\n\nFail assumption check:\n\n');
    fprintf('CPLEX status of solving the feasibility problem: %s', output.cplexstatusstring);
    x = []; fval = []; time = 0;
    stat.status = 'inf_or_unb';
    return

  end

end
"""
