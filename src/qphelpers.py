import numpy as np
from .pymatlab import isempty, length


def scale(H, f, A, Aeq, UB, idxU):
    # SCALE scales the part of UB indexed by 'idxU' to 1, assuming that LB(idxU)==0
    m, _ = A.shape
    n, _ = H.shape
    meq, _ = Aeq.shape
    if idxU.size > 0:
        idxUc = np.setdiff1d(np.arange(n), idxU)
        ub = UB[idxU]
        H[np.ix_(idxU, idxU)] = np.diag(ub) @ H[np.ix_(idxU, idxU)] @ np.diag(ub)
        H[np.ix_(idxUc, idxU)] = H[np.ix_(idxUc, idxU)] @ np.diag(ub)
        H[np.ix_(idxU, idxUc)] = np.diag(ub) @ H[np.ix_(idxU, idxUc)]
        f[idxU] = np.multiply(ub, f[idxU])

        if m > 0:
            A[:, idxU] = A[:, idxU] @ np.diag(ub)
        if meq > 0:
            Aeq[:, idxU] = Aeq[:, idxU] @ np.diag(ub)

        UB[idxU] = 1

    return H, A, Aeq, f, UB


def shift(H, f, A, b, Aeq, beq, LB, UB, cons, idxL):
    # SHIFT shifts L(idxL) to zero.
    m, _ = A.shape
    n, _ = H.shape
    meq, _ = Aeq.shape
    if idxL.size > 0:
        idxLc = np.setdiff1d(np.arange(n), idxL)
        lb = LB[idxL]

        cons = cons + 0.5 * np.transpose(lb) @ H[np.ix_(idxL, idxL)] @ lb + np.transpose(f[idxL]) @ lb
        f[idxL] = f[idxL] + H[np.ix_(idxL, idxL)] @ lb

        # if idxLc is empty, then dimension will not agree on the following equations
        # so add a check here
        if idxLc.size > 0:
            f[idxLc] = f[idxLc] + H[np.ix_(idxL, idxL)] @ lb

        if m > 0:
            b = b - A[:, idxL] @ lb
        if meq > 0:
            beq = beq - Aeq[:, idxL] @ lb

        UB[idxL] = UB[idxL] - lb
        LB[idxL] = 0

    return f, b, beq, LB, UB, cons


def checkinputs(H, LB, UB):
    n, p = H.shape

    if LB.ndim > 1 or UB.ndim > 1:
        raise Exception('Both LB and UB must be column vectors.')

    lp, = LB.shape
    up, = UB.shape
    if lp != up:
        raise Exception('Both LB and UB must have dimension.')

    if n * p == 0 or np.linalg.norm(H, ord='fro') == 0:
        raise Exception('H should be nonzero.')

    if np.any(UB < LB):
        raise Exception('Need UB >= LB.')


def getsol(x, sstruct):
    # it reverse the transformation made, and return the solution to the original problem
    if not isempty(x):
        x = x[0:sstruct["n"]]
    else:
        x = sstruct["xx"]
        if isempty(x):
            x = x[0:sstruct["n"]]

    if sstruct.flag == 0:
        x = (sstruct["ub6"] - sstruct["lb6"]) * x + sstruct["lb6"]

    if not isempty(sstruct["idxU5"]):
        x[sstruct["idxU5"]] = sstruct["ub5"] * x[sstruct["idxU5"]]

    if not isempty(sstruct["idxL5"]):
        x[sstruct["idxL5"]] = x[sstruct["idxL5"]] + sstruct["lb5"]

    if not isempty(sstruct["idxU4"]):
        x[sstruct["idxU4"]] = sstruct["ub4"] * x[sstruct["idxU4"]]

    if not isempty(sstruct["idxL3"]):
        x[sstruct["idxL3"]] = x[sstruct["idxL3"]] + sstruct["lb3"]

    if not isempty(sstruct["idxU2"]):
        x[sstruct["idxU2"]] = sstruct["ub2"] - x[sstruct["idxU2"]]

    if not isempty(sstruct["fx1"]):
        lenn = length(sstruct["fx1"]) + length(x)
        y = np.zeros(lenn)
        y[sstruct["fx1"]] = sstruct["fxval1"]
        y[np.setdiff1d(np.arange(lenn), sstruct["fx1"])] = x
        x = y

    return x
