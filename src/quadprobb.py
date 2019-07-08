from .pymatlab import size, length
from .qphelpers import checkinputs, getsol
from .qpreformulation import standardform
import sys
import numpy as np
from time import time
from cvxopt import solvers, matrix

solvers.options['show_progress'] = True
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


def quadprog(H, q, Aeq, beq, lower, upper):
    d = size(lower)
    ell_lower = matrix(lower, (d, 1))
    ell_upper = matrix(upper, (d, 1))
    P = matrix(H)
    q = matrix(q)
    I = matrix(0.0, (d, d))
    I[::d + 1] = 1
    G = matrix([I, -I])
    h = matrix([ell_upper, -ell_lower])
    Aeq = matrix(Aeq)
    beq = matrix(beq)
    print(np.linalg.matrix_rank(Aeq), np.linalg.matrix_rank(H))
    return solvers.qp(P=P, q=q, G=G, h=h, A=Aeq, b=beq)


def quadprogbb(H, f, A=np.empty([0, 0]), b=np.array([]), Aeq=np.empty([0, 0]),
               beq=np.array([]), LB=np.array([]), UB=np.array([]), options=None):
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
    # ========================
    # handle missing arguments
    # ========================
    # set default options

    defaultopt = dict({
        'max_time': 86400,
        'tol': 1e-8,
        'fathom_tol': 1e-6,
        'max_iter': 1000,
        'use_quadprog': True,
        'verbosity': True,
        'constant': 0,
        'use_single_processor': 1,
        'checkpt': 0,
        'checkfile': ''
    })

    options = {**defaultopt}
    assert type(H).__module__ == np.__name__, 'H must be a numpy array type.'
    n, m = H.shape
    assert n == m, 'H must be a square matrix.'

    H = .5 * (H + H.T)
    p, = f.shape
    assert n == p, 'Dimensions of H and f are not consistent!'

    cons = options["constant"]

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

    # %% @salmuz not implemented yet
    # %% check feasibility
    # if (~isempty(A)) || (~isempty(Aeq))
    #
    #   cplexopts = cplexoptimset('Display','off');
    #   [x,fval,exitflag,output] = cplexlp(zeros(n,1),A,b,Aeq,beq,LB,UB,[],cplexopts);
    #   fprintf('\n==%f====\n', x);
    #   if output.cplexstatus > 1
    #
    #     fprintf('\n\nFail assumption check:\n\n');
    #     fprintf('CPLEX status of solving the feasibility problem: %s', output.cplexstatusstring);
    #     x = []; fval = []; time = 0;
    #     stat.status = 'inf_or_unb';
    #     return
    #   end
    # end

    # =========================================
    # Turn problem into standard form
    #
    #  min   0.5*x'*H*x + f'*x
    #  s.t.  A*x = b, x >= 0
    #          [x*x']_E == 0
    # Bounds x <= 1 are valid
    # =========================================
    H, f, A, b, E, cons, L, U, sstruct = standardform(H, f, A, b, Aeq, beq, LB, UB, cons, options['tol'])
    print('\n****  Pre-Processing is complete, time = %.2f  ****\n')

    # stat.time_pre = toc
    # stat.time_LP = timeLP

    if sstruct["flag"] == 1:
        fval = sstruct.obj
        x = getsol([], sstruct)
        toc = time()
        nodes_solved = 0
        print('=========================== Node 0 ============================\n\n')
        print('FINAL STATUS 1: optimal value = %.8e\n', fval)
        print('FINAL STATUS 2: (created,solved,pruned,infeas,left) = (%d,%d,%d,%d,%d)\n', 0, 0, 0, 0, 0)
        print('FINAL STATUS 3: solved = fully + fathomed + poorly : %d = %d + %d + %d\n', 0, 0, 0, 0)
        print('FINAL STATUS 4: time = %d\n', toc)
        stat["status"] = 'opt_soln'
        return x, fval

    cmp1 = sstruct["cmp1"]
    cmp2 = sstruct["cmp2"]
    lenB = sstruct["lenB"]
    lenL = sstruct["lenL"]
    m = sstruct["m"]
    n = sstruct["n"]
    m0 = length(cmp1)

    findFx = np.where((np.absolute(L[cmp1] - U[cmp1]) < options["tol"]) & (np.absolute(L[cmp1] - 0) < options["tol"]))
    findFz = np.where((np.absolute(L[cmp2] - U[cmp2]) < options["tol"]) & (np.absolute(L[cmp2] - 0) < options["tol"]))
    Fx = findFx[0]
    Fz = findFz[0]

    # Setup constants for passage into opt_dnn subroutine
    # Constant n saved above
    bign = size(A, axis=1)

    # Assign fixed values to lower and upper bounds
    L_save = np.copy(L)  # Subproblems will only differ in L and U
    U_save = np.copy(U)
    B = np.array([])

    # -------------------------
    # Initialize B&B structures
    # -------------------------
    LBLB = -np.inf
    # FxFx{1} = Fx
    # FzFz{1} = Fz
    SS = np.zeros((1 + bign) ** 2)
    # SIGSIG = -1; % Signal that we want default sig in aug Lag algorithm

    # ------------------------------------------------------------------
    # Calculate first global upper bound and associated fathoming target
    # ------------------------------------------------------------------
    if options['use_quadprog']:
        # quadopts = optimset('LargeScale', 'off', 'Display', 'off', 'Algorithm', 'interior-point-convex');
        # [xx, gUB] = quadprog(H, f, [], [], A, b, L_save, U_save, [], quadopts);
        solution = quadprog(H, f, A, b, L_save, U_save)
        if solution['status'] != 'optimal':
            print('Not solution optimal in quadratic programming!\n\n')
            raise Exception("Not exist solution optimal!!")
        xx = np.array(solution['x']).reshape((n,))
        print("---->", xx)
    else:
        xx = np.array([])
        gUB = np.inf

    # if gUB == Inf
    #   LB_target = Inf;
    # elsex
    #   LB_target = gUB - options.fathom_tol*max(1,abs(gUB));
    # end
    # LB_beat = -Inf;

    # ----------------------
    # Begin BRANCH-AND-BOUND
    # ----------------------

    nodes_created = 1
    nodes_solved = 0
    nodes_solved_fully = 0
    nodes_solved_fathomed = 0
    nodes_solved_poorly = 0
    nodes_infeasible = 0
    nodes_pruned = 0

    # %% ------------------------------------------
    # %% While there are still nodes in the tree...
    # %% ------------------------------------------
    #
    # %% Store fixed components index to prevent these variables's bounds from
    # %% changing
    # Fx0 = Fx;
    # Fz0 = Fz;
    #
    # t0 = m+lenL;
    # t1 = m+lenL+lenB;
    #
    # k = 0;
    #
    # if options.verbosity >=1
    #   fprintf('\n=============================== Initial Status =================================\n');
    # end

    # while length(LBLB) > 0
    #
    #     %% -----------------------------
    #     %% Load check point if requested
    #     %% -----------------------------
    #
    #     if length(options.checkfile) > 0 & nodes_solved == 0
    #       load(options.checkfile);
    #     end
    #
    #     %% ------------------------
    #     %% Check point if requested
    #     %% ------------------------
    #
    #     if options.checkpt > 0 & mod(nodes_solved+1,options.checkpt) == 0
    #       filestr = strcat('check',sprintf('%04d',nodes_solved+1),'.mat');
    #       save(filestr);
    #     end
    #
    #     %% ------------
    #     %% Print status
    #     %% ------------
    #
    #     if options.verbosity >= 1
    #
    #       fprintf('\n');
    #       fprintf('STATUS 1: (gUB,gLB,gap) = (%.8e, %.8e, %.3f%%)\n', ...
    #           gUB+cons, min(LBLB)+cons, 100*(gUB - min(LBLB))/max([1,abs(gUB+cons)]));
    #       fprintf('STATUS 2: (created,solved,pruned,infeas,left) = (%d,%d,%d,%d,%d)\n', ...
    #           nodes_created, nodes_solved, nodes_pruned, nodes_infeasible, length(LBLB));
    #       fprintf('STATUS 3: solved = fully + fathomed + poorly : %d = %d + %d + %d\n', ...
    #           nodes_solved, nodes_solved_fully, nodes_solved_fathomed, nodes_solved_poorly);
    #       fprintf('STATUS 4: time = %d\n', toc);
    #
    #       fprintf('\n\n==================================== Node %d =====================================\n',nodes_solved+1);
    #     end
    #
    #     %% -------------------------------------
    #     %% Terminate if too much time has passed
    #     %% -------------------------------------
    #
    #     if toc > options.max_time
    #       stat.status = 'time_limit';
    #       break;
    #     end
    #
    #     %% -----------------------------------------------
    #     %% Sort nodes for 'best-bound' node-selection rule
    #     %% -----------------------------------------------
    #
    #     [LBLB,I] = sort(LBLB,2,'descend');
    #     FxFx = FxFx(I);
    #     FzFz = FzFz(I);
    #     SS = SS(:,I);
    #     SIGSIG = SIGSIG(I);
    #
    #     %% ---------------------------------------------------
    #     %% Pull last problem off the problem list (best-bound)
    #     %% ---------------------------------------------------
    #
    #     LB = LBLB(end);
    #
    #     Fx = FxFx(end); Fx = Fx{1};
    #     Fz = FzFz(end); Fz = Fz{1};
    #     S = reshape(SS(:,end), 1+bign, 1+bign);
    #     SIG = SIGSIG(end);
    #     if SIG < 0.0 % Signal that we want default sig in aug Lag algorithm
    #         SIG = [];
    #     end
    #
    #     %% ---------------------------------
    #     %% Delete that problem from the tree
    #     %% ---------------------------------
    #
    #     LBLB = LBLB(1:end-1);
    #     FxFx = FxFx(1:end-1);
    #     FzFz = FzFz(1:end-1);
    #     SS = SS(:,1:end-1);
    #     SIGSIG = SIGSIG(1:end-1);
    #
    #     %% ------------------
    #     %% Handle single node
    #     %% ------------------
    #
    #     %% ----------------------------
    #     %% Prepare problem to be solved
    #     %% ----------------------------
    #
    #     L = L_save;
    #     U = U_save;
    #
    #     %%------------------------------------------
    #     %% Prepare L and U for new structure of vars
    #     %%------------------------------------------
    #     %% X(Fx) = 0, X(Fz) = 0
    #
    #     set1 = cmp1(Fx);
    #     set2 = cmp2(Fz);
    #     U(set1) = 0;     % set the components to be zero
    #     U(set2) = 0;     % set the lambda components to be zero
    #
    #     %% Setup LB_beat
    #
    #     if LB == -Inf
    #       LB_beat = -Inf;
    #     else
    #       LB_beat = LB - options.fathom_tol*max(1,abs(LB));
    #     end
    #
    #     %% Sam: Need to use U-L to find fixed variables. Add same
    #     %% fixings to Ax=b. If variable is fixed to 0, can zero
    #     %% out other entries in same column of A.
    #
    #
    #     [local_A,local_b] = fixedAb(A,b,L,U);
    #
    #     %% -----------------------------------
    #     %% Solve doubly nonnegative relaxation
    #     %% -----------------------------------
    #
    # %    if options.verbosity > 2
    # %      Fx, Fz
    # %    end
    #
    #     if isfeasible(local_A,local_b,L,U)
    #       [newLB,Y,Z,S,SIG,ret] = opt_dnn(H,f,local_A,local_b,B,E,L,U,options.max_iter,S,SIG,LB_target,LB_beat,options.max_time-toc,cons,options.verbosity);
    #     else
    #       ret = 'infeas';
    #     end
    #
    #     if ~strcmp(ret,'infeas')
    #
    #       %% ------------
    #       %% Post-process
    #       %% ------------
    #
    #       %% If newLB < LB, then it means that the subproblem did not solve
    #       %% well because theoretically, newLB >= LB at optimality. So we take
    #       %% this as a sign that sig needs to be reset. So we set SIG = -1 to
    #       %% signal that we want sig reset for the children.
    #       %%
    #       %% Otherwise, we update LB and save SIG for any children.
    #
    #       if strcmp(ret,'poor')
    #         SIG = -1.0;
    #         S = zeros(size(S,1));
    #       end
    #       if newLB >= LB
    #         LB = newLB;
    #       end
    #
    #       nodes_solved = nodes_solved + 1;
    #       if strcmp(ret,'fathom')
    #         nodes_solved_fathomed = nodes_solved_fathomed + 1;
    #       elseif strcmp(ret,'poor')
    #         nodes_solved_poorly = nodes_solved_poorly + 1;
    #       else
    #         nodes_solved_fully = nodes_solved_fully + 1;
    #       end
    #
    #       %% Save multiplier
    #
    #       S = reshape(S, (1+bign)^2, 1);
    #
    #       %% Extract upper bound (4-part primal heuristic)
    #       %%
    #       %% First extract 0-th column of Y and project it onto Ax=b,x>=0
    #       %% using CPLEX. Get value and update gUB if necessary.
    #       %%
    #       %% Then run quadprog() from this point (if desired).
    #       %%
    #       %% Next extract 0-th column of Z and project it onto Ax=b,x>=0 using
    #       %% CPLEX. Get value and update gUB if necessary.
    #       %%
    #       %% Then run quadprog() from this point (if desired).
    #       %%
    #       %% We presume that CPLEX can return as good a feasible solution as
    #       %% any algorithm. (We do not check anything at the moment. Is this
    #       %% safe?)
    #       %%
    #       %% Right now, we use projected version of 0-th col of Z for
    #       %% branching. Is this a good choice?
    #
    #       x0 = Y(2:bign+1,1);
    #       x0 = project(x0,local_A,local_b,L,U); %% In CPLEX we trust!
    #       x0val = 0.5*x0'*H*x0 + f'*x0;
    #       if feasible(x0,A,b,L_save,U_save,options.tol) & x0val < gUB % x0 is best so far
    #         gUB = x0val;
    #         xx = x0;
    #       end
    #
    #       if options.use_quadprog
    #         quadopts = optimset('LargeScale','off','Display','off','Algorithm','interior-point-convex');
    #         [tmpx,tmpval] = quadprog(H,f,[],[],A,b,L_save,U_save,x0,quadopts);
    #       else
    #         tmpx = [];
    #         tmpval = Inf;
    #       end
    #
    #       if feasible(tmpx,A,b,L_save,U_save,options.tol) & tmpval < gUB
    #         gUB = tmpval;
    #         xx = tmpx;
    #       end
    #
    #       x0 = Z(2:bign+1,1)/Z(1,1);
    #       % special case when x0=(NaN,...NaN)
    #       if(any(isnan(x0)))
    #           x0(isnan(x0)) = 0;
    #       end
    #       x0 = project(x0,local_A,local_b,L,U); %% In CPLEX we trust!
    #       x0val = 0.5*x0'*H*x0 + f'*x0;
    #       if feasible(x0,A,b,L_save,U_save,options.tol) & x0val < gUB % x0 is better than what quadprog found
    #         gUB = x0val;
    #         xx = x0;
    #       end
    #
    #       if options.use_quadprog
    #         quadopts = optimset('LargeScale','off','Display','off','Algorithm','interior-point-convex');
    #         [tmpx,tmpval] = quadprog(H,f,[],[],A,b,L_save,U_save,x0,quadopts);
    #       else
    #         tmpx = [];
    #         tmpval = Inf;
    #       end
    #       if feasible(tmpx,A,b,L_save,U_save,options.tol) & tmpval < gUB
    #         gUB = tmpval;
    #         xx = tmpx;
    #       end
    #
    #       %% Update fathoming target
    #
    #       if gUB == Inf
    #         LB_target = Inf;
    #       else
    #         LB_target = gUB - options.fathom_tol*max(1,abs(gUB));
    #       end
    #
    #       %% ----------------------
    #       %% Prune tree by gUB
    #       %% ----------------------
    #
    #       tmpsz = length(LBLB);
    #
    #       I = find(LBLB < LB_target);
    #       LBLB = LBLB(I);
    #       FxFx = FxFx(I);
    #       FzFz = FzFz(I);
    #       SS = SS(:,I);
    #       SIGSIG = SIGSIG(I);
    #
    #       nodes_pruned = nodes_pruned + (tmpsz - length(LBLB));
    #
    #       %% ------------------------------------------------------------------
    #       %% Select index to branch on (but will only branch if LB < LB_target)
    #       %% ------------------------------------------------------------------
    #
    #       if length(union(Fx,Fz)) < m0
    #           x0 = Y(2:bign+1,1);
    #           s = x0(cmp1);
    #           lambda = x0(cmp2);
    #           [vio,index] = max( s .* lambda );
    #           if vio == 0 % Got unlucky, just select first index available for branching
    #               Ffix = union(Fx0, Fz0);
    #               tmpI = setdiff( setdiff(1:m0,Ffix) , union(Fx,Fz));
    #               index = tmpI(1);
    #           end
    #
    #           %% ---------------------
    #           %% Branch (if necessary)
    #           %% ---------------------
    #           %%
    #           %% We do not check primal feasibility because (x,z) are assumed
    #           %% part of a feasible x0 via CPLEX (see above). In CPLEX we
    #           %% trust!
    #           %%
    #            if LB < LB_target & vio > options.fathom_tol
    #
    #               if index <= t0
    #                 Fxa = union(Fx,index);
    #                 Fza = Fz;
    #
    #                 Fxb = Fx;
    #                 Fzb = union(Fz,index);
    #
    #                 LBLB   = [LBLB  ,LB ,LB ];
    #                 SS     = [SS    ,S  ,S  ];
    #                 SIGSIG = [SIGSIG,SIG,SIG];
    #
    #                 FxFx{length(FxFx)+1} = Fxa;
    #                 FzFz{length(FzFz)+1} = Fza;
    #
    #                 FxFx{length(FxFx)+1} = Fxb;
    #                 FzFz{length(FzFz)+1} = Fzb;
    #
    #                 nodes_created = nodes_created + 2;
    #
    #               elseif index <= t1
    #               %% if t0 < index <= t1, then only add index to Fx if index+lenB not in Fx
    #               %% in case the second 'if' holds, then add index+lenB to Fz
    #
    #               if ~ismember(index+lenB,Fx)
    #                 Fxa = union(Fx,index);
    #                 Fza = union(Fz,index+lenB);
    #                 FxFx{length(FxFx)+1} = Fxa;
    #                 FzFz{length(FzFz)+1} = Fza;
    #                 LBLB   = [LBLB  ,LB ];
    #                 SS     = [SS    ,S  ];
    #                 SIGSIG = [SIGSIG,SIG];
    #                 nodes_created = nodes_created + 1;
    #               end
    #               Fxb = Fx;
    #               Fzb = union(Fz,index);
    #               FxFx{length(FxFx)+1} = Fxb;
    #               FzFz{length(FzFz)+1} = Fzb;
    #               LBLB   = [LBLB  ,LB ];
    #               SS     = [SS    ,S  ];
    #               SIGSIG = [SIGSIG,SIG];
    #               nodes_created = nodes_created + 1;
    #             else
    #               %% if index > t1, then only add index to Fx if index-lenB not in Fx
    #               %% in case the second if holds, then add index-lenB to Fz
    #
    #               if ~ismember(index-lenB,Fx)
    #                 Fxa = union(Fx,index);
    #                 Fza = union(Fz,index-lenB);
    #                 FxFx{length(FxFx)+1} = Fxa;
    #                 FzFz{length(FzFz)+1} = Fza;
    #                 LBLB   = [LBLB  ,LB ];
    #                 SS     = [SS    ,S  ];
    #                 SIGSIG = [SIGSIG,SIG];
    #                 nodes_created = nodes_created + 1;
    #               end
    #               Fxb = Fx;
    #               Fzb = union(Fz,index);
    #               FxFx{length(FxFx)+1} = Fxb;
    #               FzFz{length(FzFz)+1} = Fzb;
    #               LBLB   = [LBLB  ,LB ];
    #               SS     = [SS    ,S  ];
    #               SIGSIG = [SIGSIG,SIG];
    #               nodes_created = nodes_created + 1;
    #             end
    #
    #               %% ----------------------
    #               %% End branching decision
    #               %% ----------------------
    #
    #           end
    #
    #       end
    #
    #     else
    #
    #       nodes_infeasible = nodes_infeasible + 1;
    #
    #     end
    #
    #     %% ---------------------------
    #     %% End handling of single node
    #     %% ---------------------------
    #
    #     %% -------------------------------
    #     %% End loop over nodes in the tree
    #     %% -------------------------------
    #
    # end
