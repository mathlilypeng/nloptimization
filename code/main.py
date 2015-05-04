import numpy as np
import time
from solvers import nl_solvers

from obj_fct import *

if __name__ == "__main__":
    maxIter = 20000
    tol = 1E-6
    sizes = 1600
    methods = ["FR_CG", "PR_CG", "BFGS", "DFP", "LM_BFGS"]
    for i in xrange(len(methods)):
        for k in xrange(1):
            n = sizes  # problem size
            fct = Fct2(n)  # select which function to study. Fct1 is Extended Rosenbrock Function, Fct2 is Extended Powell Singular Function
            x = fct.start  # initialize starting point

            solver = eval("nl_solvers."+methods[i])(maxIter, tol, x, n)  # run the nonlinear solver
            start_time = time.time()
            res, iter_num, dist = solver.run(fct, x)  # return the results
            run_time = time.time() - start_time




