from inexact_line_search import  f_inexact_lsearch
import numpy as np
from scipy import linalg

# Abstract Solver Class
class Solver(object):
    def __init__(self,
                 iter_num,
                 delta,
                 init_x,
                 n):
        """
        :param iter_num: Maximum iteration number
        :param delta:
        :param init_x: iteration initial point
        :param n: number of domain's dimensions
        """
        self.maxIter = iter_num
        self.tol = delta
        self.size = n
        self.startpoint = init_x

    def compute_distance(self, cur_val, opt_val, init_val):
        return (cur_val - opt_val)/float(init_val - opt_val)
    def compute_distance2(self, grad, orig_grad):
        return linalg.norm(grad)/(1+linalg.norm(orig_grad))

# Steepest Descent Solver
class SD(Solver):
    def __init__(self, iter_num, delta, init_x, n):
        super(SD, self).__init__(iter_num, delta, init_x, n)

    def run(self, fct, x_new):
        f_init_val = fct.val(x_new)
        grad = fct.grad(x_new)

        iter = 0
        dist = 1.0
        f_val = f_init_val
        while iter < self.maxIter and dist > self.tol:
            d = -grad
            self.alpha = f_inexact_lsearch(x_new,d, grad, f_val, self.size, fct)
            x_new = x_new + self.alpha*d
            grad = fct.grad(x_new)
            f_val = fct.val(x_new)
            dist = self.compute_distance(f_val, 0.0, f_init_val)
            iter = iter+1

            print x_new[:8]
            print("--iter %d out of %d: dist = %f" %(iter, self.maxIter, dist))

        return x_new, iter, dist

# Abstract Class for Conjugate Gradient
class CG(Solver):
    def __init__(self, iter_num, delta, init_x, n):
        super(CG, self).__init__(iter_num, delta, init_x, n)

    def run(self, fct, x_new):
        f_init_val = fct.val(x_new)
        grad_new = fct.grad(x_new)
        d = -grad_new
        f_val = f_init_val

        iter = 0
        dist = 1.0
        while iter < self.maxIter and dist > self.tol:
            self.alpha = f_inexact_lsearch(x_new, d, grad_new, f_val, self.size, fct)
            x_new = x_new + self.alpha*d
            f_val = fct.val(x_new)
            grad_old = grad_new
            grad_new = fct.grad(x_new)

            beta = self.compute_beta(grad_old, grad_new)
            d = -grad_new + beta*d
            iter = iter+1
            dist = self.compute_distance(f_val, 0.0, f_init_val)

            print x_new[:8]
            print("--iter %d out of %d: dist = %f"%(iter, self.maxIter, dist))

        return x_new, iter, dist

    def compute_beta(self, grad_old, grad_new):
        beta = 0.0
        return beta
# Fletcher-Reeves Conjugate Gradient
class FR_CG(CG):
    def __init__(self, iter_num, delta, init_x, n):
        super(FR_CG, self).__init__(iter_num, delta, init_x, n)

    def compute_beta(self, grad_old, grad_new):
        beta = np.inner(grad_new, grad_new)/np.inner(grad_old, grad_old)
        return beta

# Polak-Ribiere Conjugate Gradient
class PR_CG(CG):
    def __init__(self, iter_num, delta, init_x, n):
        super(PR_CG, self).__init__(iter_num, delta, init_x, n)

    def compute_beta(self, grad_old, grad_new):
        beta = np.inner(grad_new, grad_new-grad_old)/np.inner(grad_old, grad_old)
        return beta

#Abstract Class for Quasi Newton
class QN(Solver):
    def __init__(self, iter_num, delta, init_x, n):
        super(QN, self).__init__(iter_num, delta, init_x, n)

    def initialize_H(self, fct, x_new):
        H_inv = fct.hessian(x_new)
        H = linalg.pinv(H_inv)
        return H

    def update_H(self, H, x_old, x_new, grad_old, grad_new):

        return H

    def run(self, fct, x_new):
        f_init_val = fct.val(x_new)
        f_val = f_init_val
        H = self.initialize_H(fct, x_new)
        grad_new = fct.grad(x_new)

        iter = 0
        dist = 1.0
        while iter < self.maxIter and dist > self.tol:
            d = -np.dot(H, grad_new)
            alpha = f_inexact_lsearch(x_new, d, grad_new, f_val, self.size, fct)

            x_old = x_new
            x_new = x_old + alpha*d
            grad_old = grad_new
            grad_new = fct.grad(x_new)
            f_val = fct.val(x_new)

            H = self.update_H(H, x_old, x_new, grad_old, grad_new)

            dist = self.compute_distance(f_val, 0.0, f_init_val)
            iter = iter+1

            print x_new[:8]
            print('--iter %d out of %d: dist = %f'%(iter, self.maxIter, dist))

        return x_new, iter, dist

# BFGS Quasi Newton
class BFGS(QN):
    def __init__(self, iter_num, delta, init_x, n):
        super(BFGS, self).__init__(iter_num, delta, init_x, n)

    def update_H(self, H, x_old, x_new, grad_old, grad_new):
        q = grad_new - grad_old
        p = x_new - x_old
        rho = p.dot(q)
        Hq = H.dot(q)
        r = q.dot(Hq)

        H = H - np.outer(p, Hq)/(rho+1E-16) - np.outer(Hq, p)/(rho+1E-16) + \
            r*np.outer(p, p)/(rho**2+1E-16) + np.outer(p, p)/(rho+1E-16)

        return H

# DFP Quasi Newton
class DFP(QN):
    def __init__(self, iter_num, delta, init_x, n):
        super(DFP, self).__init__(iter_num, delta, init_x, n)

    def update_H(self, H, x_old, x_new, grad_old, grad_new):
        q = grad_new - grad_old
        p = x_new - x_old
        r1 = p.dot(q)
        y = H.dot(q)
        r2 = q.dot(y)

        p = p[np.newaxis].T
        H = H - np.outer(y, y)/(r2+1E-16) + np.outer(p, p)/(r1+1E-16)

        return H

# Limited Memory Quasi Newton
class LM_BFGS(QN):
    def __init__(self, iter_num, delta, init_x, n, m=5):
        super(LM_BFGS, self).__init__(iter_num, delta, init_x, n)
        self.m = m

    def init_Hk(self, p_q_pair):
        p, q = p_q_pair
        gamma = p.dot(q)/q.dot(q)
        H_zero = gamma*np.diag([1]*self.size)

        return H_zero


    def lm_update_H(self, H, p_q_pair):
        p, q = p_q_pair
        rho = p.dot(q)
        Hq = H.dot(q)
        r = q.dot(Hq)

        H = H - np.outer(p, Hq)/(rho+1E-16) - np.outer(Hq, p)/(rho+1E-16) + \
            r*np.outer(p, p)/(rho**2+1E-16) + np.outer(p, p)/(rho+1E-16)

        return H

    def lm_update_d(self, cache, rho_inv, grad):
        H = self.init_Hk(cache[-1])
        g = grad
        alpha = np.zeros(self.m)
        for i in xrange(1, self.m+1):
            p, q = cache[-i]
            alpha[-i] = p.dot(g)/rho_inv[-i]
            g = g - alpha[-i]*q
        r = np.dot(H, g)
        for i in xrange(self.m):
            p, q = cache[i]
            beta = q.dot(r)/rho_inv[i]
            r = r+p*(alpha[i]-beta)

        return -r




    def run(self, fct, x_new):
        f_init_val = fct.val(x_new)
        f_val = f_init_val
        grad_new = fct.grad(x_new)
        cache = list()
        rho_inv = list()

        iter = 0
        dist = 1.0
        H = self.initialize_H(fct, x_new)
        while(iter < self.m and dist > self.tol):
            d = -np.dot(H, grad_new)
            alpha = f_inexact_lsearch(x_new, d, grad_new, f_val, self.size, fct)

            x_old = x_new
            x_new = x_new + alpha*d
            grad_old = grad_new
            grad_new = fct.grad(x_new)
            f_val = fct.val(x_new)

            p = x_new - x_old
            q = grad_new - grad_old
            cache.append((p, q))
            rho_inv.append(p.dot(q))
            H = self.lm_update_H(H, cache[-1])
            dist = self.compute_distance(f_val, 0.0, f_init_val)
            iter = iter+1

            print x_new[:8]
            print('--iter %d out of iter %d: dist = %f'%(iter, self.maxIter, dist))

        if  dist <= self.tol:
            return x_new, iter, dist

        while(iter < self.maxIter and dist > self.tol):
            d = self.lm_update_d(cache, rho_inv, grad_new)

            alpha = f_inexact_lsearch(x_new, d, grad_new, f_val, self.size, fct)
            x_old = x_new
            x_new = x_new + alpha*d
            grad_old = grad_new
            grad_new = fct.grad(x_new)
            f_val = fct.val(x_new)

            cache.pop(0)
            p = x_new - x_old
            q = grad_new - grad_old
            cache.append((p, q))
            rho_inv.pop(0)
            rho_inv.append(p.dot(q))

            dist = self.compute_distance(f_val, 0.0, f_init_val)
            iter = iter+1

            print(x_new[:8])
            print("--iter %d out of %d: dist = %f"%(iter, self.maxIter, dist))

        return x_new, iter, dist



















