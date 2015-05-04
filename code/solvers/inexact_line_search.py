import numpy as np
import math

# Solving an optimum alpha to minimize f(x+alpha*d)
def f_inexact_lsearch(x, d, g, f_val, n, fct):
    """
    :param x: start point
    :param d: descent direction
    :param g: gradient at x
    :param f_val: function's value at x
    :param n: number of dimensions of function's domain
    :param fct: function
    :return: an optimum stepsize decided by this line search process
    """
    sigma = 0.1     #Wolfe Powell parameter
    rho = 0.01      #Goldstein parameter
    tau1  = 9.0                 # preset factor for jump of alpha */
    tau2  = 0.1                 # preset factor for alpha in sect. */
    tau3  = 0.5                 # preset factor for alpha in sect. */
    f_lower_bound = -1.0e20      # lower bound of objective function */

    alpha = f_bracketing(x, d, g, f_val, n, sigma, rho, tau1, tau2, tau3, f_lower_bound, fct)

    return alpha

def f_bracketing(x, d, g, f_val, n, sigma, rho, tau1, tau2, tau3, f_lower_bound, fct):
    max_iter = 20
    alpha_prev = 0.0
    f_zero = f_val
    df_zero = np.inner(d,g)
    f_a_prev = f_zero
    df_a_prev = df_zero
    mu = (f_lower_bound - f_zero)/(rho*df_zero)

    alpha = 1.0

    k=0
    while(k<max_iter):
        x_alpha = x+alpha*d
        f_a = fct.val(x_alpha)

        if(f_a > (f_zero + alpha*rho*df_zero) or f_a >= f_a_prev):
            alpha = f_sectioning(x, d, rho, tau2, tau3, sigma, alpha_prev, alpha, f_zero,
                         df_zero, f_a_prev, f_a, df_a_prev, alpha, fct)
            return alpha

        g_alpha = fct.grad(x_alpha)
        df_a = np.inner(g_alpha, d)
        if math.fabs(df_a) <= ((-sigma)*df_zero):
            return alpha
        if df_a >= 0.0:
            return f_sectioning(x, d, rho, tau2, tau3, sigma, alpha, alpha_prev, f_zero, df_zero, f_a,
                         f_a_prev, df_a, alpha, fct)

        if mu < 2*alpha-alpha_prev:
            alpha_prev = alpha
            f_a_prev = f_a
            df_a_prev = df_a
            alpha = mu
        else:
            a1 = 2*alpha-alpha_prev
            b1 = min(mu, alpha+tau1*(alpha-alpha_prev))
            tmp = alpha
            alpha =  f_interpolation_quadratic(f_a_prev, df_a_prev, f_a, alpha_prev, alpha, a1, b1)
            alpha_prev = tmp
            f_a_prev = f_a
            df_a_prev = df_a

        k = k+1

def f_sectioning(x, d, rho, tau2, tau3, sigma, a, b, f_zero, df_zero, f_a, f_b, df_a,
                 alpha, fct):

    k=0
    max_iter = 20
    while(k < max_iter):
        a1 = a+tau2*(b-a)
        b1 = b-tau3*(b-a)

        alpha = f_interpolation_quadratic(f_a, df_a, f_b, a, b, a1, b1)

        f_a_prev = f_a
        x_alpha = x+alpha*d
        f_a = fct.val(x_alpha)
        if(f_a > (f_zero + rho*alpha*df_zero) or f_a >= f_a_prev):
            b = alpha
            f_b = f_a
        else:
            g_alpha = fct.grad(x_alpha)
            df_a = np.inner(g_alpha, d)
            if math.fabs(df_a) <= ((-sigma)*df_zero):
                return alpha

            a_prev = a
            a = alpha
            if (b-a_prev)*df_a >= 0.0:
                b = a_prev

        k = k+1

    return alpha

def f_interpolation_quadratic(f_a, df_a, f_b, a, b, a1, b1):
    alpha = 0.0
    za1 = f_a + df_a*(a1 - a) + (f_b - f_a - (b-a)*df_a)*(a1 - a)*(a1 - a)/((b - a)*(b - a))
    zb1 = f_a + df_a*(b1 - a) + (f_b - f_a - (b-a)*df_a)*(b1 - a)*(b1 - a)/((b - a)*(b - a))
    if za1 < zb1:
        endptmin = a1
    else:
        endptmin = b1

    root = a - (b-a)*(b-a)*df_a/(2*(f_b - f_a - (b-a)*df_a))

    if f_b - f_a - (b-a)*df_a < 0:
        if a1 < b1:
            if a1 <= root and root <= b1:
                alpha = endptmin
            if root < a1:
                alpha = b1
            if root > b1:
                alpha = a1
        else:
            if b1 <= root and root <= a1:
                alpha = endptmin
            if root < b1:
                alpha = a1
            if root > a1:
                alpha = b1
    else:
        if a1 < b1:
            if a1 <= root and root <= b1:
                alpha = root
            if root < a1:
                alpha = a1
            if root > b1:
                alpha = b1
        else:
            if b1 <= root and root <= a1:
                alpha = root
            if root < b1:
                alpha = b1
            if root > a1:
                alpha = a1

    return alpha

