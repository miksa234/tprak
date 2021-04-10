#!/usr/bin/env python3.9

import numpy as np
import sympy as sp

global x; x = sp.symbols("x")

def design_matrix(model, var):
    A = sp.diff(model(x, *var), var.T)
    A_f = sp.lambdify((x, *var), A, modules=['numpy', {'Heaviside': np.heaviside}])
    return A_f


def f2_lambda(model, var):
    f2_f = sp.lambdify((x, *var), model(x, *var), modules=['numpy', {'Heaviside': np.heaviside}])
    return f2_f

def t0_fit(model, var_str, x_data, y_data,\
           p0, cov_stat, cov_relsyst, iterations=100, alpha=0.1):

    var = sp.Matrix(sp.symbols(var_str))
    A_func = design_matrix(model, var)
    f2_func = f2_lambda(model, var)

    for i in range(iterations):
        A = A_func(x_data, *p0)[0].T
        f2 = f2_func(x_data, *p0)

        P = np.linalg.inv(cov_relsyst @ np.outer(f2, f2) + cov_stat)

        delta_p = np.linalg.inv(A.T @ P @ A) @ A.T @ P @ (y_data - f2)
        p0 = p0 + alpha*delta_p
        print(f"{i} Iterations completed")

    print('Done\n')

    dp = np.sqrt(np.diag(np.linalg.inv(A.T@P@A)))
    return p0, dp
