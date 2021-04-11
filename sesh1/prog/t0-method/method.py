#!/usr/bin/env python3.9

import numpy as np
import sympy as sp

global x; x = sp.symbols("x")

def t0_fit(model, var_str, x_data, y_data,\
           p0, cov_stat, cov_relsyst, way=0, iterations=100, alpha=0.1):

    var = sp.Matrix(sp.symbols(var_str))

    jacobi = sp.diff(model(x, *var), var.T)
    A_func = sp.lambdify((x, *var), jacobi, modules=['numpy', {'Heaviside': np.heaviside}])

    f2_func = sp.lambdify((x, *var), model(x, *var), modules=['numpy', {'Heaviside': np.heaviside}])

    chi_sq = []

    for i in range(iterations):
        A = A_func(x_data, *p0)[0].T
        f2 = f2_func(x_data, *p0)

        if way == 0: # consider D'Agostini bias
            P = np.linalg.inv(cov_relsyst @ np.outer(f2, f2) + cov_stat)
        else:       # don't consider D'Agostini bias
            P = np.linalg.inv(cov_relsyst * (p0@p0) + cov_stat)

        delta_p = np.linalg.inv(A.T @ P @ A) @ A.T @ P @ (y_data - f2)
        p0 = p0 + alpha*delta_p

        chi_sq.append((y_data - f2).T @ P @ (y_data - f2))

        print(f"{i} Iterations completed")

    print('Done\n')

    dp = np.sqrt(np.diag(np.linalg.inv(A.T@P@A)))
    return p0, dp, np.array(chi_sq)
