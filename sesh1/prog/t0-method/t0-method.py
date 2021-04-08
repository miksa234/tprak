#!/usr/bin/env python3.9

import numpy as np
import sympy as sp

def design_matrix(model, x_data, p0, var_str):
    var = sp.Matrix(sp.symbols(var_str))
    x = symbols("x")

    A = sp.Matrix(model(x, *var)).jacobian(var)
    A = A.subs({var[i]: p0[i] for i in range(len(p0))})

    return np.array([np.array([A.subs({x: x_data[i]]})) for i in range(len(s))])

def cov_syst(model, x_data, p0, cov_relsyst)
    return cov_relsyst @ model(x_data, *p0) @ model(x_data, *p0).T


def t0_fit(model, var_str, x_data, y_data, p0, cov_stat, cov_relsyst, iterations=100):

    A = design_matrix(model, x_data, p0, var_str)

    for i in range(iterations):
        P = np.linalg.inv(cov_syst(model, x_data, p0, cov_relsyst) + cov_stat)
        p0 = (A.T @ P @ A).inv() @ A.T @ P @ y

    dp = (A.T@P@A).inv()**(1/2)

    return p0, dp
