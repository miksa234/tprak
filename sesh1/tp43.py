#!/usr/bin/env python3.9

from sympy import *

def main():
    p_, n, nprime = symbols('p_, n, nprime')
    vr = Matrix([p_, n, nprime]) # array of variables

    p = Matrix([1, 0.8, 1.2])       # initial guess
    y = Matrix([0.8, 1.2, 1, 1])    # observables
    P = eye(4)*25                   # inverse convariace matrix
    f = Matrix(([n*p_, nprime*p_, n, nprime])) # parameters


    for _ in range(10):

        f0 = f.subs({p_:p[0], n:p[1], nprime:p[2]})
        I0 = y - f0

        # design matrix
        A = f.jacobian(vr).subs({p_:p[0], n:p[1], nprime:p[2]})

        dp=(A.T @ P @ A).inv() @ A.T @ P @ I0

        p = p + dp

    cov = (A.T @ P @ A).inv()
    for i in range(3):
        print(f'{vr[i]}:\t {p[i]}\t\\pm {sqrt(cov[i, i])}')

if __name__ == "__main__":
    main()
