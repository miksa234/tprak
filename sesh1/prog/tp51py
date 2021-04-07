#!/usr/bin/env python3.9

import numpy as np

def main():

    # one experiment
    sig = 0.2; s = 0.2
    p0 = 0.9            # guess

    y0 = np.array([0.8, 1.2])
    A = np.array([1, 1])
    cov_sta = np.array([[sig**2, 0], [0, sig**2]])

    for i in range(10):
        cov_sys = np.array([[s**2*p0**2, s**2*p0**2], [s**2*p0**2, s**2*p0**2]])
        P = np.linalg.inv(cov_sys+cov_sta)
        p0 = (A.T@P@A)**(-1) * A.T@P@y0

    dp = (A.T@P@A)**(-1/2)

    print('One Experiment')
    print(f'p = {p0} \pm {round(dp, 2)}\n')

    # two experiments

    p0 = 0.9            # guess

    for i in range(10):
        cov_sys = np.array([[s**2*p0**2, 0], [0, s**2*p0**2]])
        P = np.linalg.inv(cov_sys+cov_sta)
        p0 = (A.T@P@A)**(-1) * A.T@P@y0

    dp = (A.T@P@A)**(-1/2)

    print('Two Experiments')
    print(f'p = {p0} \pm {round(dp, 2)}\n')


if __name__ == "__main__":
    main()
