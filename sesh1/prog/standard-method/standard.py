#!/usr/bin/env python3.9

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from model import *


global p0
p0 = [0.77, 0.15, 0.78, 0.09, 0.0085, 0.002, 0.16, 0.3]   # in GeV

def SND():
    data = np.loadtxt('../data/SND-VFF.txt')
    x_data = data[:,0]
    y_data = data[:, 1]
    sigma = data[:, 2]

    p, pcov = curve_fit(model, x_data, y_data, p0, sigma=sigma)
    myplot('SND', x_data, y_data, p, np.sqrt(np.diag(pcov)), sigma)

    return p, np.sqrt(np.diag(pcov))

def CMD2():
    data = np.loadtxt('../data/CMD2-VFF.txt')
    x_data = data[:,0]
    y_data = data[:, 1]
    sigma = data[:, 2]

    p, pcov = curve_fit(model, x_data, y_data, p0, sigma=sigma)
    myplot('CMD2', x_data, y_data, p, np.sqrt(np.diag(pcov)), sigma)

    return p, np.sqrt(np.diag(pcov))

def BABAR():
    data = np.loadtxt('../data/BABAR-VFF.txt')
    cov_stat = np.loadtxt('../data/BABAR-StatCov.txt')
    x_data = data[:,0]
    y_data = data[:, 1]

    p, pcov = curve_fit(model, x_data, y_data, p0, sigma=cov_stat)
    myplot('BABAR', x_data, y_data, p, np.sqrt(np.diag(pcov)), sigma=np.sqrt(np.diag(cov_stat)))

    return p, np.sqrt(np.diag(pcov))

def KLOE():
    data = np.loadtxt('../data/KLOE-VFF.txt')
    cov_stat = np.loadtxt('../data/KLOE-StatCov.txt')
    x_data = data[:,0]
    y_data = data[:, 1]

    p, pcov = curve_fit(model, x_data, y_data, p0, sigma=cov_stat)
    myplot('KLOE', x_data, y_data, p, np.sqrt(np.diag(pcov)), sigma=np.sqrt(np.diag(cov_stat)))

    return p, np.sqrt(np.diag(pcov))


def main():
    p1, dp1 = SND()
    p2, dp2 = CMD2()
    p3, dp3 = BABAR()
    p4, dp4 = KLOE()

    p = 1/4*(p1 + p2 + p3 + p4)
    dp = 1/4*(dp1 + dp2 + dp3 + dp4)

    for i in range(len(p)):
        print(p[i], dp[i], sep='\t')


if __name__ == "__main__":
    main()
