#!/usr/bin/env python3.9

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from model import *


global p0
p0 = [0.9, 0.2, 0.81, 0.04, 0.02, -1, 0.84, 1.55]   # in GeV

def SND():
    data = np.loadtxt('../data/SND-VFF.txt')
    x_data = data[:,0]
    y_data = data[:, 1]
    sigma = data[:, 2]

    p, pcov = curve_fit(model, x_data, y_data, p0, sigma=sigma)
    myplot('SND', x_data, y_data, p, np.sqrt(np.diag(pcov)))

    return p

def CMD2():
    data = np.loadtxt('../data/CMD2-VFF.txt')
    x_data = data[:,0]
    y_data = data[:, 1]
    sigma = data[:, 2]

    p, pcov = curve_fit(model, x_data, y_data, p0, sigma=sigma)
    myplot('CMD2', x_data, y_data, p, np.sqrt(np.diag(pcov)))

    return p

def BABAR():
    data = np.loadtxt('../data/BABAR-VFF.txt')
    cov_stat = np.loadtxt('../data/BABAR-StatCov.txt')
    x_data = data[:,0]
    y_data = data[:, 1]

    p, pcov = curve_fit(model, x_data, y_data, p0, sigma=cov_stat)
    myplot('BABAR', x_data, y_data, p, np.sqrt(np.diag(pcov)))

    return p

def KLOE():
    data = np.loadtxt('../data/KLOE-VFF.txt')
    cov_stat = np.loadtxt('../data/KLOE-StatCov.txt')
    x_data = data[:,0]
    y_data = data[:, 1]

    p, pcov = curve_fit(model, x_data, y_data, p0, sigma=cov_stat)
    myplot('KLOE', x_data, y_data, p, np.sqrt(np.diag(pcov)))

    return p


def main():
    p1 = SND()
    p2 = CMD2()
    p3 = BABAR()
    p4 = KLOE()

    p = 1/4*(p1 + p2 + p3 + p4)

    print(*p, sep='\n')


if __name__ == "__main__":
    main()
