#!/usr/bin/env python3.9

from my_plot import *
from method import *
from model import *

global p0;
p0 = [0.9, 0.2, 0.81, 0.04, 0.02, -1, 0.84, 1.55]   # in GeV

def SND():
    data = np.loadtxt('../data/SND-VFF.txt')
    x_data = data[:, 0]
    y_data = data[:, 1]
    cov_stat = (np.eye(len(x_data)) * data[:, 2])**2
    cov_relsyst = (np.eye(len(x_data)) * np.array([0.032 if i<=2 else 0.013 for i in range(len(x_data))]))**2

    var_str = "m_q g_q m_w g_w e_w a b c"
    p, dp, chi_sq = t0_fit(model, var_str, x_data, y_data, p0, cov_stat,\
                    cov_relsyst, way=1)

    sigma = np.diag(cov_stat)
    my_plot('wrong-SND', x_data, y_data, p, dp, chi_sq, sigma)

def CMD2():
    data = np.loadtxt('../data/CMD2-VFF.txt')
    x_data = data[:, 0]
    y_data = data[:, 1]
    cov_stat = (np.eye(len(x_data)) * data[:, 2])**2

    cov_relsyst = []
    for i in range(len(x_data)):
        if i<=43:
            cov_relsyst.append(0.006)
        elif i>43 and i<=53:
            cov_relsyst.append(0.007)
        else:
            cov_relsyst.append(0.008)
    cov_relsyst = (np.eye(len(x_data)) * np.array(cov_relsyst))**2


    var_str = "m_q g_q m_w g_w e_w a b c"
    p, dp, chi_sq = t0_fit(model, var_str, x_data, y_data, p0, cov_stat,\
                    cov_relsyst, way=1)

    sigma = np.sqrt(np.diag(cov_stat))
    my_plot('wrong-CMD2', x_data, y_data, p, dp, chi_sq, sigma)

def KLOE():
    data = np.loadtxt('../data/KLOE-VFF.txt')
    x_data = data[:, 0]
    y_data = data[:, 1]
    cov_relsyst = np.loadtxt('../data/KLOE-RelSystCov.txt')
    cov_stat = np.loadtxt('../data/KLOE-StatCov.txt')

    var_str = "m_q g_q m_w g_w e_w a b c"
    p, dp, chi_sq = t0_fit(model, var_str, x_data, y_data, p0, cov_stat,\
                    cov_relsyst, way=1)

    sigma = np.sqrt(np.diag(cov_stat))
    my_plot('wrong-KLOE', x_data, y_data, p, dp, chi_sq, sigma)

def BABAR():
    data = np.loadtxt('../data/BABAR-VFF.txt')
    x_data = data[:, 0]
    y_data = data[:, 1]
    cov_relsyst = np.loadtxt('../data/BABAR-RelSystCov.txt')
    cov_stat = np.loadtxt('../data/BABAR-StatCov.txt')

    var_str = "m_q g_q m_w g_w e_w a b c"
    p, dp, chi_sq = t0_fit(model, var_str, x_data, y_data, p0, cov_stat,\
                    cov_relsyst, way=1)

    sigma = np.sqrt(np.diag(cov_stat))
    my_plot('wrong-BABAR', x_data, y_data, p, dp, chi_sq, sigma)

def main():
    SND()
    CMD2()
    KLOE()
    BABAR()

if __name__ == "__main__":
    main()
