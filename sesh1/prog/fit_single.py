#!/usr/bin/env python3.9

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

global m_p; m_p = 0.13957

sig_p = lambda x: np.sqrt(1 - 4*m_p**2/x)
g_s = lambda s, m_q, g_q: g_q*s/m_q**2 * (sig_p(s)/sig_p(m_q**2))**2 * np.heaviside(s, 4*m_p**2)

def model(s, m_q, g_q, m_w, g_w, e_w, a, b):
    part1 = (m_q)**4/((m_q**2 - s)**2 + m_q**2*g_s(s, m_q, g_q)**2)
    part2 = 1 + (e_w * 2*s * (m_w**2 - s))/((m_w**2 - s)**2 + m_w**2*g_w**2)
    part3 = (1 + a*s + b*s)**2
    return part1 * part2 * part3

def main():
    data = np.loadtxt('./data/SND-VFF.txt')
    s = data[:,0]
    F2 = data[:, 1]

    p0 = [0.7, 0.2, 0.8, 0.2, 2e-3, 1e4, -1e5]   # in GeV
    popt, pcov = curve_fit(model, s, F2, p0)
    popt, pcov = np.round(popt, 3), np.round(pcov, 3)

    plt.figure(figsize=[10, 7])
    plt.title('SND DATA FIT')
    plt.scatter(s, F2, marker='.', c='black')
    plt.plot(s, model(s, *popt), color='red')
    plt.annotate('Very Bad, A guessing game with the parameters', (0.15, 40))
    plt.annotate(r'No $\omega$ resonance recognized by the fit', (0.15, 35))

    plt.annotate(r'$M_{\rho} = $' + f'({popt[0]}' + r'$\pm$' + f'{np.sqrt(pcov[0][0])}) GeV', (0.7, 40))
    plt.annotate(r'$\Gamma_{\rho} = $' + f'({popt[1]}' + r'$\pm$' + f'{np.sqrt(pcov[1][1])}) GeV', (0.7, 36))
    plt.annotate(r'$M_{\omega} = $' + f'({popt[2]}' + r'$\pm$' + f'{np.sqrt(pcov[2][2])}) GeV', (0.7, 34))
    plt.annotate(r'$\Gamma_{\omega} = $' + f'({popt[3]}' + r'$\pm$' + f'{np.sqrt(pcov[3][3])}) GeV', (0.7, 32))
    plt.annotate(r'$\varepsilon_{\omega} = $' + f'({popt[4]}' + f'$\pm$' + f'{np.sqrt(pcov[4][4])}) GeV', (0.7, 30))
    plt.annotate(f'a = ({popt[5]}' + r'$\pm$' + f'{np.sqrt(pcov[5][5])}) GeV', (0.7, 28))
    plt.annotate(f'b = ({popt[6]}' + r'$\pm$' + f'{np.sqrt(pcov[6][6])}) GeV', (0.7, 26))

    plt.savefig('automated.png')


if __name__ == "__main__":
    main()
