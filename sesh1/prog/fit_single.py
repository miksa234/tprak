#!/usr/bin/env python3.9

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

global m_p; m_p = 0.13957

sig_p = lambda x: np.sqrt(1 - 4*m_p**2/x)
g_s = lambda s, m_q, g_q: g_q*s/m_q**2 * (sig_p(s)/sig_p(m_q**2))**3 * np.heaviside(s,  4*m_p**2)

def model(s, m_q, g_q, m_w, g_w, e_w, a, b, c):
    part1 = (m_q)**4/((m_q**2 - s)**2 + m_q**2*g_s(s, m_q, g_q)**2)
    part2 = 1 + (e_w * 2*s * (m_w**2 - s))/((m_w**2 - s)**2 + m_w**2*g_w**2)
    part3 = c*(1 + a*s + b*s**2)**2
    return part1 * part2 * part3

def main():
    data = np.loadtxt('./data/SND-VFF.txt')
    s = data[:,0]
    F2 = data[:, 1]

    p0 = [0.7, 0.2, 0.8, 0.2, 2e-3, -522.90, 191.40, 0.5]   # in GeV
    popt, pcov = curve_fit(model, s, F2, p0)
    popt, uncert = np.round(popt, 3), np.round(np.sqrt(np.diagonal(pcov)), 3)

    s_model = np.linspace(s[0], s[-1], 500)

    plt.figure(figsize=[10, 7])
    plt.title('SND DATA FIT')
    plt.scatter(s, F2, marker='.', c='black')
    plt.plot(s_model, model(s_model, *popt), color='red')
    plt.annotate('A guessing game with the parameters', (0.15, 40))
    plt.annotate(r'$\Gamma_{\omega}$ bad fit', (0.15, 38))

    plt.annotate(r'$M_{\rho} = $' + f'({popt[0]}' + r'$\pm$' + f'{uncert[0]}) GeV', (0.7, 40))
    plt.annotate(r'$\Gamma_{\rho} = $' + f'({popt[1]}' + r'$\pm$' + f'{uncert[1]}) GeV', (0.7, 38))
    plt.annotate(r'$M_{\omega} = $' + f'({popt[2]}' + r'$\pm$' + f'{uncert[2]}) GeV', (0.7, 36))
    plt.annotate(r'$\Gamma_{\omega} = $' + f'({popt[3]}' + r'$\pm$' + f'{uncert[3]}) GeV', (0.7, 34))

    plt.savefig('fit_single.png')


if __name__ == "__main__":
    main()
