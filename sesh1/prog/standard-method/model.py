#!/usr/bin/env python3.9

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

global m_p; m_p = 0.13957

sig_p = lambda x: np.sqrt(1 - 4*m_p**2/x)
g_s = lambda s, m_q, g_q: g_q*s/m_q**2 * (sig_p(s)/sig_p(m_q**2))**3 * np.heaviside(s - 4*m_p**2, 0)

def model(s, m_q, g_q, m_w, g_w, e_w, a, b, c):
    part1 = (m_q)**4/((m_q**2 - s)**2 + m_q**2*g_s(s, m_q, g_q)**2)
    part2 = 1 + (e_w * 2*s * (m_w**2 - s))/((m_w**2 - s)**2 + m_w**2*g_w**2)
    part3 = c*(1 + a*s + b*s**2)**2
    return part1 * part2 * part3

def myplot(name, x_data, y_data, p, dp, sigma):

    x_model = np.linspace(x_data[0], x_data[-1], 500)
    y_model = model(x_model, *p)

    plt.figure(figsize=[10, 7])
    plt.title(f'{name} DATA FIT')

    plt.errorbar(x_data, y_data, yerr=sigma, fmt='.k', c='black', label=f'{name}')
    plt.plot(x_model, y_model, color='red', label='fit')

    p, dp = np.round(p, 3), np.round(dp, 3)
    plt.annotate(r'$M_{\rho} = $' + f'({p[0]}' + r'$\pm$' + f'{dp[0]}) GeV', (0.2, 40))
    plt.annotate(r'$\Gamma_{\rho} = $' + f'({p[1]}' + r'$\pm$' + f'{dp[1]}) GeV', (0.2, 38))
    plt.annotate(r'$M_{\omega} = $' + f'({p[2]}' + r'$\pm$' + f'{dp[2]}) GeV', (0.2, 36))
    plt.annotate(r'$\Gamma_{\omega} = $' + f'({p[3]}' + r'$\pm$' + f'{dp[3]}) GeV', (0.2, 34))

    plt.title(f't0-Singlefit of {name}')
    plt.ylabel(r'$|F_{\pi}^V(s)|^2|$')
    plt.xlabel(r'$\sqrt{s}$[GeV]')
    plt.legend(loc='best')

    plt.savefig(f'./plots/{name}.png')
    plt.close()
