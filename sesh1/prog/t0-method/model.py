#!/usr/bin/env python3.9

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


global m_p; m_p = 0.13957

sig_p = lambda x: sp.sqrt(1 - 4*m_p**2/x)
g_s = lambda s, m_q, g_q: g_q*s/m_q**2 * (sig_p(s)/sig_p(m_q**2))**3 * sp.Heaviside(s - 4*m_p**2, 0)

def model(s, m_q, g_q, m_w, g_w, e_w, a, b, c):
    part1 = (m_q)**4/((m_q**2 - s)**2 + m_q**2*g_s(s, m_q, g_q)**2)
    part2 = 1 + (e_w * 2*s * (m_w**2 - s))/((m_w**2 - s)**2 + m_w**2*g_w**2)
    part3 = c*(1 + a*s + b*s**2)**2
    return part1 * part2 * part3


def my_plot(name, x_data, y_data, p, dp, sigma):

    for i in range(len(p)):
        print(f'{p[i]}\t\\pm {dp[i]}')

    x_model = np.linspace(x_data[0], x_data[-1], 500)

    x = sp.symbols('x')
    la_mod = model(x, *p)
    la_mod = sp.lambdify(x, la_mod, modules=['numpy', {'Heaviside': np.heaviside}])
    y_model = la_mod(x_model)

    plt.figure(figsize=[10, 7])
    plt.errorbar(x_data, y_data, yerr=sigma, c='black', fmt='.k')
    plt.plot(x_model, y_model, c='red')

    p, dp = np.round(p, 5), np.round(dp, 3)
    plt.annotate(r'$M_{\rho} = $' + f'({p[0]}' + r'$\pm$' + f'{dp[0]}) GeV', (0.2, 40))
    plt.annotate(r'$\Gamma_{\rho} = $' + f'({p[1]}' + r'$\pm$' + f'{dp[1]}) GeV', (0.2, 38))
    plt.annotate(r'$M_{\omega} = $' + f'({p[2]}' + r'$\pm$' + f'{dp[2]}) GeV', (0.2, 36))
    plt.annotate(r'$\Gamma_{\omega} = $' + f'({p[3]}' + r'$\pm$' + f'{dp[3]}) GeV', (0.2, 34))

    plt.savefig(f'./plots/{name}.png')
    plt.close()
