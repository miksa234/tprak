#!/usr/bin/env python3.9

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.special import gammaincc

from model import *

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

def my_plot(name, x_data, y_data, p, dp, chi_sq, sigma):

#    for i in range(len(p)):
#        print(f'{p[i]}\t\\pm {dp[i]}')

    dof = len(x_data) - len(p)
    p_value = 1 - gammaincc(chi_sq[-1]/2, dof/2)
    chi_min = chi_sq[-1]/dof

    with open(f'./results/{name}.txt', 'w+') as f:
        for i in range(len(p)):
            f.write(f'{round(p[i]*1000, 2)}\t\\pm {round(dp[i]*1000, 2)}\n')

        f.write(f'chi_sq_min_dof = {chi_min}\n')
        f.write(f'p_value = {p_value}\n')

    x_model = np.linspace(x_data[0], x_data[-1], 500)

    x = sp.symbols('x')
    la_mod = model(x, *p)
    la_mod = sp.lambdify(x, la_mod, modules=['numpy', {'Heaviside': np.heaviside}])
    y_model = la_mod(x_model)

    plt.figure(figsize=[10, 7])
    plt.errorbar(x_data, y_data, yerr=sigma, c='black', fmt='.k', label=f'{name}')
    plt.plot(x_model, y_model, c='red', label='Fit')

#     p, dp = np.round(p, 5), np.round(dp, 3)

    #plt.annotate(r'$M_{\rho} = $' + f'({p[0]}' + r'$\pm$' + f'{dp[0]}) GeV', (0.2, 40))
    #plt.annotate(r'$\Gamma_{\rho} = $' + f'({p[1]}' + r'$\pm$' + f'{dp[1]}) GeV', (0.2, 38))
    #plt.annotate(r'$M_{\omega} = $' + f'({p[2]}' + r'$\pm$' + f'{dp[2]}) GeV', (0.2, 36))
    #plt.annotate(r'$\Gamma_{\omega} = $' + f'({p[3]}' + r'$\pm$' + f'{dp[3]}) GeV', (0.2, 34))
    #plt.annotate(r'$\epsilon_{\omega} = $' + f'({p[4]}' + r'$\pm$' + f'{dp[4]}) GeV', (0.2, 32))
    #plt.annotate(r'$a = $' + f'({p[5]}' + r'$\pm$' + f'{dp[5]}) GeV', (0.2, 30))
    #plt.annotate(r'$b = $' + f'({p[6]}' + r'$\pm$' + f'{dp[6]}) GeV', (0.2, 28))
    #plt.annotate(r'$c = $' + f'({p[7]}' + r'$\pm$' + f'{dp[7]}) GeV', (0.2, 26))

    plt.legend(loc='best')
    plt.ylabel(r'$|F_{\pi}^V(s)|^2|$')
    plt.xlabel(r'$\sqrt{s}$[GeV]')

    plt.savefig(f'./plots/{name}.png')
    plt.close()



    #plt.figure(figsize=[10, 7])
    #plt.plot(chi_sq/dof, p_value, label=r'$\chi^2_{min}/dof = $' + f'{round(chi_sq[-1]/dof, 3)}')
    #plt.title(r'$\chi^2$'+f'-{name}')
    #plt.xlabel(r'$\chi^2/dof$')
    #plt.ylabel('p-value')
    #plt.legend(loc='best')

    #plt.savefig(f'./plots/{name}-chisq.png')
    #plt.close()
