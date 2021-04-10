
import numpy as np
import sympy as sp
from scipy.linalg import block_diag
from matplotlib import pyplot as plt

from method import *
from model import *

global p0;
p0 = [0.9, 0.2, 0.81, 0.04, 0.02, -1, 0.84, 1.55]   # in GeV

def main():
    # Preparation
    SND_data = np.loadtxt('../data/SND-VFF.txt')
    SND_x = SND_data[:, 0]
    SND_y = SND_data[:, 1]
    SND_statcov = (np.eye(len(SND_x)) * SND_data[:, 2])**2
    SND_relsystcov = (np.eye(len(SND_x)) * \
                      np.array([0.032 if i<=2 else 0.013 for i in range(len(SND_x))]))**2

    CMD2_data = np.loadtxt('../data/SND-VFF.txt')
    CMD2_x = CMD2_data[:, 0]
    CMD2_y = CMD2_data[:, 1]
    CMD2_statcov = (np.eye(len(CMD2_x)) * CMD2_data[:, 2])**2
    CMD2_relsystcov = []
    for i in range(len(CMD2_x)):
        if i<=43:
            CMD2_relsystcov.append(0.006)
        elif i>10 and i<=53:
            CMD2_relsystcov.append(0.007)
        else:
            CMD2_relsystcov.append(0.008)
    CMD2_relsystcov = (np.eye(len(CMD2_x)) * np.array(CMD2_relsystcov))**2

    KLOE_data = np.loadtxt('../data/KLOE-VFF.txt')
    KLOE_x = KLOE_data[:, 0]
    KLOE_y = KLOE_data[:, 1]
    KLOE_statcov = np.loadtxt('../data/KLOE-StatCov.txt')
    KLOE_relsystcov = np.loadtxt('../data/KLOE-RelSystCov.txt')

    BABAR_data = np.loadtxt('../data/BABAR-VFF.txt')
    BABAR_x = BABAR_data[:, 0]
    BABAR_y = BABAR_data[:, 1]
    BABAR_statcov = np.loadtxt('../data/BABAR-StatCov.txt')
    BABAR_relsystcov = np.loadtxt('../data/BABAR-RelSystCov.txt')

    x_data = np.block([SND_x, CMD2_x, KLOE_x, BABAR_x])
    y_data = np.block([SND_y, CMD2_y, KLOE_y, BABAR_y])
    cov_stat = block_diag(SND_statcov, CMD2_statcov, KLOE_statcov, BABAR_statcov)
    cov_relsyst = block_diag(SND_relsystcov, CMD2_relsystcov, KLOE_relsystcov, BABAR_relsystcov)
    var_str = "m_q g_q m_w g_w e_w a b c"

    # Fit
    p, dp = t0_fit(model, var_str, x_data, y_data, p0, cov_stat,\
                    cov_relsyst)


    # Plot
    x_model = np.linspace(x_data[0], x_data[-1], 500)
    x = sp.symbols('x')
    la_mod = model(x, *p)
    la_mod = sp.lambdify(x, la_mod, modules=['numpy', {'Heaviside': np.heaviside}])
    y_model = la_mod(x_model)

    plt.figure(figsize=[10, 7])

    plt.errorbar(SND_x, SND_y, yerr=np.sqrt(np.diag(SND_statcov)), fmt='.', ms=8, label='SND', c='blue', alpha=0.4)
    plt.errorbar(CMD2_x, CMD2_y, yerr=np.sqrt(np.diag(CMD2_statcov)), fmt='.', ms=8, label='CMD2', c='green', alpha=0.4)
    plt.errorbar(KLOE_x, KLOE_y, yerr=np.sqrt(np.diag(KLOE_statcov)), fmt='.', ms=8, label='KLOE', c='orange', alpha=0.4)
    plt.errorbar(BABAR_x, BABAR_y, yerr=np.sqrt(np.diag(BABAR_statcov)), fmt='.', ms=8, label='BABAR', c='red', alpha=0.4)

    plt.plot(x_model, y_model, lw=2, c='black', label='Fit')

    p, dp = np.round(p, 5), np.round(dp, 5)
    plt.annotate(r'$M_{\rho} = $' + f'({p[0]}' + r'$\pm$' + f'{dp[0]}) GeV', (0.1, 40))
    plt.annotate(r'$\Gamma_{\rho} = $' + f'({p[1]}' + r'$\pm$' + f'{dp[1]}) GeV', (0.1, 38))
    plt.annotate(r'$M_{\omega} = $' + f'({p[2]}' + r'$\pm$' + f'{dp[2]}) GeV', (0.1, 36))
    plt.annotate(r'$\Gamma_{\omega} = $' + f'({p[3]}' + r'$\pm$' + f'{dp[3]}) GeV', (0.1, 34))
    plt.annotate(r'$\epsilon_{\omega} = $' + f'({p[4]}' + r'$\pm$' + f'{dp[4]}) GeV', (0.1, 32))
    plt.annotate(r'$a = $' + f'({p[5]}' + r'$\pm$' + f'{dp[5]}) GeV', (0.1, 30))
    plt.annotate(r'$b = $' + f'({p[6]}' + r'$\pm$' + f'{dp[6]}) GeV', (0.1, 28))
    plt.annotate(r'$c = $' + f'({p[7]}' + r'$\pm$' + f'{dp[7]}) GeV', (0.1, 26))


    plt.title('t0-Multifit')
    plt.legend(loc='best')
    plt.ylabel(r'$|F_{\pi}^V(s)|^2|$')
    plt.xlabel(r'$\sqrt{s}$[GeV]')

    plt.savefig('./plots/all-fit.png')
    plt.close()

if __name__ == "__main__":
    main()
