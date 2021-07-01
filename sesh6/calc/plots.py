import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sympy import *
from scipy.integrate import quad

font = {'family' : 'sans',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)
matplotlib.matplotlib_fname()

def main():
    # numpy representation
    w_00 = 1
    gamma =  w_00/20
    G_np = lambda w: 1/(-w**2 - 1j*gamma*w + w_00**2)
    w_np = np.linspace(w_00-2*gamma, w_00+2*gamma, 200)

    # sympy representation
    w = Symbol('w', real=True)
    z = Symbol('z')
    g = Symbol('g', real=True)
    w_0 = Symbol('w_0', real=True)

    G = 1/(-w**2 - 1j*g*w + w_0**2)
    G_n = np.abs(G.subs([(w, w_00), (w_0, w_00), (g, gamma)]))**2

    #equation for half maximum solve for w
    solutions = solve(Eq(1/2*1/(g*w_0)**2, abs(G)**2), w)

    so = solve(Eq(0, 1/G.subs(w, z)), z)

    a_1 = solutions[1].subs([(w_0, w_00), (g, gamma)])
    f_1 = abs(G.subs([(w, a_1), (g, gamma), (w_0, w_00)]))**2

    a_2 = solutions[3].subs([(w_0, w_00), (g, gamma)])
    f_2 = abs(G.subs([(w, a_2), (g, gamma), (w_0, w_00)]))**2
    fig, ax= plt.subplots(1, 2, figsize=[17,7])

    # Plots for |G(w)|^2 and arg(G(w))
    for i in range(len(ax)):
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].yaxis.set_ticks_position('left')
        ax[i].xaxis.set_ticks_position('bottom')

    ax[0].plot(w_np/w_00, np.abs(G_np(w_np))**2/G_n, c='black')
    ax[0].set_xlabel(r'$\frac{\omega}{\omega_0}$')
    ax[0].set_ylabel(r'$\frac{|G(\omega)|^2}{|G(\omega_0)|^2}$')
    ax[0].scatter(1, 1/(gamma*w_00)**2/G_n, c='r')
    ax[0].scatter(a_1/w_00, f_1/G_n, c='r')
    ax[0].scatter(a_2/w_00, f_2/G_n, c='r')
    ax[0].plot(np.linspace(w_00-gamma/2, w_00+gamma/2, 20)/w_00, f_1*np.ones(20)/G_n)

    ax[1].plot(w_np/w_00, np.angle(G_np(w_np))/np.pi, c='black')
    ax[1].set_xlabel(r'$\frac{\omega}{\omega_0}$')
    ax[1].set_ylabel(r'$arg(G(\omega)\frac{1}{\pi}$')

    fig.tight_layout()

    fig.savefig('section2.png')


    # CHAPTER 5
    M_rho = 0.77
    G_rho = 0.15
    M_pi = 0.14

    def F_BW(s):
        sigma = lambda x: np.sqrt(1 - 4*M_pi**2/x)
        G = G_rho* s/M_rho**2* (sigma(s)/sigma(M_rho**2))**3 * np.heaviside(s- 4*M_pi**2, 0 )
        return  M_rho**2 / (M_rho**2 - s - 1j*M_rho*G)

    s = np.linspace(0.1, 1, 200)
    delta = lambda x: np.angle(F_BW(x))

    # P.V.
    s_0 = 4*M_pi**2
    def integrand(s_, x):
        return (delta(s_) - delta(x))/(s_*(s_ - x))
    def integral(x):
        return quad(integrand, s_0, np.inf, args=(x))[0]

    I = np.vectorize(integral)

    first = s/np.pi * I(s) # first integral
    second = delta(s) * 1/np.pi * np.log(s_0/(s-s_0)) # second integral

    F = np.exp(first + second + 1j * delta(s))

    # BW - Omnes representations
    plt.figure(figsize=[10, 7])
    plt.plot(s, np.abs(F), label='Omnes', c='black')
    plt.plot(s, np.abs(F_BW(s)), label='Breit-Wigner', c='red')
    plt.legend(loc='best', fontsize=12)
    plt.xlabel(r'$s\ [GeV^2]$')
    plt.ylabel(r'$|F_\pi^V(s)|$')
    plt.savefig('omnes_bw.png')


if __name__ == '__main__':
    main()
