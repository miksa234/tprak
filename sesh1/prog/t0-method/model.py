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
    part3 = (1 + a*s + b*s**2 + c*s**3)**2
    return part1 * part2 * part3

