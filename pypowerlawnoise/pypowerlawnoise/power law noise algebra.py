# Copyright (C) 2020 Landmark Acoustics LLC

from sympy import *
import numpy as np
from matplotlib import pyplot as plt


def print_equation(lhs, rhs) -> str:
    print('\\begin{equation}\n' + latex(Eq(lhs, rhs)) + '\n\\end{equation}')

# Computations for estimating the optimal degree from a statistical model
b = symbols('b_{\\alpha\\\,(:10)}')

K, N, alpha, alpha_hat, delta = symbols('K N alpha \\hat{\\alpha} delta')

E = b[1] + b[2]/N + b[3]/N**2 + b[4]/K + b[5]/(K*N) + b[6]/(K*N**2) + b[7]/(K**2) + b[8]/(K**2*N) + b[9]/(K*N)**2

E_alpha = Function('E_{\\alpha}')

print_equation(E_alpha(K, N), E)

L = Limit(E, K, oo)

B_alpha = Function('B_{\\alpha}')

print_equation(Eq(B_alpha(N), L), simplify(L.doit()))

D = simplify(E - simplify(L.doit()))

D_alpha = Function('D_{\\alpha}')

print_equation(D_alpha(K, N), D)

K_check_emp = solve(D - delta, K)[1]

K_check_alpha = Function('\\check{K}_{\\alpha}')

print_equation(K_check_alpha(N), K_check_emp)

# Computations for estimating the relationship between power and degree

# when -2 <= power <= 0

alpha, alpha_hat, p = symbols('\\alpha \\hat{\\alpha} p')

negative_k_prime = p*alpha*(alpha-alpha_hat)

negative_k_star = negative_k_prime.integrate(alpha)

negative_p = solve(negative_k_star.subs(alpha, -2) - 1, p)

negative_k_check = simplify(negative_k_star.subs(p, negative_p[0]))

K_check_pp = p * (2 * alpha - alpha_hat) / 2
K_tmp = integrate(integrate(K_check_pp, alpha), alpha)

pea = solve(K_tmp.subs(alpha, -2) - 1, p)[0]

K_check = K_tmp.subs(p, pea)

b2s = symbols('b_{2\\\,:8}')
K_2 = Symbol('\\check{K}_2')

alpha_fit = solve(K_check.subs(alpha, 2) - K_2, alpha_hat)[0]

alpha_function = simplify(alpha_fit.subs(K_2, K_check_emp.subs(zip(b, b2s))))
                                                            
