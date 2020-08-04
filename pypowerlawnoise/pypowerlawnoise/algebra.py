# Copyright (C) 2020 by Landmark Acoustics LLC

import sympy as sp
import numpy as np
from matplotlib import pyplot as plt

K, K_check, N, alpha = sp.symbols(r'K \check{K} N alpha')
A, B, C, epsilon = sp.symbols(r'A B C epsilon')

D = A * sp.log(N) + B * sp.log(N) / K + C

D_check = D.subs(K, K_check) - D.subs(K, sp.sqrt(N))

K_expression = sp.simplify(sp.solve(sp.Eq(D_check, epsilon), K_check)[0])

kay_predictor = sp.lambdify((B, N, epsilon), K_expression)

print(sp.latex(K_expression))

alphas = np.linspace(-2, 2, 401)
b = sp.symbols(r'b_{\alpha\\,0:10}')
def_better = alpha * sum([b[i+1]*(alpha**i - (-2)**i) for i in (1, 2, 4, 6)])
bee_coefs = np.array([-0.0069256872,
                      0.0029379664,
                      -0.0007799595,
                      0.0002673209])

bee_predictor = sp.lambdify(alpha,
                            def_better.subs([(b[i], bc) for
                                             i, bc in
                                             zip([2, 3, 5, 7], bee_coefs)]))
bees = bee_predictor(alphas)
da = 0.01

plt.grid()
plt.hlines(0, -2, 2)
for i in range(5):
    plt.plot(alphas[i:]-i*da/2, np.diff(bees, i)/da**i)

plt.ylim([-0.1, 0.1])
plt.show()


def kay_check(alpha, N, epsilon):
    return kay_predictor(np.abs(bee_predictor(alpha)),
                         N,
                         epsilon)


