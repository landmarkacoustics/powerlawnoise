# Copyright (C) 2020 by Landmark Acoustics LLC

import sympy as sp

z, w, K, N = sp.symbols("z w K N")

R = sp.Function("R")

Rp = sp.diff(R(K), K)

Rpp = sp.diff(Rp, K)

U = Rpp + 2*z*w*Rp + w**2 * R(K)

print(U)

V = sp.dsolve(U, R(K))

sp.latex(V)
