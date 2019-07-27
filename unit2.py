import sympy as sp
import numpy as  np
import scipy as scp
from scipy.integrate import solve_ivp


A = [[0, 1, 0],
     [0, 0, 1],
     [-2, 1, 2]]

print('rank A', np.linalg.matrix_rank(A))

A_sy = sp.Matrix(A)
sol = sp.linsolve(A_sy)
print(sol)
print('')
w, v = np.linalg.eig(A)
print(w)
print('')
print(v)



from sympy.solvers import nonlinsolve
from sympy import symbols
x, y, z, a, b = symbols('x y z a b')

nls = nonlinsolve([1 - x -a*y - b*z,
                   1 - b*x -y -a*z,
                   1 - a*x- b*y -z],
                  [x, y, z])

print(nls)

k1 = 0.04
k2 = 0.02

def x_prime(t, X):
    #ax = scp.matmul(A, x)
    # b = [50*np.exp(-0.1*t), 0, 0]

    x, y, z = X[0], X[1], X[2]
    xp = [ x * (1 - x) - k1 * x * y - k2 * x * z,
           y * (1 - y) - k2 * x * y - k1 * y * z,
           z * (1 - z) - k1 * x * z - k2 * y * z,
           ]

    return xp

sol = solve_ivp(x_prime, [0, 35], [10/(k1+k2+1), 10/(k1+k2+1), 11/(k1+k2+1)])
print(sol)

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot


data = [go.Scatter(x=sol.t, y=sol.y[0], name='elephant'),
        go.Scatter(x=sol.t, y=sol.y[1], name='rhino'),
        go.Scatter(x=sol.t, y=sol.y[2], name='antelope')
        ]
plot(data)
