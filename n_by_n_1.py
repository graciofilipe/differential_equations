import numpy as np
import sympy as sp
from scipy.linalg import orth





# A = [[3, 6, -1, 1, 7],
#      [1, -2, 2, 3, -1],
#      [2, -5, 5, 8, -4]]
# print('rank A', np.linalg.matrix_rank(A))
#
# A_sy = sp.Matrix(A)
# sol = sp.linsolve(A_sy)
# print(sol)

x, y, z = sp.symbols('x, y, z', real=True)
eq1 = x**2 + y**2 + (z - 4)**2 - 9
eq2 = (x - 0.2)**2 + (y + 0.6)**2 + (z - 4)**2 - 9.4
eq3 = (x - 0.2)**2 + (y - 1.7)**2 + (z - 3.5)**2 - 9.18
eq4 = (x)**2 + (y - 1.7)**2 + (z - 3.8)**2 - 10.73

system = [eq1, eq2, eq3, eq4]
from sympy.polys.polytools import is_zero_dimensional
print('zd', is_zero_dimensional(system))
non_lin_sol = sp.nonlinsolve(system, [x, y, z])
print(non_lin_sol)


A = [[1, 2, 3],
     [-1, -2, -3],
     [1, 2, 3],
     [0, 0, 9]]
print('orth', orth(A))