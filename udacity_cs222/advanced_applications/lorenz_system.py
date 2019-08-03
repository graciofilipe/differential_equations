

import numpy as np
import math
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
from plotly.offline import plot

end_time = 50.
sigma = 10.
beta = 8. / 3.
rho = 28.

def x_prime(t, x_vec):
    x, y, z = x_vec[0], x_vec[1], x_vec[2]
    x_p = sigma*(y - x)
    y_p = x*(rho-z) - y
    z_p = x*y - beta*z

    return np.array([x_p,
                     y_p,
                     z_p])

tval = np.linspace(0, end_time, 1000)

ic1 = [0, 0.3, 40]
ic2 = [0, 0.300000000000001, 40]
ic_list = [ic1, ic2]
sol_list = []
for ic in ic_list:

    sol = solve_ivp(x_prime,
                        t_span=(0, end_time),
                        t_eval=tval,
                        y0=ic,
                        vectorized=False,
                        rtol=1e-6,
                        atol=1e-6)

    sol_list.append(sol)

sol1, sol2 = sol_list[0], sol_list[1]
distance = []
for t in range(len(tval)):
    pos1 = sol1.y[:,t]
    pos2 = sol2.y[:,t]
    distance.append(np.linalg.norm(pos1-pos2))

fig = go.Figure()
fig.add_trace(go.Scatter(x=tval, y=np.log(distance), name=str('ere')))

plot(fig)


