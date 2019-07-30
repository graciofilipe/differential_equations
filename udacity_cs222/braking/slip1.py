

import numpy as np
import math
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
from plotly.offline import plot

v0 = 120 * 1000 / 3600
w0 = 120 * 1000 / 3600

ic = [v0, w0]

mass_quarter_car = 250. # kg
mass_effective_wheel = 20. # kg
g = 9.81 # m / s2

end_time = 5. # s


def friction_coeff(slip):
    return 1.1 * (1. - math.exp(-20. * slip)) - 0.4 * slip

bvals = np.arange(70., 190.1, 30.)
for b in bvals: # m / s2

    def x_prime(t, x):

        v, w = x[0], x[1]
        w = max([0, w])
        s = np.max([0, 1.0 - 1.0*w/v])
        mu = friction_coeff(s)
        F = mu * mass_quarter_car * g
        vdot = - F/mass_quarter_car
        wdot = F/mass_effective_wheel - b

        if w <= 0:
            wdot = 0
        if v <= 0:
            vdot = 0

        return [vdot, wdot]


    tval = np.linspace(0, end_time, 100)

    sol = solve_ivp(x_prime,
                        t_span=(0, end_time),
                        t_eval=tval,
                        y0=ic,
                        vectorized=True,
                        rtol=1e-9,
                        atol=1e-9)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sol.t, y=sol.y[0], name='v'))
    fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name='w'))
    plot(fig)


