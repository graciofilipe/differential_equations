


import numpy as np
import math
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
from plotly.offline import plot

v0 = 120 * 1000 / 3600
w0 = 120 * 1000 / 3600

ic = [0, v0, w0]

mass_quarter_car = 250. # kg
mass_effective_wheel = 20. # kg
g = 9.81 # m / s2

end_time = 5. # s


def friction_coeff(slip):
    return 1.1 * (1. - math.exp(-20. * slip)) - 0.4 * slip

b_values = [130., 200.]  # m / s2

for b in b_values: # m / s2

    def x_prime(t, x):

        pos, v, w = x[0], x[1], x[2]
        v = max([0, v])
        w = max([0, w])

        lock=1
        if ((t < 1 and t >0.7) or (t < 2 and t >1.7) or (t < 3 and t > 2.7)  or
                (t < 4 and t >3.7)  or (t < 5 and t >4.7)  or (t < 6 and t >5.7) or
                (t < 7 and t > 6.7) or (t < 8 and t >7.7)  or (t < 9 and t >8.7)):
            lock=0

        if v < 1e-9:
            s = 1
        else:
            s = np.max([0, 1.0 - w/v])

        mu = friction_coeff(s)
        F = mu * mass_quarter_car * g
        vdot = - F/mass_quarter_car
        wdot = F/mass_effective_wheel - b*lock

        if w <= 0:
            wdot = 0
        if v <= 0:
            vdot = 0

        return [v, vdot, wdot]


    tval = np.linspace(0, end_time, 100)

    def stopped(t, y):
        return y[1] # stopped

    stopped.terminal = True
    stopped.direction = -1

    sol = solve_ivp(x_prime,
                    # events=stopped,
                    t_span=(0, end_time),
                    t_eval=tval,
                    y0=ic,
                    vectorized=False,
                    tol=1e-9,
                    atol=1e-9)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sol.t, y=sol.y[0], name='pos'))
    fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name='v'))
    fig.add_trace(go.Scatter(x=sol.t, y=sol.y[2], name='w'))

    plot(fig)


