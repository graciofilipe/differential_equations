# Implement the control algorithm for the hydraulic brake (see Section 14 and 15 of Unit 5).
# When the wheel slip gets larger than high_slip, start to decrease the pressure;
# when the wheel slip gets smaller than low_slip, start to increase the pressure.
# Increasing/decreasing the pressure leads to a rate of change of hydraulic_speed in the
# strength b of the brake (as measured in deceleration of the velocity of the rim of the
# wheel if there was no friction force from the road.)
# The strength of the brake must always lie between 0 and max_brake.


import numpy as np
import math
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
from plotly.offline import plot


v0 = 120 * 1000 / 3600
w0 = 120 * 1000 / 3600
b0 =
ic = [0, v0, w0, b0]

h = 0.001 # s
mass_quarter_car = 250. # kg
mass_effective_wheel = 20. # kg
g = 9.81 # m / s2
max_brake = 250. # m / s2
hydraulic_speed = 3300. # m / s3
low_slip = 0.17
high_slip = 0.23

end_time = 5. # s

def friction_coeff(slip):
    return 1.1 * (1. - math.exp(-20. * slip)) - 0.4 * slip

def p_control(actual_value, target_value):
    dif = target_value - actual_value
    k = 100000
    action = k * dif
    action = max([0, min([action, 200])])
    return action


def x_prime(t, x):

    pos, v, w = x[0], x[1], x[2]
    v = max([0, v])
    w = max([0, w])

    if v < 1e-9:
        s = 1
    else:
        s = np.max([0, 1.0 - w/v])

    b = p_control(actual_value=s,
                  target_value=0.2)

    mu = friction_coeff(s)
    F = mu * mass_quarter_car * g
    vdot = - F/mass_quarter_car
    wdot = F/mass_effective_wheel - b

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
fig.add_trace(go.Scatter(x=t_list, y=b_list, name='bvals'))


plot(fig)


