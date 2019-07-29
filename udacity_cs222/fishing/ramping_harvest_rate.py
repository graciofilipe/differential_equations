
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
from plotly.offline import plot

maximum_growth_rate = 0.5  # 1 / year
carrying_capacity = 2e6  # tons

maximum_harvest_rate = 0.8 * 2.5e5 # tons / year
ramp_start = 4. # years
ramp_end = 6. # years

fish0 = 2e5
end_time = 33.
ic = [fish0]

b = (1- ramp_end/ramp_start) / maximum_harvest_rate
m = -b/ramp_start

print('m, b', m, b)

def x_prime(t, x):

    if t < ramp_start:
        harvest_rate = 0
    elif t > ramp_end:
        harvest_rate = maximum_harvest_rate
    else:
        harvest_rate = m*t + b

    if x[0] <= 0:
        harvest_rate = 0

    xdot =  maximum_growth_rate*(1-x[0]/carrying_capacity)*x[0] - harvest_rate
    return [xdot]


tval = np.linspace(0, end_time, 100)

sol = solve_ivp(x_prime,
                    t_span=(0, end_time),
                    t_eval=tval,
                    y0=ic,
                    vectorized=False,
                    rtol=1e-4,
                    atol=1e-2)

fig = go.Figure()
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[0], name='s'))
plot(fig)


