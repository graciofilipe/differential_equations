
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
from plotly.offline import plot

h = 0.5 # days
transmission_coeff = 5e-9 # 1 / day person
latency_time = 1. # days
infectious_time = 5. # days

end_time = 60.0

s0 = 1e8 - 1e6 - 1e5
e0 = 0.
i0= 1e5
r0= 1e6
ic =[s0, e0, i0, r0]

def x_prime(t, x):
    s, e, i, r = x[0], x[1], x[2], x[3]
    sdot = -transmission_coeff * s * i
    edot = transmission_coeff * s * i - e*1/latency_time
    idot = 1/latency_time*e - (1/infectious_time)*i
    rdot = (1/infectious_time)*i
    return [sdot, edot, idot, rdot]


tval = np.linspace(0, end_time, 60)
sol = solve_ivp(x_prime,
                    t_span=(0, end_time),
                    t_eval=tval,
                    y0=ic,
                    vectorized=False,
                    rtol=1e-4,
                    atol=1e-2)

print(sol)

fig = go.Figure()
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[0], name='s'))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name='e'))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[2], name='i'))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[3], name='r'))
plot(fig)


