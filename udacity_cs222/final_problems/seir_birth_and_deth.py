
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
from plotly.offline import plot

contacts_per_day = 8. # 1 / day
transmission_probability = 0.8 # dimensionless
latency_time = 7. # days
infectious_time = 7. # days
birth_rate = 0.023 / 365. # 1 / days
mortality_rate = 0.013 / 365. # 1 / days

end_time = 15. * 365. # days

s0 = 1e8 - 1e6 - 1e5
e0 = 0.
i0= 1e5
r0= 1e6
ic =[s0, e0, i0, r0]

def x_prime(t, x):
    s, e, i, r = x[0], x[1], x[2], x[3]
    n = s+e+i+r
    sdot = -contacts_per_day*(s/n)*transmission_probability*i + birth_rate*n - mortality_rate*s
    edot = contacts_per_day*(s/n)*transmission_probability*i - e/latency_time -mortality_rate*e
    idot = 1/latency_time*e - (1/infectious_time)*i - mortality_rate*i
    rdot = (1/infectious_time)*i -mortality_rate*r
    return [sdot, edot, idot, rdot]


tval = np.linspace(0, end_time, 666)
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


