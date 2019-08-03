
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
from plotly.offline import plot
import math

bi0 = 1e24
po0 = 0.
pb0 = 0.

k_bi = 1/(5. / math.log(2.)) # days
k_po = 1/(138. / math.log(2)) # days
k_vec = [-k_bi, -k_po, 0]
end_time = 5.0 * 365. # days


ic =[bi0, po0, pb0]

def x_prime(t, x):
    bi, po, pb = x[0], x[1], x[2]
    bidot = -k_bi*bi
    podot = k_bi*bi - k_po*po
    pbdot = k_po*po

    return [bidot, podot, pbdot]


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
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[0], name='pi'))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name='po'))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[2], name='pb'))
plot(fig)


