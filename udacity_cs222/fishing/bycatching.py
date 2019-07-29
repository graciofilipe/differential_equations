
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
from plotly.offline import plot


maximum_growth_rate_1 = 0.5  # 1 / year
carrying_capacity_1 = 2.0e6  # tons
maximum_growth_rate_2 = 0.3  # 1 / year
carrying_capacity_2 = 1.0e6  # tons
harvest_rate = 0.8 * 2.5e5  # rate of catching both kinds of fish, tons / year

# Insert the correct value of p
p = 0.377 # fraction of bycatch, i.e. fish 2

end_time = 100.  # years
f10 = 1.3e6
f20 = 7.5e5

ic = [f10, f20]

def x_prime(t, x):
    if x[0] > 0:
        hr1=harvest_rate
    else:
        hr1=0

    if x[1] > 0:
        hr2=harvest_rate
    else:
        hr2=0

    xdot1 = maximum_growth_rate_1*(1-x[0]/carrying_capacity_1)*x[0] - hr1*(1-p)
    xdot2 = maximum_growth_rate_2*(1-x[1]/carrying_capacity_2)*x[1] - hr2*p

    return [xdot1, xdot2]


tval = np.linspace(0, end_time, int(end_time))

sol = solve_ivp(x_prime,
                    t_span=(0, end_time),
                    t_eval=tval,
                    y0=ic,
                    vectorized=False,
                    rtol=1e-6,
                    atol=1e-6)

fig = go.Figure()
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[0], name='f1'))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name='f2'))
plot(fig)


