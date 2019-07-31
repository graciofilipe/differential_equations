

import numpy as np
import math
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
from plotly.offline import plot

ambient_temperature = 300. # K
flame_temperature = 1000. # K
velocity = 0.003 # m / s
dx = 0.001 # m
size = 200 # grid units
positions = dx * np.arange(size) # m
end_time = 10.0 # s

ic = np.zeros(size)+ ambient_temperature

for i in range(size):
    ic[i] += (flame_temperature - ambient_temperature) * 0.5 \
                               * (1. + math.sin(1. * 2. * math.pi * i / size))

print('ic', ic)

def x_prime(t, x):
    
    z = np.array([ambient_temperature])
    left_gain =  x[:-1]
    left_gain = np.concatenate((z, left_gain))
    right_gain = x[1:]
    right_gain = np.concatenate((right_gain, z))
    loss = 2*x
    dif = - velocity *(left_gain + right_gain - loss)
    return dif

tval = np.linspace(0, end_time, 100)

sol = solve_ivp(x_prime,
                    t_span=(0, end_time),
                    t_eval=tval,
                    y0=ic,
                    vectorized=False,
                    rtol=1e-9,
                    atol=1e-9)


t_plot = [0, 10, 20, 30]
profile_dict = {}
for t in t_plot:
    profile_dict[t] = sol.y[:,t]

fig = go.Figure()
for time, profile in profile_dict.items():
    fig.add_trace(go.Scatter(x=list(range(size)), y=profile_dict[time], name=str(time)))

plot(fig)


