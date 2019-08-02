

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
    x_augment = np.concatenate((z, x, z))
    left_of =  x_augment[:-2]
    right_of= x_augment[2:]
    dif = - velocity *(right_of - left_of)/(2*dx)
    dif[0] = 0
    dif[-1] = 0

    return dif

tval = np.linspace(0, end_time, 100)

sol = solve_ivp(x_prime,
                    t_span=(0, end_time),
                    t_eval=tval,
                    y0=ic,
                    vectorized=False,
                    rtol=1e-12,
                    atol=1e-12)


t_plot = [0, 10, 20, 30]
profile_dict = {}
for t in t_plot:
    profile_dict[t] = sol.y[:,t]

fig = go.Figure()
for time, profile in profile_dict.items():
    fig.add_trace(go.Scatter(x=list(range(size)), y=profile_dict[time], name=str(time)))

plot(fig)


