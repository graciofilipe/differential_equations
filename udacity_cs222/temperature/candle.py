

import numpy as np
import math
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
from plotly.offline import plot

ambient_temperature = 300. # K
flame_temperature = 1000. # K
coefficient = 10. # 1 / s
dx = 0.001 # m
size = 100 # grid units
positions = dx * np.arange(size) # m
h = 0.01 # s
end_time = 10.0 # s

ic = np.zeros(100)+ ambient_temperature

for i in range(int(4 * size / 10), int(5 * size / 10)):
    ic[i] = flame_temperature

print('ic', ic)

def x_prime(t, x):
    z = np.array([ambient_temperature])
    left_gain =  x[:-1]
    left_gain = np.concatenate((z, left_gain))
    right_gain = x[1:]
    right_gain = np.concatenate((right_gain, z))
    loss = 2*x

    dif = coefficient * (left_gain + right_gain - loss)


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
    profile_dict[t] = [sol.y[tile][t]  for tile in range(size)]

fig = go.Figure()
for time, profile in profile_dict.items():
    fig.add_trace(go.Scatter(x=sol.t, y=profile_dict[time], name=str(time)))

plot(fig)


