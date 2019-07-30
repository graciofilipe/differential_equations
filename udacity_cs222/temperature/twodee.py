

import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
from plotly.offline import plot

ambient_temperature = 300. # K
flame_temperature = 1000. # K
thermal_diffusivity = 10. * 0.001 ** 2 # m2 / s
dx = 0.001 # m
size = 50 # grid units
positions = dx * np.arange(size) # m
end_time = 666 # s

ic = np.zeros([size, size])+ ambient_temperature

for iy in range(int(4 * size / 10), int(5 * size / 10)):
    for ix in range(int(4 * size / 10), int(5 * size / 10)):
        ic[iy, ix] = flame_temperature

print(ic[20:30, 20:30])

def x_prime(t, x):

    x = x.reshape((size, size))
    z = np.zeros(size) + ambient_temperature

    left_gain =  x[:,:-1]
    right_gain = x[:,1:]
    top_gain = x[1:,:]
    bottom_gain = x[:-1,:]

    background = np.zeros([size, size])

    background[:,1:] += left_gain
    background[:,:-1] += right_gain
    background[:-1,:] += top_gain
    background[1:,:] += bottom_gain

    background[:,size-1] = 4*z
    background[:,0] = 4*z
    background[size-1,:] = 4*z
    background[0,:] = 4*z



    diff = (thermal_diffusivity/dx)*(background -4*x)
    xp = diff.reshape((size*size,))

    return xp

tval = np.linspace(0, end_time, 100)

sol = solve_ivp(x_prime,
                    t_span=(0, end_time),
                    t_eval=tval,
                    y0=ic.reshape((size*size,)),
                    vectorized=False,
                    rtol=1e-9,
                    atol=1e-9)


t_plot = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
profile_dict = {}
for t in t_plot:
    heat = (sol.y[:,t]).reshape((size, size))
    profile_dict[t] = heat
    print(heat[20:30, 20:30])
    print('\n \n')

for time, profile in profile_dict.items():
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=profile_dict[time],
                             name=str(time),
                             zmin=0,
                             zmax=1000))
    plot(fig)


