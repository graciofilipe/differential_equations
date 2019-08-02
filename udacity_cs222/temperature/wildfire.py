
import math
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
from plotly.offline import plot

diffusion_coefficient = 5. # m2 / s

ambient_temperature = 310. # K
heat_loss_time_constant = 120. # s
velocity_x = 0.03 # m / s
velocity_y = 0.12 # m / s
ignition_temperature = 561. # K
burn_temperature = 1400. # K
burn_time_constant = 0.5 * 3600. # s
heating_value = (burn_temperature - ambient_temperature) / (
    heat_loss_time_constant * 100.) * burn_time_constant # K / (kg / m2)
slope = 0.4 # dimensionless
intercept_1 = 100. # m
intercept_2 = 170. # m
wood_1 = 100. # kg / m2
wood_2 = 70. # kg / m2

length = 650. # meters; domain extends from -length to +length
# A grid size of 50 x 50 ist much too small to see the correct result. For a better result, set the size to 200 x 200. That computation would, however, be far too long for the Web-based development environment. You may want to run it offline.
size = 50 # number of points per dimension of the grid
dx = 2. * length / size

end_time = 30. * 60. # s

def grid2physical(i, j):
    return i * dx - length + 0.5 * dx, j * dx - length + 0.5 * dx




ic_x = np.zeros([size, size])
for r in range(ic_x.shape[0]):
    for c in range(ic_x.shape[1]):
        x, y = grid2physical(r, c)
        ic_x[r,c] = (burn_temperature - ambient_temperature) * \
                math.exp(-((x + 50.) ** 2 + (y + 250.) ** 2) / (2. * 50. ** 2)) \
                + ambient_temperature
ic_x = ic_x.reshape(size*size)


ic_w = np.zeros([size, size])
for r in range(ic_w.shape[0]):
    for c in range(ic_w.shape[1]):
        x, y = grid2physical(r, c)
        intercept = y - slope*x
        w = max([wood_2, min([intercept, wood_1])])
        ic_w[r, c] = w
# import ipdb; ipdb.set_trace()
ic_w = ic_w.reshape(size*size)


ic = np.concatenate((ic_x, ic_w))


def x_prime_diffusion(x):

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

    diff = (diffusion_coefficient/dx)*(background -4*x)
    xp = diff.reshape((size*size,))

    return xp

def x_prime_heat_loss(x):
    x = x.reshape((size, size))
    prime = (ambient_temperature - x)/heat_loss_time_constant
    prime = prime.reshape((size*size))
    return prime

def x_prime_wind(x):

    x_square = x.reshape((size, size))

    z = np.zeros(size+2) + ambient_temperature

    enhanced_x = np.zeros([size+2, size+2])
    enhanced_x[:,size+1] = z
    enhanced_x[:,0] = z
    enhanced_x[size+1,:] = z
    enhanced_x[0,:] = z
    enhanced_x[1:-1, 1:-1] = x_square

    left_of = enhanced_x[1:-1, :-2]
    right_of = enhanced_x[1:-1, 2:]
    top_of = enhanced_x[:-2, 1:-1]
    bottom_of = enhanced_x[2:,1:-1]

    x_prime = - velocity_x *(right_of - left_of)/(2*dx)
    y_prime = - velocity_y *(top_of - bottom_of)/(2*dx)

    wind_prime = x_prime + y_prime
    wind_prime = wind_prime.reshape((size*size))
    return wind_prime


def x_prime_combustion(x, w):

    x_prime = np.zeros(x.shape)
    w_prime = np.zeros(w.shape)

    x_prime[x < ignition_temperature] = 0
    w_prime[x < ignition_temperature] = 0

    w_prime[x >= ignition_temperature] = -w[x >= ignition_temperature]/burn_time_constant
    x_prime[x >= ignition_temperature] = (w[x >= ignition_temperature]/burn_time_constant)*heating_value

    return {'x_prime':x_prime, 'w_prime':w_prime}

def x_prime_total(t, x_temp_and_wood):
    x, w = x_temp_and_wood[:size*size], x_temp_and_wood[size*size:]

    wind_prime = x_prime_wind(x)
    heat_loss_prime = x_prime_heat_loss(x)
    diffusion_prime = x_prime_diffusion(x)
    combustion_prime = x_prime_combustion(x, w)

    total_prime_x = 1*wind_prime + 0*heat_loss_prime + diffusion_prime + combustion_prime['x_prime']
    w_prime = combustion_prime['w_prime']

    x_and_w_prime = np.concatenate((total_prime_x, w_prime))
    return x_and_w_prime


tval = np.linspace(0, end_time, 100)

print(ic.shape)

sol = solve_ivp(x_prime_total,
                    t_span=(0, end_time),
                    t_eval=tval,
                    y0=ic,
                    vectorized=False,
                    rtol=1e-9,
                    atol=1e-9)

t_plot = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
heat_profile_dict = {}
wood_profile_dict = {}

for t in t_plot:
    x_and_w = (sol.y[:,t])
    x_only, w_only = x_and_w[:size*size], x_and_w[size*size:]
    heat_profile_dict[t] = x_only.reshape((size, size))
    wood_profile_dict[t] = w_only.reshape((size, size))


for time in heat_profile_dict.keys():
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=heat_profile_dict[time],
                             name=str(time),
                             zmin=0,
                             zmax=burn_temperature))
    plot(fig)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=wood_profile_dict[time],
                             name=str(time),
                             zmin=0,
                             zmax=111))
    plot(fig)



