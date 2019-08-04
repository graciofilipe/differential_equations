
# PROBLEM 6
#
# We will use an activator-inhibitor model to show how an initially only
# slightly random distribution of chemicals in a square grid evolves into
# a stable pattern.  The inhibitor chemical causes a reduction of both the
# activator and the inhibitor, and it diffuses quickly.  The activator,
# however, causes production of both the activator and the inhibitor, and it
# diffuses slowly.  First, insert periodic boundary conditions for the grid
# in both dimensions.  Then apply the explicit finite-difference scheme to
# the activator-inhibitor model.  Although the video shows this in 1D, please
# write your solution in 2D.



import math
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
from plotly.offline import plot

diffusion_coefficient_a = 0.0005 # m2 / s
diffusion_coefficient_b = 0.004 # m2 / s

length = 1. # meters; domain extends from 0 to length
size = 20 # number of points per dimension of the grid
dx = length / size

# Pick a time step below the threshold of instability
end_time = 3. # s

def grid2physical(i, j):
    return i * dx - length + 0.5 * dx, j * dx - length + 0.5 * dx

np.random.seed(42)
ic_a = np.random.normal(0., 0.03, [size, size])
ic_a = ic_a.reshape(size*size)
ic_b = np.random.normal(0., 0.03, [size, size])
ic_b = ic_b.reshape(size*size)

ic = np.concatenate((ic_a, ic_b))


def x_prime_diffusion(x):

    x = x.reshape((size, size))

    #circular boundries
    x_extended = np.zeros([size+2, size+2])
    x_extended[1:-1, 1:-1] = x
    x_extended[0,1:-1] = x[-1,:]
    x_extended[-1,1:-1] = x[0,:]
    x_extended[1:-1, 0] = x[:,-1]
    x_extended[1:-1, -1] = x[:,0]

    gains = np.zeros([size+2, size+2])
    gains[:-1, :] +=  x_extended[1:, :] #right gain
    gains[1:, :] += x_extended[:-1, :] # left gain
    gains[:, 1:] +=   x_extended[:, :-1] #top gain
    gains[:, :-1] += x_extended[:, 1:] #bottom gain
    shaved_gains = gains[1:-1, 1:-1]
    diff = (shaved_gains-4*x)
    xp = diff.reshape((size*size,))

    return xp


def activator_inhibitor_prime(x):
    a, b = x[:size*size], x[size*size:]
    aprime = np.multiply((1 - a**2), a) - b
    bprime = a - 0.9*b
    return np.concatenate((aprime, bprime))

def x_prime_total(t, x):

    a, b = x[:size*size], x[size*size:]
    a_prime_diff = diffusion_coefficient_a*(1/dx)*x_prime_diffusion(a)
    b_prime_diff = diffusion_coefficient_b*(1/dx)*x_prime_diffusion(b)
    diffusion_prime = np.concatenate((a_prime_diff, b_prime_diff))
    ab_activator_inhibitor_prime =activator_inhibitor_prime(x)
    total_prime = diffusion_prime + ab_activator_inhibitor_prime
    return total_prime



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
a_profile_dict = {}
b_profile_dict = {}

for t in t_plot:
    x_and_w = (sol.y[:,t])
    x_only, w_only = x_and_w[:size*size], x_and_w[size*size:]
    a_profile_dict[t] = x_only.reshape((size, size))
    b_profile_dict[t] = w_only.reshape((size, size))


for time in a_profile_dict.keys():
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=a_profile_dict[time],
                             name=str(time)))
    plot(fig)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=b_profile_dict[time],
                             name=str(time)))
    plot(fig)



