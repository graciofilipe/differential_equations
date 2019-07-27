# QUIZ
#
# Determine the step size h so that after
# num_points the time total_times has passed.
# Compute the trajectory of the spacecraft
# starting from a point a distance r from
# the origin with a velocity of magnitude
# equal to the speed. Use the Forward Euler
# Method. Return the distance between the final
# and the initial position in the variable
# error.

import math
import numpy as np
from scipy.integrate import solve_ivp


# These are used to keep track of the data we want to plot
h_array = []
error_array = []

total_time = 24. * 3600.  # s
g = 9.81  # m / s2
earth_mass = 5.97e24  # kg
gravitational_constant = 6.67e-11  # N m2 / kg2
radius = (gravitational_constant * earth_mass * total_time ** 2. / 4. / math.pi ** 2.) ** (1. / 3.)
speed = 2.0 * math.pi * radius / total_time


def x_prime(t, x):
    vector_to_earth = - np.array([x[0], x[1]])  # earth located at origin

    a = gravitational_constant * earth_mass / np.linalg.norm(vector_to_earth) ** 3 * vector_to_earth

    speed_x, speed_y = -speed*math.sin(2*math.pi * (t/total_time)), \
                       speed*math.cos(2*math.pi * (t/total_time))

    return [speed_x, speed_y, a[0], a[1]]


def calculate_error():
    tval = np.linspace(0, total_time, 30)
    ###Your code here.
    sol = solve_ivp(x_prime,
                    t_span=(0, total_time),
                    t_eval=tval,
                    y0=[radius, 0, speed, 0],
                    dense_output=True)
    #
    # h_array.append(h)
    # error_array.append(error)
    # return error
    return sol



sol = calculate_error()
print(sol)
import plotly.graph_objs as go
from plotly.offline import plot

data = [go.Scatter(x=sol.y[0], y=sol.y[1], name='traj'),
        go.Scatter(x=[radius*math.cos(i) for i in np.linspace(0, 2*math.pi, 10000)],
                   y=[radius*math.sin(i) for i in np.linspace(0, 2*math.pi, 10000)],
                     name='ideal')
        ]
plot(data)


