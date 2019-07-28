# We need to demonstrate firing the
# boost rocket at the appropriate
# time, and show how that alters the
# trajectory of the capsule! Modify
# the apply_boost function below to
# fire the boost rocket 2 hours
# after start, increasing the speed
# by 300 m / s in the current
# direction of travel when you do.

import math
import numpy as np
from scipy.integrate import solve_ivp


# These are used to keep track of the data we want to plot
h_array = []
error_array = []

total_time = 6 * 3600.  # s
g = 9.81  # m / s2
earth_mass = 5.97e24  # kg
gravitational_constant = 6.67e-11  # N m2 / kg2
radius = (gravitational_constant * earth_mass * total_time ** 2. / 4. / math.pi ** 2.) ** (1. / 3.)
speed = 2.0 * math.pi * radius / total_time
spacecraft_mass = 30000. # kg
boost_speed = speed + 300

def x_prime(t, x):
    vector_to_earth = -np.array([x[0], x[1]])  # earth located at origin
    a = gravitational_constant * earth_mass / np.linalg.norm(vector_to_earth) ** 3 * vector_to_earth
    # speed_x, speed_y = -boost_speed*math.sin(2*math.pi * (t/total_time)), \
    #                     boost_speed*math.cos(2*math.pi * (t/total_time))
    speed_x, speed_y = x[2], x[3]
    vec = [speed_x, speed_y, a[0], a[1]]
    print('x_prime vec = ', vec)
    return vec



def integrate_until(ti, tf, ic, x_prime_fun):
    tval = np.linspace(ti, tf, 111)
    sol = solve_ivp(x_prime_fun,
                    t_span=(ti, tf),
                    t_eval=tval,
                    y0=ic,
                    vectorized=False,
                    rtol=1e-4,
                    atol=1e-2)

    return sol


#ic1 = [radius, 0, speed, 30]
ic1 = [15e6, 1e6, 2e3, 4e3]
sol1 = integrate_until(ti=0, tf=3600*2, ic=ic1, x_prime_fun=x_prime)

ic2 = sol1.y[:, 110]
v = ic2[2], ic2[3]
new_v = v + 300. * np.array(v/np.linalg.norm(v))
ic2[2], ic2[3] = new_v[0], new_v[1]

sol2 = integrate_until(ti=3600*2, tf=total_time, ic=ic2, x_prime_fun=x_prime)


import plotly.graph_objs as go
from plotly.offline import plot

data = [go.Scatter(x=sol1.y[0], y=sol1.y[1], name='pre boost'),
        go.Scatter(x=sol2.y[0], y=sol2.y[1], name='post boost')]
plot(data)


