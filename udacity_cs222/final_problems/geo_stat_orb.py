# PROBLEM 3
#
# A rocket orbits the earth at an altitude of 200 km, not firing its engine.  When
# it crosses the negative part of the x-axis for the first time, it turns on its
# engine to increase its speed by the amount given in the variable called boost and
# then releases a satellite.  This satellite will ascend to the radius of
# geostationary orbit.  Once that altitude is reached, the satellite briefly fires
# its own engine to to enter geostationary orbit.  First, find the radius and speed
# of the initial circular orbit.  Then make the the rocket fire its engine at the
# proper time.  Lastly, enter the value of boost that will send the satellite
# into geostationary orbit.
#


import math
import numpy as np
from scipy.integrate import solve_ivp


# These are used to keep track of the data we want to plot
h_array = []
error_array = []

period = 24. * 3600.  # s
earth_mass = 5.97e24 # kg
earth_radius = 6.378e6 # m (at equator)
gravitational_constant = 6.67e-11 # m3 / kg s2

total_time = 9. * 3600. # s
marker_time = 0.25 * 3600. # s

# Task 1: Use Section 2.2 and 2.3 to determine the speed of the inital circular orbit.
initial_radius = earth_radius + 200*1000
initial_speed = np.sqrt(earth_mass*gravitational_constant/initial_radius)
final_radius = 42164e3

boost_time = initial_radius*math.pi/initial_speed

# Task 3: Which is the appropriate value for the boost in velocity? 2.453, 24.53, 245.3 or 2453. m/s?
# Change boost to the correct value.

boost = 245.3 # m / s

ic = [initial_radius, 0, 0, initial_speed]

def x_prime(t, x):
    vector_to_earth = -np.array([x[0], x[1]])  # earth located at origin
    a = gravitational_constant * earth_mass / np.linalg.norm(vector_to_earth) ** 3 * vector_to_earth
    speed_x, speed_y = x[2], x[3]
    vec = [speed_x, speed_y, a[0], a[1]]
    return vec


def integrate_until(ti, tf, ic, x_prime_fun):
    tval = np.linspace(ti, tf, 111)
    sol = solve_ivp(x_prime_fun,
                    t_span=(ti, tf),
                    t_eval=tval,
                    y0=ic,
                    vectorized=False,
                    rtol=1e-6,
                    atol=1e-6)
    return sol


ic1 = [initial_radius, 0, 0, initial_speed]
sol1 = integrate_until(ti=0, tf=boost_time , ic=ic1, x_prime_fun=x_prime)

ic2 = sol1.y[:, 110]
v = ic2[2], ic2[3]
new_v = v + boost * np.array(v/np.linalg.norm(v))
ic2[2], ic2[3] = new_v[0], new_v[1]

sol2 = integrate_until(ti=boost_time , tf=total_time, ic=ic2, x_prime_fun=x_prime)


import plotly.graph_objs as go
from plotly.offline import plot

data = [go.Scatter(x=[earth_radius*math.cos(i) for i in np.linspace(0, 2*math.pi, 10000)],
                   y=[earth_radius*math.sin(i) for i in np.linspace(0, 2*math.pi, 10000)],
                     name='earth'),
        go.Scatter(x=sol1.y[0], y=sol1.y[1], name='pre boost'),
        go.Scatter(x=sol2.y[0], y=sol2.y[1], name='post boost')]
plot(data)
