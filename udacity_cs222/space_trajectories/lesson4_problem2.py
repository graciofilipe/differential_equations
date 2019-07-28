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
earth_mass = 5.97e24 # kg
earth_radius = 6.378e6 # m (at equator)
gravitational_constant = 6.67e-11 # m3 / kg s2
moon_mass = 7.35e22 # kg
moon_radius = 1.74e6 # m
moon_distance = 400.5e6 # m (actually, not at all a constant)
moon_period = 27.3 * 24.0 * 3600. # s
moon_initial_angle = math.pi / 180. * -61. # radian

total_duration = 11. * 24. * 3600. # s
tolerance = 100000. # m

boost1 = -7.04
boost2 = 10.  # m/s Change this to the correct value from the list above after everything else is done.

t1 = 101104
t2 = 212100

# Task 3: Include a retrograde rocket burn at 101104 seconds that reduces the velocity by 7.04 m/s
# and include a rocket burn that increases the velocity at 212100 seconds by the amount given in the variable called boost.
# Both velocity changes should happen in the direction of the rocket's motion at the time they occur.


def moon_position(s):
    x =  moon_distance * math.cos((2 * math.pi * s / moon_period) + 1*moon_initial_angle)
    y =  moon_distance * math.sin((2 * math.pi * s / moon_period) + 1*moon_initial_angle)
    pos = [x, y]
    return pos

def make_earth_circle():
    x = [earth_radius * math.cos(2.0*math.pi*i/100) for i in range(100)]
    y = [earth_radius * math.sin(2.0*math.pi*i/100) for i in range(100)]
    return (x, y)

def craft_acceleration(t, pos):
    vector_craft_to_earth = -np.array(pos)  # earth located at origin
    vector_moon = np.array(moon_position(t))
    vector_craft_to_moon = vector_moon - np.array(pos)
    a_earth = gravitational_constant * earth_mass / np.linalg.norm(vector_craft_to_earth) ** 3 * vector_craft_to_earth
    a_moon = gravitational_constant * moon_mass / np.linalg.norm(vector_craft_to_moon) ** 3 * vector_craft_to_moon
    total_a = a_moon + a_earth
    return total_a

def x_prime(t, x):
    a = craft_acceleration(t, [x[0], x[1]])
    speed_x, speed_y = x[2], x[3]
    vec = [speed_x, speed_y, a[0], a[1]]
    return vec


n_evals=6666
def integrate_until(ti, tf, ic, x_prime_fun):
    tval = np.linspace(ti, tf, n_evals)
    sol = solve_ivp(x_prime_fun,
                    t_span=(ti, tf),
                    t_eval=tval,
                    y0=ic,
                    vectorized=False,
                    rtol=1e-4,
                    atol=1e-2)
    return sol


#ic1 = [radius, 0, speed, 30]
ic1 = [-6.701e6, 0., 0., -10.818e3]
sol1 = integrate_until(ti=0, tf=t1, ic=ic1, x_prime_fun=x_prime)
ic2 = sol1.y[:, n_evals-1]
v = ic2[2], ic2[3]
new_v = v + boost1 * np.array(v/np.linalg.norm(v))
ic2[2], ic2[3] = new_v[0], new_v[1]

sol2 = integrate_until(ti=t1, tf=t2, ic=ic2, x_prime_fun=x_prime)
ic3 = sol2.y[:, n_evals-1]
v = ic3[2], ic3[3]
new_v = v + boost2 * np.array(v/np.linalg.norm(v))
ic3[2], ic3[3] = new_v[0], new_v[1]


sol3 = integrate_until(ti=t2, tf=total_duration, ic=ic3, x_prime_fun=x_prime)


import plotly.graph_objs as go
from plotly.offline import plot
moon_pos = [moon_position(i) for i in range(0, int(total_duration), 100)]
moonx = [x[0] for x in moon_pos]
moony = [x[1] for x in moon_pos]

fig = go.Figure()

# Create scatter trace of text labels

fig.add_trace(go.Scatter(x=sol1.y[0], y=sol1.y[1], name='pre red'))
fig.add_trace(go.Scatter(x=sol2.y[0], y=sol2.y[1], name='post red'))
fig.add_trace(go.Scatter(x=sol3.y[0], y=sol3.y[1], name='post boost'))
fig.add_trace(go.Scatter(x=moonx, y=moony, name='moon'))
earthx, earthy = make_earth_circle()
fig.add_trace(go.Scatter(x=earthx, y=earthy, name='earth'))

plot(fig)


