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

star_0_mass = 1e30 # kg
star_1_mass = 2e30 # kg
star_2_mass = 3e30 # kg
gravitational_constant = 6.67e-11 # m3 / kg s2


end_time = 10. * 365.26 * 24. * 3600. # s

ic_positions = np.array([[1., 3., 2.], [6., -5., 4.], [7., 8., -7.]]) * 1e11
ic_positions = ic_positions.reshape(3*3)
print(ic_positions)
ic_velocities = np.array([[-2., 0.5, 5.], [7., 0.5, 2.], [-4., -0.5, -3.]]) * 1e3
ic_velocities = ic_velocities.reshape(3*3)

ic = np.concatenate((ic_positions, ic_velocities))

def a_from_x_to_y(x, y, y_mass):
    vector_x_to_y = y - x
    a = gravitational_constant * y_mass / np.linalg.norm(vector_x_to_y) ** 3 * vector_x_to_y
    return a


def x_prime(t, x):
    positions_flat = x[:9]; positions = positions_flat.reshape([3, 3])
    velocities_flat = x[9:]
    
    p0 = positions[0]
    p1 = positions[1]
    p2 = positions[2]

    a01 = a_from_x_to_y(p0, p1, star_1_mass)
    a02 = a_from_x_to_y(p0, p2, star_2_mass)
    a10 = a_from_x_to_y(p1, p0, star_0_mass)
    a12 = a_from_x_to_y(p1, p2, star_2_mass)
    a20 = a_from_x_to_y(p2, p0, star_0_mass)
    a21 = a_from_x_to_y(p2, p1, star_1_mass)

    a0 = a01 + a02
    a1 = a10 + a12
    a2 = a20 + a21

    xdot = np.concatenate((velocities_flat, a0, a1, a2))
    return xdot


tval = np.linspace(0, end_time, 111)
sol = solve_ivp(x_prime,
                t_span=(0, end_time),
                t_eval=tval,
                y0=ic,
                vectorized=False,
                rtol=1e-6,
                atol=1e-0)




import plotly.graph_objs as go
from plotly.offline import plot

x1, y1, z1 = sol.y[:3,:]
x2, y2, z2 = sol.y[3:6,:]
x3, y3, z3 = sol.y[6:9,:]
t = sol.t


fig = go.Figure()
fig.add_trace(go.Scatter3d(x=x1, y=y1, z=z1, marker=dict(size=2,  color=t,  colorscale='Viridis')))
plot(fig)

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=x2, y=y2, z=z2, marker=dict(size=4,  color=t,  colorscale='Viridis')))
plot(fig)

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=x3, y=y3, z=z3, marker=dict(size=6,  color=t,  colorscale='Viridis')))
plot(fig)


fig = go.Figure()
fig.add_trace(go.Scatter3d(x=x1, y=y1, z=z1, marker=dict(size=2,  color=t,  colorscale='Viridis')))
fig.add_trace(go.Scatter3d(x=x2, y=y2, z=z2, marker=dict(size=4,  color=t,  colorscale='Viridis')))
fig.add_trace(go.Scatter3d(x=x3, y=y3, z=z3, marker=dict(size=6,  color=t,  colorscale='Viridis')))
plot(fig)
