# QUIZ
#
# Compute the total energy of the chain, where the
# 2D positions of the nodes are given by the elements
# of the array x, then return it in the variable
# energy.

import sys
import math
import numpy as np

total_length = 2. # m
total_mass = 1. # kg
spring_constant = 50.0 # N / m
num_points = 20
g = 9.81 # m / s2
initial_energy=sys.float_info.max
np.random.seed(42)

rest_length = total_length / (num_points - 1) # m
mass_per_point = total_mass / num_points # kg
movable_points = num_points-2
xy = np.zeros([2, movable_points])
x0 = np.array([0])    #x0
y0= np.array([0])    #y0
xend = np.array([1.3])  #xend
yend = np.array([0.4])  #yend

xy_vec = xy.reshape([movable_points*2])

def chain_energy(xy_vec):
    x, y = xy_vec[:movable_points], xy_vec[movable_points:] # I can't move the first and last point of x and y (those are fixed
    x = np.concatenate((x0, x, xend))
    y = np.concatenate((y0, y, yend))

    spring_lengths =[]
    for i in range(num_points-1):
        spring_lengths.append(np.linalg.norm([x[i+1] - x[i], y[i+1]-y[i]]))

    spring_energy_components = [(le - rest_length)**2 for le in spring_lengths]
    total_spring_energy = sum(spring_energy_components)
    Uis = [mass_per_point*g*y[i] for i in range(num_points)]
    U = sum(Uis)
    return U + 0.5*spring_constant*total_spring_energy

from scipy.optimize import minimize

sol = minimize(chain_energy, xy_vec)
x, y = sol.x[:movable_points], sol.x[movable_points:]
x = np.concatenate((x0, x, xend))
y = np.concatenate((y0, y, yend))

print(x)
print(y)

import plotly.graph_objs as go
from plotly.offline import plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, name=str('str')))
plot(fig)