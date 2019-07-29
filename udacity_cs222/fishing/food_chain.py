
# Create a simple model of a food chain with two species of fish,
# one a predator and the other its prey.  Things to keep in mind:
#   Prey multiplies at a certain rate; DONE
#   prey is consumed by predators;
#   predators die of old age;
#   predators multiply depending on the amount of prey.
# using the following information:
#   the growth rate of the prey is 0.5/year;
#   the average lifespan of a predator is 5 years;
#   the predator and prey populations will remain constant over time once there are 5.0*10^6 tons
# of prey and 1.0*10^6 tons of predator.
# Use the Forward Euler Method to
# find out how each population changes over time.  Assuming that A, B, C, D,
# and the initial amounts of both fish are uncertain by +/-10%, determine
# which of these factors has the largest impact on the maximum amount of prey.
# Please note that food_chain should show how predator and prey
# populations change with time depending on these six factors, and the function
# should also pair each factor with the proper color.  Also, set result_up and
# result_down equal to the maximum prey values calculated when the parameters are
# at +10% and -10%, respectively.

import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
from plotly.offline import plot




end_time = 100.  # years
prey0 = 1.3e6
pred0 = 7.5e5

k1 = 5e-7
k2 = 4e-8

ic = [prey0, pred0]

def x_prime(t, x):

    prey, pred = x[0], x[1]
    preydot = 0.5*prey - pred*prey*k1
    preddot = k2*pred*prey -(1/5)*pred

    return [preydot, preddot]


tval = np.linspace(0, end_time, int(end_time))

sol = solve_ivp(x_prime,
                    t_span=(0, end_time),
                    t_eval=tval,
                    y0=ic,
                    vectorized=False,
                    rtol=1e-6,
                    atol=1e-6)

fig = go.Figure()
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[0], name='f1'))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name='f2'))
plot(fig)


