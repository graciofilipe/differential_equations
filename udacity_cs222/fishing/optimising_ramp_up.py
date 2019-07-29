
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
from plotly.offline import plot

maximum_growth_rate = 0.5  # 1 / year
carrying_capacity = 2e6  # tons

maximum_harvest_rate = 0.8 * 2.5e5 # tons / year
ramp_start = 4. # years
ramp_end = 6. # years

fish0 = 2e5
end_time = 15
ic = [fish0]




def get_harvest_rate(t, ramp_start, ramp_end):
    b = (1 - ramp_end / ramp_start) / maximum_harvest_rate
    m = -b / ramp_start
    if t < ramp_start:
        harvest_rate = 0
    elif t > ramp_end:
        harvest_rate = maximum_harvest_rate
    else:
        harvest_rate = m * t + b
    return harvest_rate

def calculate_total_harvest(ramp_start, ramp_end):

    if ramp_end <= ramp_start:
        return 0
    else:

        def x_prime(t, x):

            harvest_rate = get_harvest_rate(t=t, ramp_start=ramp_start, ramp_end=ramp_end)

            if x[0] <= 0:
                harvest_rate = 0

            xdot =  maximum_growth_rate*(1-x[0]/carrying_capacity)*x[0] - harvest_rate
            return [xdot]


        tval = np.linspace(0, int(end_time), num=int(end_time+1))

        sol = solve_ivp(x_prime,
                            t_span=(0, end_time),
                            t_eval=tval,
                            y0=ic,
                            vectorized=False,
                            rtol=1e-4,
                            atol=1e-4)

        yearly_harvest = []
        for t in tval:
            fish_available = sol.y[0][int(t)]
            hr = get_harvest_rate(t,ramp_start, ramp_end)
            yearly_harvest.append(np.min([fish_available, hr]))

        return sum(yearly_harvest)

# total_harvest = calculate_total_harvest(ramp_start, ramp_end)

from bayes_opt import BayesianOptimization
# Bounded region of parameter space
pbounds = {'ramp_start': (0, end_time), 'ramp_end': (0, end_time)}

optimizer = BayesianOptimization(
    f=calculate_total_harvest,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1)

optimizer.maximize(
     init_points=6,
     n_iter=66)

# |  61       |  6.2e+06  |  2.561    |  1.236    |

total_23 = calculate_total_harvest(ramp_start=2,  ramp_end=3)
print(total_23)
total_opt= calculate_total_harvest(ramp_start=0.6406,  ramp_end=2.674)
print(total_opt)