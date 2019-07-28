
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
from plotly.offline import plot


sh0 = 1e8
ih0 = 0
sm0 = 1e10 - 1e6
im0 = 1e6
ic =[sh0, ih0, sm0, im0]
total_humans = 1e8

bites_per_day_and_mosquito = 0.1  # 1 / (day*m)
transmission_probability_mosquito_to_human = 0.3  # probability
transmission_probability_human_to_mosquito = 0.5  # probability
human_recovery_time = 70.0  # days
mosquito_lifetime = 10.0  # days
death_rate = 1/mosquito_lifetime
mosquito_br = (sm0 + im0) * death_rate
bite_reduction_by_net = 0.9  # probability

end_time = 60.0



def x_prime(t, x):
    sh, ih, sm, im = x[0], x[1], x[2], x[3]
    shdot = (1/human_recovery_time)*ih - transmission_probability_mosquito_to_human*im*bites_per_day_and_mosquito*sh/total_humans
    ihdot = transmission_probability_mosquito_to_human*im*bites_per_day_and_mosquito - (1/human_recovery_time)*ih
    smdot = -sm*death_rate + mosquito_br - transmission_probability_human_to_mosquito*sm*bites_per_day_and_mosquito*ih/total_humans
    imdot = -im*death_rate + transmission_probability_human_to_mosquito*sm*bites_per_day_and_mosquito*ih/total_humans
    return [shdot, ihdot, smdot, imdot]


tval = np.linspace(0, end_time, 100)
sol = solve_ivp(x_prime,
                    t_span=(0, end_time),
                    t_eval=tval,
                    y0=ic,
                    vectorized=False,
                    rtol=1e-4,
                    atol=1e-2)

print(sol)

fig = go.Figure()
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[0], name='s'))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name='i'))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[2], name='i'))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[3], name='i'))

plot(fig)


