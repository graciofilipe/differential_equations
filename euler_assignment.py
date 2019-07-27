def euler(dy_dx, h, y0, x0, n_steps):
    y_current = y0
    x_current = x0
    y_list = [y_current]
    x_list = [x_current]
    for _ in range(n_steps):
        x_next = x_current + h
        y_next = y_current + dy_dx(x_current, y_current)*h
        y_list.append(y_next)
        x_list.append(x_next)
        x_current = x_next
        y_current = y_next

    return x_list, y_list


def some_deriv(x, y):
    return x**2 - y**2


x, y = euler(dy_dx=some_deriv,
             h=0.1,
             y0=1,
             x0=0,
             n_steps=5)

print(x, y)

from scipy.integrate import solve_ivp

sol = solve_ivp(some_deriv, t_span=(0, 0.3), y0=[1])

print(sol)