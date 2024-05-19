import numpy as np
import matplotlib.pyplot as plt

a = 1
b = 3
c = 1
d = 5
r = 0.006
s = 4
xbar = -1.6
Io = 3

x0 = 0.005
y0 = 0.003
z0 = 0.003

Ts = 0.1
N = 1000

def hr_model(x, y, z, a, b, c, d, r, s, xbar, Io):
    return np.array([y - a * x**3 + b * x**2 - z + Io, c - d * x**2 - y, r * (s * (x - xbar) - z)])

def euler_integration(x0, y0, z0, a, b, c, d, r, s, xbar, Io, Ts, N):
    x_values = np.zeros(N)
    y_values = np.zeros(N)
    z_values = np.zeros(N)
    x, y, z = x0, y0, z0
    for i in range(N):
        x_values[i] = x
        y_values[i] = y
        z_values[i] = z
        dx, dy, dz = hr_model(x, y, z, a, b, c, d, r, s, xbar, Io)
        x += Ts * dx
        y += Ts * dy
        z += Ts * dz
    return x_values, y_values, z_values

def rk4_integration(x0, y0, z0, a, b, c, d, r, s, xbar, Io, Ts, N):
    x_values = np.zeros(N)
    y_values = np.zeros(N)
    z_values = np.zeros(N)
    x, y, z = x0, y0, z0
    for i in range(N):
        x_values[i] = x
        y_values[i] = y
        z_values[i] = z
        k1 = hr_model(x, y, z, a, b, c, d, r, s, xbar, Io)
        k2 = hr_model(x + Ts / 2 * k1[0], y + Ts / 2 * k1[1], z + Ts / 2 * k1[2], a, b, c, d, r, s, xbar, Io)
        k3 = hr_model(x + Ts / 2 * k2[0], y + Ts / 2 * k2[1], z + Ts / 2 * k2[2], a, b, c, d, r, s, xbar, Io)
        k4 = hr_model(x + Ts * k3[0], y + Ts * k3[1], z + Ts * k3[2], a, b, c, d, r, s, xbar, Io)
        x += Ts / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        y += Ts / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        z += Ts / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    return x_values, y_values, z_values

x_euler, y_euler, z_euler = euler_integration(x0, y0, z0, a, b, c, d, r, s, xbar, Io, Ts, N)
x_rk4, y_rk4, z_rk4 = rk4_integration(x0, y0, z0, a, b, c, d, r, s, xbar, Io, Ts, N)

time = np.linspace(0, N * Ts, N)

plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plt.plot(time, x_euler, label='x - Euler')
plt.plot(time, y_euler, label='y - Euler')
plt.plot(time, z_euler, label='z - Euler')
plt.title('Euler Entegrasyonu ile HR Modeli')
plt.xlabel('Zaman')
plt.ylabel('Değerler')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, x_rk4, label='x - RK4')
plt.plot(time, y_rk4, label='y - RK4')
plt.plot(time, z_rk4, label='z - RK4')
plt.title('Runge-Kutta Entegrasyonu ile HR Modeli')
plt.xlabel('Zaman')
plt.ylabel('Değerler')
plt.legend()

plt.tight_layout()
plt.show()
