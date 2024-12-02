import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# Создание массива времени
T = np.linspace(0, 10, 1000)
R = 1
t = sp.Symbol('t')

# Параметрические уравнения траектории
r = sp.cos(6 * t)
fi = t + 0.2 * sp.cos(3 * t)

# Координаты точки на плоскости
x = R * r * sp.cos(fi)
y = R * r * sp.sin(fi)

# Первая производная по времени (скорость)
x_diff = sp.diff(x, t)
y_diff = sp.diff(y, t)

# Вторая производная по времени (ускорение)
x_diff2 = sp.diff(x_diff, t)
y_diff2 = sp.diff(y_diff, t)

# Инициализация массивов для хранения значений
X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
AX = np.zeros_like(T)
AY = np.zeros_like(T)

# Вычисление значений координат, скорости и ускорения в каждый момент времени
for i in range(len(T)):
    X[i] = float(x.subs(t, T[i]))
    Y[i] = float(y.subs(t, T[i]))
    VX[i] = float(x_diff.subs(t, T[i]))
    VY[i] = float(y_diff.subs(t, T[i]))
    AX[i] = float(x_diff2.subs(t, T[i]))
    AY[i] = float(y_diff2.subs(t, T[i]))

# Настройка графика анимации
fig, ax = plt.subplots()
ax.axis('equal')
a_lim = 0.8
ax.set_xlim([min(X) - a_lim, max(X) + a_lim])
ax.set_ylim([min(Y) - a_lim, max(Y) + a_lim])

# Инициализация графических элементов
point, = ax.plot([], [], 'go', markersize=10)
ax.plot(X, Y, 'r-', lw=1)
velocity_line, = ax.plot([], [], 'b-', lw=1)
velocity_arrow_head, = ax.plot([], [], 'b-')
acceleration_line, = ax.plot([], [], 'g-', lw=1)
acceleration_arrow_head, = ax.plot([], [], 'g-')
radius_vector_line, = ax.plot([], [], 'y-', lw=1)
radius_vector_arrow_head, = ax.plot([], [], 'y-')
curvature_radius_line, = ax.plot([], [], 'm--', lw=1)
curvature_radius_arrow_head, = ax.plot([], [], 'm--')


def rotate_2d(x_arr, y_arr, angle):
    x_new = x_arr * np.cos(angle) - y_arr * np.sin(angle)
    y_new = x_arr * np.sin(angle) + y_arr * np.cos(angle)
    return x_new, y_new


def update(frame):
    x0 = X[frame]
    y0 = Y[frame]
    vx = VX[frame]
    vy = VY[frame]
    ax0 = AX[frame]
    ay0 = AY[frame]

    point.set_data([x0], [y0])

    velocity_line.set_data([x0, x0 + vx], [y0, y0 + vy])
    angle_v = math.atan2(vy, vx)
    arrow_x = np.array([-0.08, 0, -0.08])
    arrow_y = np.array([0.04, 0, -0.04])
    VArrowX, VArrowY = rotate_2d(arrow_x, arrow_y, angle_v)
    velocity_arrow_head.set_data(VArrowX + x0 + vx, VArrowY + y0 + vy)

    acceleration_line.set_data([x0, x0 + ax0], [y0, y0 + ay0])
    angle_a = math.atan2(ay0, ax0)
    AArrowX, AArrowY = rotate_2d(arrow_x, arrow_y, angle_a)
    acceleration_arrow_head.set_data(AArrowX + x0 + ax0, AArrowY + y0 + ay0)

    radius_vector_line.set_data([0, x0], [0, y0])
    angle_r = math.atan2(y0, x0)
    RArrowX, RArrowY = rotate_2d(arrow_x, arrow_y, angle_r)
    radius_vector_arrow_head.set_data(RArrowX + x0, RArrowY + y0)

    numerator = (vx**2 + vy**2)**1.5
    denominator = abs(vx * ay0 - vy * ax0)
    if denominator != 0:
        R_curv = numerator / denominator
    else:
        R_curv = np.inf

    norm_vx = -vy
    norm_vy = vx
    norm = np.hypot(norm_vx, norm_vy)
    if norm != 0:
        norm_vx /= norm
        norm_vy /= norm

    center_x = x0 + R_curv * norm_vx
    center_y = y0 + R_curv * norm_vy

    curvature_radius_line.set_data([x0, center_x], [y0, center_y])
    angle_c = math.atan2(center_y - y0, center_x - x0)
    CArrowX, CArrowY = rotate_2d(arrow_x, arrow_y, angle_c)
    curvature_radius_arrow_head.set_data(
        CArrowX + center_x, CArrowY + center_y)

    return (point,
            velocity_line, velocity_arrow_head,
            acceleration_line, acceleration_arrow_head,
            radius_vector_line, radius_vector_arrow_head,
            curvature_radius_line, curvature_radius_arrow_head)


# Создание анимации
ani = animation.FuncAnimation(
    fig, update, frames=len(T), interval=20, blit=True)
plt.show()

# Построение графиков функций от времени
plt.figure(figsize=(12, 8))

# График координат
plt.subplot(3, 1, 1)
plt.plot(T, X, label='X(t)')
plt.plot(T, Y, label='Y(t)')
plt.title('Координаты от времени')
plt.xlabel('Время t')
plt.ylabel('Координаты X, Y')
plt.legend()
plt.grid(True)

# График компонент скорости
plt.subplot(3, 1, 2)
plt.plot(T, VX, label='Vx(t)')
plt.plot(T, VY, label='Vy(t)')
plt.title('Скорость от времени')
plt.xlabel('Время t')
plt.ylabel('Скорость Vx, Vy')
plt.legend()
plt.grid(True)

# График компонент ускорения
plt.subplot(3, 1, 3)
plt.plot(T, AX, label='Ax(t)')
plt.plot(T, AY, label='Ay(t)')
plt.title('Ускорение от времени')
plt.xlabel('Время t')
plt.ylabel('Ускорение Ax, Ay')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
