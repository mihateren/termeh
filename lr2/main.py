import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

fig = plt.figure()
gr = fig.add_subplot(1, 1, 1)
gr.axis('equal')

# Дано:
a = b = l0 = 1
DE = 2 * a

# Задаю функции phi(t) и psi(t)

step = 3000

t = np.linspace(0, 10, step)
phi = 2 * np.sin(6 * t)
psi = 5 * t + 0.2 * np.cos(6 * t)


# Балка DE
Xd = 0
Yd = 0

Xe = Xd + DE * np.cos(phi)
Ye = Yd + DE * np.sin(phi)

balkaDE = gr.plot([Xd, Xe[0]], [Yd, Ye[0]], color='black', linewidth=5)[0]
pD = gr.plot(Xd, Yd, marker='o', color='r')[0]
pE = gr.plot(Xe, Ye, marker='o', color='r')[0]

# Пружина

Xc = DE
Yc = l0

pC = gr.plot(Xc, Yc, marker='o', color='r')[0]


def get_spring(coils, width, start, end):
    start, end = np.array(start).reshape((2,)), np.array(end).reshape((2,))
    len = np.linalg.norm(np.subtract(end, start))
    u_t = np.subtract(end, start) / len
    u_n = np.array([[0, -1], [1, 0]]).dot(u_t)
    spring_coords = np.zeros((2, coils + 2))
    spring_coords[:, 0], spring_coords[:, -1] = start, end
    normal_dist = np.sqrt(max(0, width ** 2 - (len ** 2 / coils ** 2))) / 2
    for i in np.arange(1, coils + 1):
        spring_coords[:, -i] = (start
                                + ((len * (2 * i - 1) * u_t) / (2 * coils))
                                + (normal_dist * (-1) ** i * u_n))
    return spring_coords[0, 2:], spring_coords[1, 2:]


pS = gr.plot(*get_spring(70, 0.1, [Xe[0], Ye[0]], [Xc, Yc]), color='black')[0]

# Стержень AB

Xa = Xd + DE / 2 * np.cos(phi)
Ya = Yd + DE / 2 * np.sin(phi)

Xb = Xa + b * np.cos(psi - np.pi / 2)
Yb = Ya + b * np.sin(psi - np.pi / 2)

sterjenAB = gr.plot([Xa[0], Xb[0]], [Ya[0], Yb[0]],
                    color='black', linewidth=1)[0]
pA = gr.plot(Xa, Ya, marker='o', color='r')[0]
pB = gr.plot(Xb, Yb, marker='o', color='black', markersize=20)[0]


def run(i):
    balkaDE.set_data([Xd, Xe[i]], [Yd, Ye[i]])
    pE.set_data(Xe[i], Ye[i])
    pS.set_data(*get_spring(70, 0.1, [Xe[i], Ye[i]], [Xc, Yc]))
    pA.set_data(Xa[i], Ya[i])
    pB.set_data(Xb[i], Yb[i])
    sterjenAB.set_data([Xa[i], Xb[i]], [Ya[i], Yb[i]])


anim = FuncAnimation(fig, run, frames=step, interval=1)

plt.show()
