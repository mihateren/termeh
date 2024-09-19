import matplotlib.pyplot as plt


class Point:
    def __init__(self, ax, color='go', markersize=10):
        self.ax = ax
        self.point, = ax.plot([], [], color, markersize=markersize)

    def set_data(self, x, y):
        self.point.set_data([x], [y])
