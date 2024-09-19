import numpy as np
import math
import matplotlib.pyplot as plt


class ArrowHead:
    def __init__(self, ax, color='b-', a_arrow=0.4):
        self.ax = ax
        self.color = color
        self.a_arrow = a_arrow
        self.arrow_line, = ax.plot([], [], color)

    def set_data(self, x, y, vx, vy):
        angle = math.atan2(vy, vx)
        ArrowX = np.array([-0.2 * self.a_arrow, 0, -0.2 * self.a_arrow])
        ArrowY = np.array([0.1 * self.a_arrow, 0, -0.1 * self.a_arrow])
        VArrowX, VArrowY = self.rotate_2d(ArrowX, ArrowY, angle)
        self.arrow_line.set_data(VArrowX + x + vx, VArrowY + y + vy)

    def rotate_2d(self, x, y, angle):
        x_new = x * np.cos(angle) - y * np.sin(angle)
        y_new = x * np.sin(angle) + y * np.cos(angle)
        return x_new, y_new
