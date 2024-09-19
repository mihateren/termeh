import numpy as np
from .arrow_head import ArrowHead


class CurvatureRadius:
    def __init__(self, ax, color='m--', a_arrow=0.4, lw=1):
        self.ax = ax
        self.line, = ax.plot([], [], color, lw=lw)
        self.arrow_head = ArrowHead(ax, color, a_arrow)

    def set_data(self, x, y, vx, vy, ax, ay):
        numerator = (vx**2 + vy**2)**1.5
        denominator = np.abs(vx * ay - vy * ax)
        if denominator != 0:
            R = numerator / denominator
        else:
            R = np.inf

        norm_vx, norm_vy = -vy, vx
        norm = np.sqrt(norm_vx**2 + norm_vy**2)
        if norm != 0:
            norm_vx /= norm
            norm_vy /= norm

        center_x = x + R * norm_vx
        center_y = y + R * norm_vy

        self.line.set_data([x, center_x], [y, center_y])

        self.arrow_head.set_data(x, y, center_x - x, center_y - y)
