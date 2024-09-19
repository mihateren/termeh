import matplotlib.pyplot as plt


class Line:
    def __init__(self, ax, color='r-', lw=1):
        self.ax = ax
        self.color = color
        self.lw = lw
        self.line, = ax.plot([], [], color, lw=lw)

    def set_data(self, x1, y1, x2, y2):
        self.line.set_data([x1, x2], [y1, y2])
