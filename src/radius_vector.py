from .line import Line
from .arrow_head import ArrowHead


class RadiusVector(Line):
    def __init__(self, ax, color='y-', a_arrow=0.4, lw=1):
        super().__init__(ax, color, lw)
        self.arrow_head = ArrowHead(ax, color, a_arrow)

    def set_data(self, x, y):
        self.line.set_data([0, x], [0, y])
        self.arrow_head.set_data(0, 0, x, y)
