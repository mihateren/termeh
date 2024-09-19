from .line import Line
from .arrow_head import ArrowHead


class AccelerationArrow(Line):
    def __init__(self, ax, color='g-', a_arrow=0.4, lw=1):
        super().__init__(ax, color, lw)
        self.arrow_head = ArrowHead(ax, color, a_arrow)

    def set_data(self, x, y, ax, ay):
        self.line.set_data([x, x + ax], [y, y + ay])
        self.arrow_head.set_data(x, y, ax, ay)
