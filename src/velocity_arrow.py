from .line import Line
from .arrow_head import ArrowHead


class VelocityArrow(Line):
    def __init__(self, ax, arrow_color='b-', line_color='b-', a_arrow=0.4, lw=1):
        super().__init__(ax, line_color, lw)
        self.arrow_head = ArrowHead(ax, arrow_color, a_arrow)

    def set_data(self, x, y, vx, vy):
        self.line.set_data([x, x + vx], [y, y + vy])
        self.arrow_head.set_data(x, y, vx, vy)
