import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .point import Point
from .velocity_arrow import VelocityArrow
from .acceleration_arrow import AccelerationArrow
from .radius_vector import RadiusVector
from .curvature_radius import CurvatureRadius


class TrajectoryAnimation:
    def __init__(self, T, R):
        self.T = T
        self.R = R
        self.t = sp.Symbol('t')

        self.calculate_trajectory()
        self.prepare_plot()
        self.create_animation()

    def calculate_trajectory(self):
        self.r = sp.cos(6 * self.t)
        self.fi = self.t + 0.2 * sp.cos(3 * self.t)

        self.x = self.R * self.r * sp.cos(self.fi)
        self.y = self.R * self.r * sp.sin(self.fi)

        self.x_diff = sp.diff(self.x, self.t)
        self.y_diff = sp.diff(self.y, self.t)
        self.x_diff2 = sp.diff(self.x_diff, self.t)
        self.y_diff2 = sp.diff(self.y_diff, self.t)

        self.X, self.Y, self.VX, self.VY, self.AX, self.AY = self.calculate_values()

    def calculate_values(self):
        X = np.zeros_like(self.T)
        Y = np.zeros_like(self.T)
        VX = np.zeros_like(self.T)
        VY = np.zeros_like(self.T)
        AX = np.zeros_like(self.T)
        AY = np.zeros_like(self.T)

        for i in range(len(self.T)):
            X[i] = float(self.x.subs(self.t, self.T[i]))
            Y[i] = float(self.y.subs(self.t, self.T[i]))
            VX[i] = float(self.x_diff.subs(self.t, self.T[i]))
            VY[i] = float(self.y_diff.subs(self.t, self.T[i]))
            AX[i] = float(self.x_diff2.subs(self.t, self.T[i]))
            AY[i] = float(self.y_diff2.subs(self.t, self.T[i]))

        return X, Y, VX, VY, AX, AY

    def prepare_plot(self):
        self.fig, self.ax = plt.subplots()
        self.ax.axis('equal')
        a_lim = 0.8
        self.ax.set_xlim([min(self.X)-a_lim, max(self.X)+a_lim])
        self.ax.set_ylim([min(self.Y)-a_lim, max(self.Y)+a_lim])

        self.point = Point(self.ax)
        self.ax.plot(self.X, self.Y, 'r-', lw=1)
        self.velocity_arrow = VelocityArrow(self.ax)
        self.acceleration_arrow = AccelerationArrow(self.ax)
        self.radius_vector = RadiusVector(self.ax)
        self.curvature_radius = CurvatureRadius(self.ax)

    def create_animation(self):
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=len(self.T),
                                           init_func=self.initialize_animation, interval=20, blit=True)

    def initialize_animation(self):
        self.point.set_data([], [])
        self.velocity_arrow.set_data(0, 0, 0, 0)
        self.acceleration_arrow.set_data(0, 0, 0, 0)
        self.radius_vector.set_data(0, 0)
        self.curvature_radius.set_data(0, 0, 0, 0, 0, 0)
        return (self.point.point,
                self.velocity_arrow.line, self.velocity_arrow.arrow_head.arrow_line,
                self.acceleration_arrow.line, self.acceleration_arrow.arrow_head.arrow_line,
                self.radius_vector.line, self.radius_vector.arrow_head.arrow_line,
                self.curvature_radius.line, self.curvature_radius.arrow_head.arrow_line)

    def update(self, frame):
        self.point.set_data(self.X[frame], self.Y[frame])
        self.velocity_arrow.set_data(self.X[frame], self.Y[frame],
                                     self.VX[frame], self.VY[frame])
        self.acceleration_arrow.set_data(self.X[frame], self.Y[frame],
                                         self.AX[frame], self.AY[frame])
        self.radius_vector.set_data(self.X[frame], self.Y[frame])
        self.curvature_radius.set_data(self.X[frame], self.Y[frame],
                                       self.VX[frame], self.VY[frame],
                                       self.AX[frame], self.AY[frame])

        return (self.point.point,
                self.velocity_arrow.line, self.velocity_arrow.arrow_head.arrow_line,
                self.acceleration_arrow.line, self.acceleration_arrow.arrow_head.arrow_line,
                self.radius_vector.line, self.radius_vector.arrow_head.arrow_line,
                self.curvature_radius.line, self.curvature_radius.arrow_head.arrow_line)
