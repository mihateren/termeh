import numpy as np
from src.trajectory_animation import TrajectoryAnimation
import matplotlib.pyplot as plt

T = np.linspace(0, 10, 1000)
R = 1

animation = TrajectoryAnimation(T, R)

plt.show()
