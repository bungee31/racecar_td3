# Imports
import numpy as np
import shapely.geometry as sg
import matplotlib.pyplot as plt
import math
from descartes import PolygonPatch


# Defining racetract class
class racetrack(object):

    def __init__(self, num_turns, seed, max_len=5, min_len=1, max_angle=3 * math.pi / 4, min_angle=-3 * math.pi / 4):
        # Setting seed
        np.random.seed(seed)

        self.num_turns = num_turns
        self.max_len = max_len
        self.min_len = min_len
        self.max_angle = max_angle
        self.min_angle = min_angle

        self.init_point = (0, 0)

    def calc_new_point(self, prev_point, prev_angle):

        # Pick random angle and length
        theta = np.random.uniform(self.min_angle, self.max_angle) + prev_angle
        len = np.random.uniform(self.min_len, self.max_len)

        # Calculating new point from previous point
        x = prev_point[0] + len * math.cos(theta)
        y = prev_point[1] + len * math.sin(theta)

        return (x, y), theta

    def generate(self, plot=False):

        # Initializing list of points
        points = [self.init_point]
        angles = [0]

        # Looping over turns
        for i in range(self.num_turns):
            prev_point = points[-1]
            prev_angle = angles[-1]
            new_point, new_angle = self.calc_new_point(prev_point, prev_angle)
            points.append(new_point)
            angles.append(new_angle)

        # Generating the track
        track = sg.LineString(points)
        outer = track.buffer(1.5)
        inner = outer.buffer(-0.5)
        racetrack = outer - inner

        if plot:
            xs = [a[0] for a in points]
            ys = [a[1] for a in points]
            fig = plt.figure(figsize=(10, 10))
            ax = fig.gca()
            # ax.plot(xs, ys, 'ro')
            ax.plot([points[-1][0]], [points[-1][1]], 'ro')
            ax.plot([0], [0], 'bo')
            ax.add_patch(PolygonPatch(racetrack, alpha=0.5, zorder=2))
            plt.show()

        return racetrack, points[-1]
