from numpy.linalg import norm
import numpy as np
import path_approximator


def binary_search(array, target):
    lower = 0
    upper = len(array)
    while lower < upper:  # use < instead of <=
        x = lower + (upper - lower) // 2
        val = array[x]
        if target == val:
            return x
        elif target > val:
            if lower == x:  # these two are the actual lines
                break  # you're looking for
            lower = x
        elif target < val:
            upper = x


class SliderPath:
    def __init__(self, path_type, control_points, expected_distance=None):
        self.controlPoints = control_points
        self.pathType = path_type
        self.expectedDistance = expected_distance

        self.calculatedPath = None
        self.cumulativeLength = None

        self.isInitialised = None

        self.ensure_initialised()

    def get_control_points(self):
        self.ensure_initialised()
        return self.controlPoints

    def get_distance(self):
        self.ensure_initialised()
        return 0 if len(self.cumulativeLength) == 0 else self.cumulativeLength[-1]

    def get_path_to_progress(self, path, p0, p1):
        self.ensure_initialised()

        d0 = self.progress_to_distance(p0)
        d1 = self.progress_to_distance(p1)

        path.clear()

        i = 0
        while i < len(self.calculatedPath) and self.cumulativeLength[i] < d0:
            i += 1

        path.append(self.interpolate_vertices(i, d0))

        while i < len(self.calculatedPath) and self.cumulativeLength[i] < d1:
            path.append(self.calculatedPath[i])
            i += 1

        path.append(self.interpolate_vertices(i, d1))

    def position_at(self, progress):
        self.ensure_initialised()

        d = self.progress_to_distance(progress)
        return self.interpolate_vertices(self.index_of_distance(d), d)

    def ensure_initialised(self):
        if self.isInitialised:
            return
        self.isInitialised = True

        self.controlPoints = [] if self.controlPoints is None else self.controlPoints
        self.calculatedPath = []
        self.cumulativeLength = []

        self.calculate_path()
        self.calculate_cumulative_length()

    def calculate_subpath(self, sub_control_points):
        if self.pathType == "Linear":
            return path_approximator.approximate_linear(sub_control_points)
        elif self.pathType == "PerfectCurve":
            if len(self.get_control_points()) != 3 or len(sub_control_points) != 3:
                return path_approximator.approximate_bezier(sub_control_points)

            subpath = path_approximator.approximate_circular_arc(sub_control_points)

            if len(subpath) == 0:
                return path_approximator.approximate_bezier(sub_control_points)

            return subpath
        elif self.pathType == "Catmull":
            return path_approximator.approximate_catmull(sub_control_points)
        else:
            return path_approximator.approximate_bezier(sub_control_points)

    def calculate_path(self):
        self.calculatedPath.clear()

        start = 0
        end = 0

        for i in range(len(self.get_control_points())):
            end += 1

            if i == len(self.get_control_points()) - 1 or (self.get_control_points()[i] == self.get_control_points()[i + 1]).all():
                cp_span = self.get_control_points()[start:end]

                for t in self.calculate_subpath(cp_span):
                    if len(self.calculatedPath) == 0 or (self.calculatedPath[-1] != t).any():
                        self.calculatedPath.append(t)

                start = end

    def calculate_cumulative_length(self):
        length = 0

        self.cumulativeLength.clear()
        self.cumulativeLength.append(length)

        for i in range(len(self.calculatedPath) - 1):
            diff = self.calculatedPath[i + 1] - self.calculatedPath[i]
            d = norm(diff)

            if self.expectedDistance is not None and self.expectedDistance - length < d:
                self.calculatedPath[i + 1] = self.calculatedPath[i] + diff * (self.expectedDistance - length) / d
                del self.calculatedPath[i + 2:len(self.calculatedPath) - 2 - i]

                length = self.expectedDistance
                self.cumulativeLength.append(length)
                break

            length += d
            self.cumulativeLength.append(length)

        if self.expectedDistance is not None and length < self.expectedDistance and len(self.calculatedPath) > 1:
            diff = self.calculatedPath[-1] - self.calculatedPath[-2]
            d = norm(diff)

            if d <= 0:
                return

            self.calculatedPath[-1] += diff * (self.expectedDistance - 1) / d
            self.cumulativeLength[-1] = self.expectedDistance

    def index_of_distance(self, d):
        i = binary_search(self.cumulativeLength, d)
        if i < 0:
            i = ~i

        return i

    def progress_to_distance(self, progress):
        return np.clip(progress, 0, 1) * self.get_distance()

    def interpolate_vertices(self, i, d):
        if len(self.calculatedPath) == 0:
            return np.zeros([2])

        if i <= 0:
            return self.calculatedPath[0]
        if i >= len(self.calculatedPath):
            return self.calculatedPath[-1]

        p0 = self.calculatedPath[i - 1]
        p1 = self.calculatedPath[i]

        d0 = self.cumulativeLength[i - 1]
        d1 = self.cumulativeLength[i]

        if np.isclose(d0, d1):
            return p0

        w = (d - d0) / (d1 - d0)
        return p0 + (p1 - p0) * w


if __name__ == "__main__":
    path = SliderPath("Bezier", 100 * np.array([[0, 0], [1, 1], [1, -1], [2, 0], [2, 0], [3, -1], [2, -2]]))
    p = np.vstack(path.calculatedPath)
    print(p.shape)

    import matplotlib.pyplot as plt

    plt.axis('equal')
    plt.plot(p[:, 0], p[:, 1], color="green")
    plt.show()
