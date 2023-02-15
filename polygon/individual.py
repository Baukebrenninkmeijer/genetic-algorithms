import copy
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=3, suppress=True)


class Polygon:
    def __init__(
        self,
        canvas_size: Tuple = (200, 200, 3),
        mutate_delta: float = 0.05,
        mutate_p: float = 0.2,
        n_points: int = 4,
    ):
        self.canvas_size = canvas_size
        self.color_c = canvas_size[-1]
        self.mutate_d = mutate_delta
        self.y_size, self.x_size, self.z_size = canvas_size
        self.mutate_p = mutate_p

        self.n_points = n_points
        self.coords = []
        for _ in range(n_points):
            x = np.random.randint(0, canvas_size[1])
            y = np.random.randint(0, canvas_size[0])
            self.coords.append([x, y])

        self.color = np.random.random(self.color_c)
        self.size = -1
        self.calc_size()

    def mutate(self):
        changed = False
        for coord in self.coords:
            copy_ = coord[:]
            if self.mutate_p > np.random.random():
                changed = True
                coord[0] += np.random.randint(0, int(self.x_size * self.mutate_d)) - int(
                    self.x_size * self.mutate_d * 0.5
                )
                coord[0] = np.clip(coord[0], 0, self.x_size)
                coord[1] += np.random.randint(0, self.y_size * self.mutate_d) - (self.y_size * self.mutate_d * 0.5)
                coord[1] = np.clip(coord[1], 0, self.y_size)

        if self.mutate_p > np.random.random():
            self.color += (np.random.random(self.color_c) * self.mutate_d) - (self.mutate_d * 0.5)
            self.color = np.clip(self.color, 0, 1)

        if changed:
            assert (0 <= self.color).all() and (self.color <= 1.0).all()
            for x, y in self.coords:
                assert 0 <= x <= self.x_size
                assert 0 <= y <= self.y_size
            self.calc_size()

    def calc_size(self):
        canvas = np.zeros(self.canvas_size)
        pts = np.array(self.coords, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        col = (1, 1, 1)
        cv2.fillPoly(canvas, [pts], col)
        self.size = np.sum(canvas)

    def draw(self):
        canvas = np.ones(self.canvas_size, dtype=np.float32)
        pts = np.array(self.coords, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts], self.color)
        plt.imshow(canvas)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_points={self.n_points}, color={self.color}, coords={self.coords})"


class Individual:
    def __init__(
        self,
        n_polygons,
        canvas_size=(200, 200),
        color_channels: int = 4,
        mutate_delta: float = 0.05,
        mutate_p: float = 0.2,
        penalty_rate: float = 0.05,
        add_or_del_p: float = 0.05,
    ):
        self.n_polygons = n_polygons
        self.fitness = -1
        self.canvas_size = canvas_size
        self.img: np.ndarray = None
        self.color_channels = color_channels
        self.mutate_delta = mutate_delta
        self.penalty_rate = penalty_rate
        self.add_or_del_p = add_or_del_p
        self.mutate_p = mutate_p

        self.polygons = [
            Polygon(
                canvas_size=canvas_size,
                mutate_delta=mutate_delta,
                mutate_p=mutate_p,
            )
            for _ in range(n_polygons)
        ]

        self.draw()

    def crossover(self, other):
        child = copy.deepcopy(self)
        for idx, poly in enumerate(other.polygons):
            if 0.5 < np.random.random():
                try:
                    child.polygons[idx] = poly
                except:
                    child.polygons.append(poly)
        return child

    def mutate(self):
        for polygon in self.polygons:
            polygon.mutate()

        # deletion chance
        if np.random.random() < self.add_or_del_p and len(self.polygons) > 1:
            idx = np.random.randint(0, len(self.polygons) - 1)
            self.polygons.pop(idx)

        # addition chance
        if np.random.random() < self.add_or_del_p:
            self.polygons.append(
                Polygon(
                    canvas_size=self.canvas_size,
                    mutate_delta=self.mutate_delta,
                    mutate_p=self.mutate_p,
                )
            )

    def calc_fitness(self, target):
        self.draw()

        err = ((self.img - target) ** 2).mean()

        # apply penalties
        penalty_ratio = 1 + (len(self.polygons) * self.penalty_rate)
        err *= penalty_ratio

        # fitness is MSE
        self.fitness = err

    def draw(self):
        # sort polygons by size
        self.polygons.sort(key=lambda x: x.size, reverse=True)

        canvas = np.ones(self.canvas_size, dtype=np.float32)

        for polygon in self.polygons:
            pts = np.array(polygon.coords, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            col = polygon.color

            overlay = canvas
            output = canvas

            alpha = 0.5
            cv2.fillPoly(overlay, [pts], col)
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, canvas)

        self.img = canvas

    def get_image(self) -> np.ndarray:
        return self.img

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_polys={len(self.polygons)}, fitness={self.fitness:.1f})"
