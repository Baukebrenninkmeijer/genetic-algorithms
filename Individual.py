# created by Sijmen van der Willik
# 2019-06-05 18:56

import copy

import cv2
import numpy as np

poly_mutate_chance = 0.2
poly_part_mutate_chance = 0.1
deletion_p = 0.02
addition_p = 0.08
x_size = 100
y_size = 100
copy_p = 0.25
penalty_f = 0.05


class Polygon:
    def __init__(self):
        points = np.random.randint(3, 5)
        self.coords = []
        for i in range(points):
            x = np.random.randint(0, x_size)
            y = np.random.randint(0, y_size)
            self.coords.append([x, y])

        r = np.random.randint(0, 255)
        g = np.random.randint(0, 255)
        b = np.random.randint(0, 255)

        self.color = [r, g, b]
        self.size = 0

        self.calc_size()

    def mutate(self):
        changed = False
        for coord in self.coords:
            if poly_part_mutate_chance < np.random.random():
                changed = True
                coord[0] += np.random.randint(0, x_size//5) - x_size//10
                coord[0] = np.clip(coord[0], 0, x_size)
                coord[1] += np.random.randint(0, y_size//5) - y_size//10
                coord[1] = np.clip(coord[1], 0, y_size)

        for idx in range(3):
            if poly_part_mutate_chance < np.random.random():
                changed = True
                self.color[idx] += np.random.randint(0, 10) - 5
                self.color[idx] = int(np.clip(self.color[idx], 0, 255))

        if changed:
            self.calc_size()

    def calc_size(self):
        canvas = np.zeros((100, 100, 3))
        pts = np.array(self.coords)
        pts = pts.reshape((-1, 1, 2))
        col = (0, 0, 1)
        cv2.fillPoly(canvas, [pts], col)
        self.size = np.sum(canvas)


class Individual:
    def __init__(self, start_n):
        self.start_n = start_n
        self.polygons = []
        self.fitness = -1
        self.canvas_size = (x_size, y_size)
        self.img = None

        for i in range(start_n):
            self.polygons.append(Polygon())

        self.draw()

    def crossover(self, other):
        for idx, poly in enumerate(other.polygons):
            if copy_p < np.random.random():
                try:
                    self.polygons[idx] = copy.deepcopy(poly)
                except:
                    self.polygons.append(copy.deepcopy(poly))

    def mutate(self):
        for polygon in self.polygons:
            if poly_mutate_chance < np.random.random():
                polygon.mutate()

        # deletion chance
        if np.random.random() < deletion_p and len(self.polygons) > 1:
            idx = np.random.randint(0, len(self.polygons) - 1)
            self.polygons.pop(idx)

        # addition chance
        if np.random.random() < addition_p:
            self.polygons.append(Polygon())

    def calc_fitness(self, target):
        self.draw()

        err = np.sum((self.img.astype("float") - target.astype("float")) ** 2)
        err /= float(self.img.shape[0] * target.shape[1])

        # apply penalties
        err *= (1 + len(self.polygons) * penalty_f)

        # fitness is MSE
        self.fitness = err

    def draw(self):
        # sort polygons by size
        self.polygons.sort(key=lambda x: x.size, reverse=True)

        canvas = np.zeros((y_size, x_size, 3))
        canvas += 255

        for polygon in self.polygons:
            pts = np.array(polygon.coords)
            pts = pts.reshape((-1, 1, 2))
            col = polygon.color

            overlay = canvas
            output = canvas

            alpha = 0.5
            cv2.fillPoly(overlay, [pts], col)

            cv2.addWeighted(overlay, 0.5, output, 1 - alpha,
                            0, canvas)

        self.img = canvas
