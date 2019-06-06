# created by Sijmen van der Willik
# 2019-06-05 18:55

import copy
import cv2
import numpy as np

from Individual import Individual

start_poly_count = 3


class Population:
    def __init__(self, popsize, target_img):
        self.popsize = popsize
        self.target_img = target_img
        self.individuals = []

        for i in range(popsize):
            self.individuals.append(Individual(start_poly_count))

        self.calc_fitness()

    def calc_fitness(self):
        for ind in self.individuals:
            ind.calc_fitness(self.target_img)

        # sort the list
        self.individuals.sort(key=lambda x: x.fitness, reverse=False)

    def combine(self):
        rt = int(np.sqrt(self.popsize) - 1)
        count = 1

        for i in range(rt):
            for j in range(rt):
                new = copy.deepcopy(self.individuals[i])
                new.crossover(self.individuals[j])
                self.individuals[count] = new
                count += 1

        while count < self.popsize:
            self.individuals[count] = copy.deepcopy(self.individuals[count % 5])
            count += 1

    def mutate(self):
        # never mutate top individual to preserve current best
        count = 0
        for ind in self.individuals:
            if count == 0:
                count += 1
                continue
            ind.mutate()

    def next_gen(self):
        self.combine()
        self.mutate()
        self.calc_fitness()

    def get_best(self):
        return self.individuals[0]


if __name__ == "__main__":
    target = cv2.imread("./bird_100.png")
    pop = Population(100, target)

    n_gen = 2000

    for n in range(n_gen):
        cv2.imwrite("./out/out_{:3<0}_{:6<0}.png".format(n, pop.get_best().fitness//1), pop.get_best().img)

        fitnesses = []
        for ind in pop.individuals:
            fitnesses.append(ind.fitness)

        print("Gen: {}, Avg: {}, Best: {}".format(n, np.average(fitnesses).astype(int), np.min(fitnesses)))
        pop.next_gen()

    cv2.imwrite("./result.png", pop.get_best().img)
