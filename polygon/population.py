import sys

sys.path.append('..')
import os
import random
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from individual import Individual
from IPython import display
from ipywidgets import Output
from tqdm.auto import tqdm

from lib.gif import create_gif
from lib.plot import plot_iteration


class Population:
    def __init__(
        self,
        target: np.ndarray,
        popsize: int = 100,
        sample_top_n: float = 0.2,
        copy_top_perc: float = 0.05,
        n_polygons: int = 10,
        mutate_d: float = 0.05,
        mutate_p: float = 0.1,
        penalty_rate: float = 0.05,
        add_or_del_p: float = 0.2,
        stop_threshold: float = 0,
    ) -> None:
        self.popsize = popsize
        if target.min() < 0:
            raise ValueError(f'No colour values in target image should be under 0. Found min: {target.min()}')
        if target.max() > 1:
            raise ValueError(f'Image max colour value should be 1, but is {target.max()}')

        assert 0 < popsize < 1e10, f'Popsize is probably too big, namely: {popsize}'
        assert 0 <= mutate_p <= 1, 'Mutation chance should be in [0, 1].'
        assert 0 <= mutate_d <= 1, 'Mutation chance should be in [0, 1].'
        assert 0 <= sample_top_n <= 1, 'Mutation chance should be in [0, 1].'
        assert 0 <= copy_top_perc <= 1, 'Mutation chance should be in [0, 1].'

        self.target = target
        self.mutate_p = mutate_p
        self.mutate_d = mutate_d
        self.sample_top_n = sample_top_n
        self.copy_top_perc = copy_top_perc
        self.start_poly_count = n_polygons
        self.stop_threshold = stop_threshold
        self.pop = [
            Individual(
                n_polygons=n_polygons,
                canvas_size=target.shape,
                mutate_delta=mutate_d,
                mutate_p=mutate_p,
                penalty_rate=penalty_rate,
                add_or_del_p=add_or_del_p,
            )
            for _ in range(popsize)
        ]
        self.calc_fitness()

    def calc_fitness(self):
        for ind in self.pop:
            ind.calc_fitness(self.target)

        self.pop = sorted(self.pop, key=lambda x: x.fitness, reverse=False)

    def combine(self):
        pairs = list(permutations(self.pop, r=2))
        sample_pairs = random.choices(pairs, k=self.popsize)  # Sample popsize pairs out of all combinations.
        sample_pairs += list(
            permutations(self.pop[: int(len(self.pop) * self.sample_top_n)], r=2)
        )  # Sample 10% of pop out of pairs from the top 10%.
        children = [
            self.get_best().copy() for _ in range(max(int(self.copy_top_perc * len(self.pop)), 2))
        ]  # Add 1/80 or 3 to pop, whichever is more.
        children += [x.crossover(y) for (x, y) in sample_pairs]
        self.pop += children

    def mutate(self):
        for ind in self.pop:
            ind.mutate()

    def next_gen(self):
        self.combine()
        self.mutate()
        self.calc_fitness()
        self.pop = self.pop[: self.popsize]
        assert len(self.pop) <= self.popsize

    def get_best(self):
        return self.pop[0]

    def optimize(
        self, n_iter: int = 1000, plot: bool = True, plot_freq: int = 10, dir_name: str = 'default', show: bool = False
    ):
        if plot:
            out = Output()
            display.display(out)

        self.output_dir = Path(f'img/{dir_name}')
        self.img_dir = self.output_dir / 'iters'
        os.makedirs(self.img_dir, exist_ok=True)

        self.metrics = pd.DataFrame(columns=["mean", "min", '#polygons'])

        for n in (pbar := tqdm(range(n_iter))):
            fitnesses = [x.fitness for x in self.pop]

            new_metrics = {
                "mean": np.mean(fitnesses),
                "min": np.min(fitnesses),
                '#polygons': len(self.get_best().polygons),
            }
            self.metrics.loc[n, :] = new_metrics
            # pbar.write("Gen: {n}, Avg: {mean}, Best: {min}".format(n=n, **metrics))
            pbar.set_postfix(new_metrics)
            if plot:
                plot_iteration(
                    self,
                    plot_freq=plot_freq,
                    out=out,
                    min_fit=new_metrics['min'],
                    i=n,
                    show=show,
                    left_y=['mean', 'min'],
                    right_y=['#polygons'],
                )
            if self.stop_threshold and new_metrics['min'] < self.stop_threshold:
                pbar.write(f'Early stopping criterion met after {n} iterations. Stopping search and creating gif.')
                create_gif(src_dir=self.img_dir, fp_out=self.output_dir / 'training.gif')
                break

            self.next_gen()


def main():
    target = cv2.imread("../targets/rick.jpg")
    target = cv2.resize(target, (200, 200))
    pop = Population(popsize=100, target=target)

    n_gen = 2000

    for n in range(n_gen):
        cv2.imwrite(
            "./out/out_{:3<0}_{:6<0}.png".format(n, pop.get_best().fitness // 1),
            pop.get_best().img,
        )

        fitnesses = []
        for ind in pop.pop:
            fitnesses.append(ind.fitness)

        print("Gen: {}, Avg: {}, Best: {}".format(n, np.average(fitnesses).astype(int), np.min(fitnesses)))
        pop.next_gen()

    cv2.imwrite("./result.png", pop.get_best().img)


if __name__ == "__main__":
    main()
