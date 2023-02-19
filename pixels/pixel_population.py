from __future__ import annotations
import sys

sys.path.append('..')
from lib.plot import plot_iteration
from lib.gif import create_gif
from pixel_individual import PixelIndividual
import numpy as np
import random
from itertools import permutations
from tqdm.auto import tqdm
from IPython import display
from ipywidgets import Output
import pandas as pd
from pathlib import Path
import os

np.set_printoptions(precision=3, suppress=True)


class Population:
    def __init__(
        self,
        target: np.ndarray,
        popsize: int = 20,
        mutation_delta: float = 0.1,
        mutation_prob: float = 0.1,
        sample_top_n=0.1,
        copy_top_perc=0.02,
        stop_threshold: float = -1,
    ):
        assert 0 <= popsize < 1e10, f'Popsize is probably too big, namely: {popsize}'
        assert 0 <= mutation_prob <= 1, 'Mutation chance should be in [0, 1].'
        assert 0 <= mutation_delta <= 1, 'Mutation chance should be in [0, 1].'
        assert 0 <= sample_top_n <= 1, 'Mutation chance should be in [0, 1].'
        assert 0 <= copy_top_perc <= 1, 'Mutation chance should be in [0, 1].'

        self.popsize = popsize
        self.target = target
        self.start_mutation_delta = self.mutation_delta = mutation_delta
        self.mutation_prob = mutation_prob
        self.sample_top_n = sample_top_n
        self.stop_threshold = stop_threshold
        self.copy_top_perc = copy_top_perc
        self.pop = [
            PixelIndividual(shape=self.target.shape, mutate_d=mutation_delta, mutate_p=mutation_prob)
            for _ in range(popsize)
        ]
        self.calculate_fitness()
        self.sort_population()

    def calculate_fitness(self):
        for i in self.pop:
            i.compute_fitness(self.target)

    def sort_population(self):
        self.pop = sorted(self.pop, key=lambda x: x.fitness)

    def crossover_pop(self):
        pairs = list(permutations(self.pop, r=2))
        sample_pairs = random.choices(pairs, k=self.popsize)
        sample_pairs += random.choices(
            list(permutations(self.pop[: int(len(self.pop) * self.sample_top_n)], r=2)),
            k=self.popsize
        )
        children = [
            self.pop[0].copy() for _ in range(int(self.copy_top_perc * len(self.pop)))
        ]
        children += [x.crossover(y) for (x, y) in sample_pairs]
        self.pop += children

    def adjust_mutation_delta(self, total_epochs, iteration):
        self.mutate_delta = max(self.start_mutation_delta * ((total_epochs - iteration) / total_epochs), self.start_mutation_delta / 10)
        for i in self.pop:
            i.mutate_d = self.mutate_delta

    def mutate_pop(self) -> None:
        for i in self.pop:
            i.mutate()

    def get_best(self) -> PixelIndividual:
        return self.pop[0]

    def optimize(
        self, epochs: int, plot_frequency: int = 20, name: str = 'default', plot: bool = False, show: bool = False, adjust_delta: bool = False
    ):
        self.metrics = pd.DataFrame(columns=['min_loss', 'mean_loss'])
        self.output_dir = Path(f'img/{name}')
        self.img_dir = self.output_dir / 'iters'
        os.makedirs(self.img_dir, exist_ok=True)
        if plot:
            out = Output()
            display.display(out)
        max_fitness, min_fitness = 0, 10e10
        for i in (pbar := tqdm(range(epochs))):
            self.crossover_pop()
            self.mutate_pop()
            self.calculate_fitness()
            self.sort_population()
            self.pop = self.pop[:self.popsize]
            self.adjust_mutation_delta(total_epochs=epochs, iteration=i)
            current_best_fitness = self.pop[0].fitness
            if current_best_fitness < min_fitness:
                min_fitness = current_best_fitness
            if current_best_fitness > max_fitness:
                max_fitness = current_best_fitness
            new_metric = pd.Series(
                {
                    'min_loss': current_best_fitness,
                    'mean_loss': np.mean([x.fitness for x in self.pop]),
                }
            )
            self.metrics.loc[i] = new_metric

            # Logging and viz.
            pbar.set_postfix(
                {'Max': f'{max_fitness:.1f}', 'Min': f'{min_fitness:.3f}', 'current': f'{current_best_fitness:.3f}'}
            )
            pbar.refresh()

            if plot:
                plot_iteration(self, plot_freq=plot_frequency, out=out, min_fit=min_fitness, i=i, show=show)
            if current_best_fitness < self.stop_threshold:
                pbar.write(f'Early stopping criterion met after {i} iterations. Stopping search and creating gif.')
                create_gif(src_dir=self.img_dir, fp_out=self.output_dir / 'training.gif')
                break

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(popsize={self.popsize}, \n'
            + ', '.join(map(str, self.pop[:2] + self.pop[-2:]))
            + ')'
        )
