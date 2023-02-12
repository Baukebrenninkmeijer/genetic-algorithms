import copy
import cv2
import numpy as np
from individual import Individual
from ipywidgets import Output
from IPython import display
from IPython.display import clear_output
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import permutations, product
import random


class Population:
    def __init__(
        self,
        target_img: np.ndarray,
        popsize: int = 100,
        sample_top_n: float = 0.2,
        copy_top_perc: float = 0.05,
        n_polygons: int = 10,
        mutate_delta: float = 0.05,
        penalty_rate=0.05,
        add_or_del_p: float = 0.2,
    ):
        self.popsize = popsize
        if target_img.min() < 0:
            raise ValueError(f'No colour values in target image should be under 0. Found min: {target_img.min()}')
        if target_img.max() > 1:
            raise ValueError(f'Image max colour value should be 1, but is {target_img.max()}')
        self.target_img = target_img
        self.sample_top_n = sample_top_n
        self.copy_top_perc = copy_top_perc
        self.start_poly_count = n_polygons
        self.individuals = [
            Individual(
                n_polygons=n_polygons,
                canvas_size=target_img.shape,
                mutate_delta=mutate_delta,
                penalty_rate=penalty_rate,
                add_or_del_p=add_or_del_p,
            )
            for _ in range(popsize)
        ]
        self.calc_fitness()

    def calc_fitness(self):
        for ind in self.individuals:
            ind.calc_fitness(self.target_img)

        self.individuals = sorted(self.individuals, key=lambda x: x.fitness, reverse=False)

    def combine(self):
        pairs = list(permutations(self.individuals, r=2))
        sample_pairs = random.choices(pairs, k=self.popsize)  # Sample popsize pairs out of all combinations.
        sample_pairs += list(
            permutations(self.individuals[: int(len(self.individuals) * self.sample_top_n)], r=2)
        )  # Sample 10% of pop out of pairs from the top 10%.
        children = [
            self.get_best().copy() for _ in range(max(int(self.copy_top_perc * len(self.individuals)), 3))
        ]  # Add 1/80 or 3 to pop, whichever is more.
        children += [x.crossover(y) for (x, y) in sample_pairs]
        self.individuals += children

    def mutate(self):
        for ind in self.individuals:
            ind.mutate()

    def next_gen(self):
        self.combine()
        self.mutate()
        self.calc_fitness()
        self.individuals = self.individuals[: self.popsize]
        assert len(self.individuals) <= self.popsize

    def get_best(self):
        return self.individuals[0]

    def optimize(self, n_iter: int = 1000, plot: bool = True):
        if plot:
            out = Output()
            display.display(out)

        metrics = pd.DataFrame(columns=["mean", "min", '#polygons'])

        for n in (pbar := tqdm(range(n_iter))):
            fitnesses = [x.fitness for x in self.individuals]

            new_metrics = {"mean": np.mean(fitnesses), "min": np.min(fitnesses), '#polygons': len(self.get_best().polygons)}
            metrics.loc[n, :] = new_metrics
            # pbar.write("Gen: {n}, Avg: {mean}, Best: {min}".format(n=n, **metrics))
            pbar.set_postfix(new_metrics)
            plot_every_iter = 10
            if plot:
                if n % plot_every_iter == 0:
                    with out:
                        clear_output(wait=True)
                        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
                        ax[0].imshow(self.get_best().img)
                        ax[0].set_title("Current best candidate")
                        ax[1].imshow(self.target_img)
                        ax[1].set_title("Target")
                        ax[2].set_title("MSE", fontsize=20)
                        ax[2].grid()
                        metrics.plot(y=['mean', 'min'], ax=ax[2], legend=False)
                        ax[2].set_xlabel('# iterations')
                        ax[2].set_ylabel('Loss')
                        ax[2].grid()
                        ax22 = ax[2].twinx()
                        ax22.set_ylabel('# polygons in top candidate')
                        metrics.plot(y=['#polygons'], ax=ax22, legend=False, color='g', linestyle='dashed')
                        ax[2].figure.legend(bbox_to_anchor=(0.68, 0.9), title='Metric')
                        ax[3].set_title("Fitness distribution of population.")
                        sns.histplot(x=fitnesses, bins=10, ax=ax[3])
                        ax[3].grid()
                        plt.tight_layout()
                        idx = n // plot_every_iter
                        plt.savefig(fname=f"../img/polygon_{idx:03d}.png", dpi=150)
                        plt.show()

            self.next_gen()


def main():
    target = cv2.imread("../targets/rick.jpg")
    target = cv2.resize(target, (200, 200))
    pop = Population(popsize=100, target_img=target)

    n_gen = 2000

    for n in range(n_gen):
        cv2.imwrite(
            "./out/out_{:3<0}_{:6<0}.png".format(n, pop.get_best().fitness // 1),
            pop.get_best().img,
        )

        fitnesses = []
        for ind in pop.individuals:
            fitnesses.append(ind.fitness)

        print("Gen: {}, Avg: {}, Best: {}".format(n, np.average(fitnesses).astype(int), np.min(fitnesses)))
        pop.next_gen()

    cv2.imwrite("./result.png", pop.get_best().img)


if __name__ == "__main__":
    main()
