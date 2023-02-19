from __future__ import annotations
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)


class PixelIndividual:
    """Individual with pixel grid for genes."""

    def __init__(
        self, shape: Tuple, genes: np.ndarray | None = None, mutate_d: float = 0.05, mutate_p: float = 0.2
    ):
        assert 0 <= mutate_d <= 1
        assert 0 <= mutate_p <= 1
        self.shape = shape
        self.mutate_d = mutate_d
        self.mutate_p = mutate_p
        if genes is None:
            self.genes = (np.random.rand(*shape)).astype(np.float32)
        else:
            self.genes = genes

    def show(self):
        plt.imshow(self.genes)
        plt.show()

    def crossover(self, other) -> PixelIndividual:
        filter = np.random.randint(low=0, high=2, size=self.genes.shape)
        output = np.where(filter, self.genes, other.genes)
        return PixelIndividual(shape=self.shape, genes=output)
        return PixelIndividual(shape=self.shape, genes=np.mean([self.genes, other.genes], axis=0))

    def mutate(self):
        if self.mutate_p > np.random.rand():
            self.mutation = ((np.random.rand(*self.shape) * self.mutate_d) - (self.mutate_d * 0.5)).astype(
                np.float32
            )
            self.genes = np.clip(self.genes + self.mutation, a_min=0.0, a_max=1.0)

    def compute_fitness(self, target):
        assert self.genes.dtype.name == 'float32', f'genes dtype should be float32 but found {self.genes.dtype.name}.'
        self.fitness = ((self.genes - target) ** 2).mean()

    def copy(self) -> PixelIndividual:
        return PixelIndividual(shape=self.shape, genes=self.genes)

    def get_image(self) -> np.ndarray:
        return self.genes

    def __repr__(self):
        return f'{self.__class__.__name__}(shape={self.shape}, fitness={self.fitness:.4f})'
