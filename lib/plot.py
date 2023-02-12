from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from ipywidgets import Output
from IPython import display
import seaborn as sns
from typing import List

def plot_iteration(pop,  plot_freq: int, out, min_fit: float, i: int, show: bool = True, left_y: List | None = None, right_y: List | None = None):
    if i % plot_freq == 0 or min_fit < pop.stop_threshold:
        with out:
            clear_output(wait=True)
            fig, ax = plt.subplots(1, 4, figsize=(20, 5))
            plt.suptitle(f'Iteration {i}, Best loss: {min_fit:.3f}')
            ax[0].imshow(pop.get_best().get_image())
            ax[0].set_title('Current best candidate')
            ax[1].imshow(pop.target)
            ax[1].set_title('Target')

            ax[2].set_title("MSE", fontsize=20)
            ax[2].grid()
            ax[2].set_xlabel('# iterations')
            ax[2].set_ylabel('Loss')
            if left_y is None:
                sns.lineplot(data=pop.metrics, ax=ax[2])
                ax[2].legend()
            elif left_y is not None:
                pop.metrics.plot(y=left_y, ax=ax[2], legend=False)
            elif right_y is not None:
                # pop.metrics.plot(y=left_y, ax=ax[2], legend=False)
                ax22 = ax[2].twinx()
                ax22.set_ylabel('# polygons in top candidate')
                pop.metrics.plot(y=right_y, ax=ax22, legend=False, color='g', linestyle='dashed')
                ax[2].figure.legend(bbox_to_anchor=(0.68, 0.9), title='Metric')

            ax[3].set_title('Fitness distribution of population.')
            sns.histplot(x=[x.fitness for x in pop.pop], bins=10, ax=ax[3])
            ax[3].grid()
            ax[3].set_ylim([0, pop.popsize])
            plt.tight_layout()
            idx = i // plot_freq
            plt.savefig(fname=pop.img_dir/f'state_{idx:03d}.png', dpi=150)
            if show:
                plt.show()
            else:
                plt.close()