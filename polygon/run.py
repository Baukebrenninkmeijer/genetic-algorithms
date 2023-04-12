import sys
sys.path.append('..')
from lib.gif import create_gif
from population import *
import cv2
from individual import Individual, Polygon
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
plt.rcParams['figure.facecolor'] = 'white'


TARGET = plt.imread('../targets/fox.jpg')
TARGET = cv2.resize(TARGET, (400, 300))
TARGET = TARGET / 255
plt.imshow(TARGET)

n_iter = 1
pop = Population(popsize=40, target=TARGET, add_or_del_p=0.1, n_polygons=20, penalty_rate=5e-5, sample_top_n=0.5, copy_top_perc=0.01)
pop.optimize(n_iter=n_iter, plot=True, show=True, dir_name='rick_polygons', plot_freq=20)