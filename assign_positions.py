import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import sys
import galsim
from math import floor



SEED = 1205802543
rng = np.random.default_rng(seed=SEED)


gal_data = pickle.load(open('iterative_run/data_file_full.pkl','rb'))


ra_grid = np.load('ra_coords.npy')
dec_grid = np.load('dec_coords.npy')

permutation = np.arange(len(ra_grid))

rng.shuffle(permutation)
ra_grid = ra_grid[permutation]
dec_grid = dec_grid[permutation]

if len(gal_data['ra']) == len(ra_grid):
    gal_data['ra'] = ra_grid
else:
    print('WARNING: ra_grid stored in ra_coords.npy does not have the same size as data_file_WL.pkl')
if len(gal_data['dec']) == len(dec_grid):
    gal_data['dec'] = dec_grid
else:
    print('WARNING: dec_grid stored in dec_coords.npy does not have the same size as data_file_WL.pkl')


pickle.dump(gal_data, open('iterative_run/data_file_gals_WL.pkl', 'wb'))

