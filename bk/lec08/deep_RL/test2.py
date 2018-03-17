import numpy as np


def state_bin_to_lin(state_coord, grid_shape):
    return np.ravel_multi_index(state_coord, grid_shape)
def state_lin_to_bin(state_lin, grid_shape):
    return np.unravel_index(state_lin, grid_shape)
def obs_bin_to_lin(obs_bin):
    return np.ravel_multi_index(obs_bin, [2,2,2,2])
print(state_bin_to_lin([3,1],[18,18]))
print(state_lin_to_bin(55,[18,18]))
print(obs_bin_to_lin([1,1,1,1]))