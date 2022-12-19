"""
useful functions for preprocessing data for GLMCC, and output results
"""
import numpy as np


def spiketime_relative(spiketime_i, spiketime_j, window_size=50.0):
    t_diff = np.array([spiketime_i - t_j for t_j in spiketime_j]).flatten()
    return t_diff[np.abs(t_diff) <= window_size]
